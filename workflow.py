"""fakinflu — graphe ComfyUI (format API) de Wan 2.2 Animate (route remplacement, docs/plan-replace.md).

Entrees venues du worker prep : video de controle (pose / depth / facemesh) + masque du personnage
(hors mode fast) ; photo du personnage. On construit le graphe natif ComfyUI autour de
WanAnimateToVideo (comfy_extras/nodes_wan.py) :

  control.mp4 (+ mask.mp4) -> FakinfluAnimatePrep -> (pose_video IMAGE, character_mask MASK)
  ref.png -> LoadImage -> reference_image + CLIP vision
  Wan 2.2 Animate 14B (+ LoRA lightx2v distill, 6 etapes cfg 1) -> ModelSamplingSD3
  WanAnimateToVideo(pose_video, character_mask, reference_image, clip_vision) -> latent
    -> SamplerCustom -> VAEDecode -> FakinfluSaveSegment -> FakinfluConcatVideo (ffmpeg) -> animation.mp4

v1 : une seule passe (length = toutes les images retenues, bornee par max_length). Les clips longs
peuvent depasser la VRAM : le decoupage par morceaux (continue_motion / video_frame_offset du noeud)
est un TODO a valider sur ComfyUI reel avant de l'activer.
"""
import json, os, sys

OUTPUT_FILES = {"video": "animation_*.mp4"}
OUTPUT_NAMES = {"video": "animation.mp4"}

UNET_FILES = {"int8": "wan_animate_2_int8_convrot.safetensors", "bf16": "wan_animate_2_bf16.safetensors"}
DISTILL_LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"

DEFAULT_MAX_LENGTH = 77  # images par passe (v1 sans decoupage) ; relever avec la VRAM


def frame_indices(n_src, src_fps, fps):
    if not fps or fps >= src_fps:
        return list(range(n_src))
    step = src_fps / fps
    return [i for i in (int(round(k * step)) for k in range(int(n_src / step) + 1)) if i < n_src]


def _len4(n):
    """Longueur valide pour Wan : 4m + 1."""
    return ((n - 1) // 4) * 4 + 1 if n >= 1 else 0


def size_for(src_w, src_h, short):
    """Largeur et hauteur multiples de 16 (exigence WanAnimateToVideo), cote court = short."""
    scale = short / min(src_w, src_h)
    r16 = lambda v: max(16, int(round(v * scale / 16)) * 16)
    return r16(src_w), r16(src_h)


def _length(params, n):
    return _len4(min(n, int(params.get("max_length") or DEFAULT_MAX_LENGTH)))


def build_workflow(params, prompt, negative, files, prefix, meta):
    """params : {preset, resolution, prompt, negative, seed, precision?, distill?, steps?, cfg?, shift?, fps?, max_length?}.
    files : chemins absolus {control, mask?} (lus par FakinfluAnimatePrep) et nom dans input/ {ref}.
    meta : {width, height, fps, frames} de la video de controle."""
    p = params or {}
    precision = p.get("precision", "int8")
    if precision not in UNET_FILES:
        raise ValueError(f"precision inconnue : {precision} ({'/'.join(UNET_FILES)})")
    distill = bool(p.get("distill", True))
    steps = int(p.get("steps") or (6 if distill else 20))
    cfg = float(p.get("cfg") or (1.0 if distill else 5.0))
    shift = float(p.get("shift", 5.0))
    seed = int(p.get("seed") if p.get("seed") is not None else 0)
    fps = float(p.get("fps") or meta["fps"])
    width, height = size_for(meta["width"], meta["height"], int(p.get("resolution", 480)))

    n = len(frame_indices(meta["frames"], meta["fps"], fps))
    length = _length(p, n)
    if length < 5:
        raise ValueError(f"video de controle trop courte : {n} images retenues")

    wf = {
        "prep": {"class_type": "FakinfluAnimatePrep", "_meta": {"title": "video de controle + masque"},
                 "inputs": {"video_path": files["control"], "mask_path": files.get("mask", ""),
                            "src_frames": int(meta["frames"]), "src_fps": float(meta["fps"]),
                            "fps": fps, "width": width, "height": height}},
        "ref": {"class_type": "LoadImage", "_meta": {"title": "photo du personnage"}, "inputs": {"image": files["ref"]}},
        "unet": {"class_type": "UNETLoader", "_meta": {"title": "chargement Wan 2.2 Animate"},
                 "inputs": {"unet_name": UNET_FILES[precision], "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "_meta": {"title": "chargement umT5"},
                 "inputs": {"clip_name": TEXT_ENCODER, "type": "wan"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "clip_vision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": CLIP_VISION}},
        "clip_vision_enc": {"class_type": "CLIPVisionEncode",
                            "inputs": {"clip_vision": ["clip_vision", 0], "image": ["ref", 0], "crop": "none"}},
        "pos": {"class_type": "CLIPTextEncode", "_meta": {"title": "prompt"}, "inputs": {"clip": ["clip", 0], "text": prompt or ""}},
        "neg": {"class_type": "CLIPTextEncode", "_meta": {"title": "prompt negatif"}, "inputs": {"clip": ["clip", 0], "text": negative or ""}},
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
    }
    model = ["unet", 0]
    if distill:
        wf["distill"] = {"class_type": "LoraLoaderModelOnly",
                         "inputs": {"model": model, "lora_name": DISTILL_LORA, "strength_model": float(p.get("distill_strength", 1.0))}}
        model = ["distill", 0]
    wf["shift"] = {"class_type": "ModelSamplingSD3", "inputs": {"model": model, "shift": shift}}

    wf["cond"] = {"class_type": "WanAnimateToVideo", "_meta": {"title": "Wan 2.2 Animate"},
                  "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                             "width": width, "height": height, "length": length, "batch_size": 1,
                             "continue_motion_max_frames": 5, "video_frame_offset": 0,
                             "clip_vision_output": ["clip_vision_enc", 0], "reference_image": ["ref", 0],
                             "pose_video": ["prep", 0], "character_mask": ["prep", 1]}}
    wf["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": ["shift", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}}
    wf["sample"] = {"class_type": "SamplerCustom", "_meta": {"title": "generation"},
                    "inputs": {"model": ["shift", 0], "add_noise": True, "noise_seed": seed, "cfg": cfg,
                               "positive": ["cond", 0], "negative": ["cond", 1],
                               "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": ["cond", 2]}}
    wf["dec"] = {"class_type": "VAEDecode", "_meta": {"title": "decodage"}, "inputs": {"samples": ["sample", 1], "vae": ["vae", 0]}}
    # trim_image (sortie 4 de WanAnimateToVideo) = images de tete a retirer (recouvrement continue_motion) ; 0 en passe unique.
    wf["save"] = {"class_type": "FakinfluSaveSegment", "_meta": {"title": "encodage"},
                  "inputs": {"images": ["dec", 0], "filename_prefix": prefix, "index": 0, "skip": ["cond", 4], "fps": fps}}
    wf["concat"] = {"class_type": "FakinfluConcatVideo", "_meta": {"title": "assemblage"},
                    "inputs": {"segments": ["save", 0], "filename_prefix": prefix}}
    return wf


def plan(params, meta):
    """Resume sans GPU (dry-run, estimation) : taille, cadence, images retenues, longueur de la passe."""
    p = params or {}
    fps = float(p.get("fps") or meta["fps"])
    n = len(frame_indices(meta["frames"], meta["fps"], fps))
    w, h = size_for(meta["width"], meta["height"], int(p.get("resolution", 480)))
    length = _length(p, n)
    return {"width": w, "height": h, "fps": fps, "frames_in": n, "segments": 1, "frames_out": length}


def ram_bytes(plan_):
    """Pic de RAM des IMAGE gardees par ComfyUI (float32) : pose + masque (frames_in) + sortie decodee."""
    per_frame = plan_["width"] * plan_["height"] * 3 * 4
    return per_frame * (2 * plan_["frames_in"] + plan_["frames_out"])


def _validate(url, control_path, ref_name):
    """Soumet le graphe a un ComfyUI (local, 0 credit) : il le valide avant toute execution.
    ref_name doit exister dans input/ ; modeles en fichiers vides."""
    import urllib.error, urllib.request
    meta = {"width": 480, "height": 832, "fps": 30.0, "frames": 120}
    wf = build_workflow({"resolution": 480}, "validation", "", {"control": control_path, "ref": ref_name}, "fakinflu/validate", meta)
    post = lambda path, body: urllib.request.urlopen(urllib.request.Request(
        url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=30)
    try:
        r = json.loads(post("/prompt", {"prompt": wf}).read())
    except urllib.error.HTTPError as e:
        print(e.read().decode()[:6000])
        sys.exit(1)
    post("/queue", {"clear": True}).close()
    post("/interrupt", {}).close()
    print(f"OK : graphe accepte par ComfyUI ({len(wf)} noeuds), prompt_id={r.get('prompt_id')}")


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--validate":
        _validate(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    params = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    meta = {"width": 480, "height": 832, "fps": 30.0, "frames": 120}
    print(json.dumps(plan(params, meta)))
    print(json.dumps(build_workflow(params, "prompt", "", {"control": "/tmp/control.mp4", "mask": "/tmp/mask.mp4", "ref": "ref.png"},
                                    "fakinflu/run", meta), indent=1))
