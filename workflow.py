"""fakinflu — graphe ComfyUI (format API) de Wan 2.2 Animate (route remplacement, docs/plan-replace.md).

Entrees venues du worker prep : video de controle (pose / depth / facemesh) + masque du personnage
(hors mode fast) ; photo du personnage. On construit le graphe natif ComfyUI autour de
WanAnimateToVideo (comfy_extras/nodes_wan.py) :

  control.mp4 (+ mask.mp4) -> FakinfluAnimatePrep -> (pose_video IMAGE, character_mask MASK)
  ref.png -> LoadImage -> reference_image + CLIP vision
  Wan 2.2 Animate 14B (+ LoRA lightx2v distill, 6 etapes cfg 1) -> ModelSamplingSD3
  WanAnimateToVideo(pose_video, character_mask, reference_image, clip_vision) -> latent
    -> SamplerCustom -> VAEDecode -> FakinfluSaveSegment -> FakinfluConcatVideo (ffmpeg) -> animation.mp4

La video entiere est animee en PLUSIEURS passes chainees de `window` images (max_length, defaut 77 =
fenetre native du modele). Chaque passe k>=1 recoit les images de la passe k-1 (continue_motion) et
son video_frame_offset : WanAnimateToVideo rembobine de OVERLAP et lit pose/masque a partir de la
(fenetre glissante), le recouvrement etant retire via trim_image. Voir build_workflow / segments().
Le pic de RAM est borne par passe (ram_bytes), pas par la duree ; la limite restante est la VRAM
(fenetre) et JOB_TIMEOUT x nombre de passes (baisser fps si trop de passes).
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


OVERLAP = 5  # continue_motion_max_frames : images regenerees d'une passe a l'autre (doit valoir 4m+1)


def segments(n, window, overlap=OVERLAP):
    """Decoupe n images retenues en passes de `window` images chainees par continue_motion.

    Chaque passe (sauf la premiere) regenere `overlap` images deja vues (retirees ensuite via
    trim_image), donc avance de `window - overlap`. Renvoie (nombre de passes N, {index: limit})
    ou `limit` cale une passe sur ses seules images reelles (padding/recouvrement en trop coupes).
    WanAnimateToVideo gere la fenetre glissante lui-meme via video_frame_offset."""
    if n <= window:
        return 1, {0: n}  # une passe ; `limit` coupe le padding si n < window
    stride = window - overlap
    N = 1 + -(-(n - window) // stride)  # 1 + ceil((n - window) / stride)
    remaining = n - (window + (N - 2) * stride)  # images reelles de la derniere passe (1..stride)
    return N, {N - 1: remaining}


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
    window = _length(p, n)
    if window <= OVERLAP:
        raise ValueError(f"video de controle trop courte : {n} images retenues")
    N, limits = segments(n, window)

    # Noeuds partages par toutes les passes : prep (pose + masque decodes une fois), modeles, prompts.
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
    wf["sigmas"] = {"class_type": "BasicScheduler", "inputs": {"model": ["shift", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}}

    # Une passe = WanAnimateToVideo (fenetre `window`) -> SamplerCustom -> TrimVideoLatent (retire
    # l'image de reference, sortie trim_latent) -> VAEDecode -> FakinfluSaveSegment. La passe k>=1
    # recoit les images de la passe k-1 (continue_motion) et son video_frame_offset (sortie 5) :
    # WanAnimateToVideo rembobine de OVERLAP puis lit pose/masque a partir de la -> fenetre glissante.
    for k in range(N):
        cond, samp, trim, dec, save = (f"cond{k}", f"sample{k}", f"trim{k}", f"dec{k}", f"save{k}")
        cond_in = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                   "width": width, "height": height, "length": window, "batch_size": 1,
                   "continue_motion_max_frames": OVERLAP,
                   "video_frame_offset": 0 if k == 0 else [f"cond{k - 1}", 5],
                   "clip_vision_output": ["clip_vision_enc", 0], "reference_image": ["ref", 0],
                   "pose_video": ["prep", 0], "character_mask": ["prep", 1]}
        if k > 0:
            cond_in["continue_motion"] = [f"dec{k - 1}", 0]
        wf[cond] = {"class_type": "WanAnimateToVideo", "_meta": {"title": f"Wan 2.2 Animate ({k + 1}/{N})"}, "inputs": cond_in}
        wf[samp] = {"class_type": "SamplerCustom", "_meta": {"title": f"generation ({k + 1}/{N})"},
                    "inputs": {"model": ["shift", 0], "add_noise": True, "noise_seed": seed + k, "cfg": cfg,
                               "positive": [cond, 0], "negative": [cond, 1],
                               "sampler": ["sampler", 0], "sigmas": ["sigmas", 0], "latent_image": [cond, 2]}}
        # trim_latent (sortie 3) = images de reference en tete du latent, a retirer avant decodage.
        wf[trim] = {"class_type": "TrimVideoLatent", "inputs": {"samples": [samp, 1], "trim_amount": [cond, 3]}}
        wf[dec] = {"class_type": "VAEDecode", "_meta": {"title": f"decodage ({k + 1}/{N})"}, "inputs": {"samples": [trim, 0], "vae": ["vae", 0]}}
        # trim_image (sortie 4) = images de recouvrement a retirer en tete (0 a la premiere passe).
        save_in = {"images": [dec, 0], "filename_prefix": prefix, "index": k, "skip": [cond, 4], "fps": fps}
        if k > 0:
            save_in["previous"] = [f"save{k - 1}", 0]
        if k in limits:
            save_in["limit"] = limits[k]
        wf[save] = {"class_type": "FakinfluSaveSegment", "_meta": {"title": f"encodage ({k + 1}/{N})"}, "inputs": save_in}

    wf["concat"] = {"class_type": "FakinfluConcatVideo", "_meta": {"title": "assemblage"},
                    "inputs": {"segments": [f"save{N - 1}", 0], "filename_prefix": prefix}}
    return wf


def plan(params, meta):
    """Resume sans GPU (dry-run, estimation) : taille, cadence, images retenues, fenetre et passes.

    frames_out = toutes les images retenues (la video entiere, decoupee en `segments` passes
    chainees de `window` images), et non plus la seule premiere fenetre."""
    p = params or {}
    fps = float(p.get("fps") or meta["fps"])
    n = len(frame_indices(meta["frames"], meta["fps"], fps))
    w, h = size_for(meta["width"], meta["height"], int(p.get("resolution", 480)))
    window = _length(p, n)
    N = segments(n, window)[0] if window > OVERLAP else 1
    return {"width": w, "height": h, "fps": fps, "frames_in": n, "window": window, "segments": N, "frames_out": n}


def ram_bytes(plan_):
    """Pic de RAM des IMAGE gardees par ComfyUI (float32), borne par PASSE et non par la duree totale :
    pose + masque decodes une fois (frames_in) + ~2 fenetres decodees vivantes (passe courante +
    continue_motion de la precedente). Ne depend donc PAS du nombre de passes."""
    per_frame = plan_["width"] * plan_["height"] * 3 * 4
    return per_frame * (2 * plan_["frames_in"] + 2 * plan_.get("window", plan_["frames_out"]))


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


# Meta representative pour un dump « template » (valeurs lisibles dans le canvas).
_TEMPLATE_META = {"width": 480, "height": 832, "fps": 30.0, "frames": 120}
_TEMPLATE_FILES = {"control": "control.mp4", "mask": "mask.mp4", "ref": "ref.png"}

if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--validate":
        _validate(sys.argv[2], sys.argv[3], sys.argv[4])
        sys.exit(0)
    if len(sys.argv) >= 2 and sys.argv[1] == "--dump-api":
        # graphe FORMAT API seul (sans la ligne plan), pour le convertisseur API->UI.
        params = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
        print(json.dumps(build_workflow(params, "", "", _TEMPLATE_FILES, "fakinflu/animation", _TEMPLATE_META), indent=1))
        sys.exit(0)
    params = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
    print(json.dumps(plan(params, _TEMPLATE_META)))
    print(json.dumps(build_workflow(params, "prompt", "", _TEMPLATE_FILES, "fakinflu/run", _TEMPLATE_META), indent=1))
