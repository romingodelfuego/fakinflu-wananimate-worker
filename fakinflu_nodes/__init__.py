"""fakinflu — noeuds ComfyUI du worker wananimate (Wan 2.2 Animate, route remplacement).

Le worker prep (docs/plan-replace.md) fournit deja la video de controle (pose / depth / facemesh) et,
hors mode fast, un masque video du personnage. On ne refait donc ni suivi ni estimation ici : on
decode ces fichiers aux cadence et taille cibles et on les passe a WanAnimateToVideo (noeud natif
ComfyUI). L'encodage de la sortie reutilise la meme strategie que scail (H.264 par morceau, concat
ffmpeg sans reencodage) pour ne pas garder toutes les images decodees en RAM.

  FakinfluAnimatePrep  control.mp4 (+ mask.mp4 optionnel) -> (pose_video IMAGE, character_mask MASK, frames)
  FakinfluSaveSegment  morceau decode -> segment_<k>.mp4 (recouvrement retire), liste chainee des segments
  FakinfluConcatVideo  segments -> animation.mp4 (ffmpeg concat, copie de flux)
"""
import os
import subprocess

import numpy as np
import torch

import comfy.model_management
import comfy.utils
import folder_paths

CHUNK = 16  # images traitees a la fois (pic memoire borne)


def frame_indices(n_src, src_fps, fps):
    """Indices des images source gardees pour passer de src_fps a fps (sans interpolation)."""
    if not fps or fps >= src_fps:
        return list(range(n_src))
    step = src_fps / fps
    return [i for i in (int(round(k * step)) for k in range(int(n_src / step) + 1)) if i < n_src]


def _fit(images, width, height, method):
    """Recadrage centre (N,H,W,C) -> (N,height,width,C), comme ComfyUI."""
    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, method, "center").movedim(1, -1)


def decode_frames(path, keep):
    """Generateur (index, image RGB uint8 HxWx3) des seules images `keep` de la video, dans l'ordre."""
    import av  # PyAV, present dans worker-comfyui
    wanted, last = set(keep), max(keep)
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in wanted:
                yield i, frame.to_ndarray(format="rgb24")
            if i >= last:
                break


class FakinfluAnimatePrep:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": ""}),
            "src_frames": ("INT", {"default": 0, "min": 1, "max": 100000}),
            "src_fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0}),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0}),
            "width": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
        }, "optional": {"mask_path": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("pose_video", "character_mask", "frames")
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, video_path, src_frames, src_fps, fps, width, height, mask_path=""):
        idx = frame_indices(src_frames, src_fps, fps)
        dev, dt = comfy.model_management.intermediate_device(), comfy.model_management.intermediate_dtype()
        pose = torch.empty((len(idx), height, width, 3), dtype=dt, device=dev)
        pos = {src: k for k, src in enumerate(idx)}
        buf = []

        def flush(dst, frames_np, method):
            ks = [pos[i] for i, _ in buf]
            t = torch.from_numpy(np.stack(frames_np)).to(device=dev, dtype=dt) / 255.0
            dst[ks[0]:ks[-1] + 1] = _fit(t, width, height, method)

        got = 0
        for i, frame in decode_frames(video_path, idx):
            buf.append((i, frame))
            got += 1
            if len(buf) == CHUNK:
                flush(pose, [f for _, f in buf], "area")
                buf.clear()
        if buf:
            flush(pose, [f for _, f in buf], "area")
            buf.clear()
        if got != len(idx):
            raise ValueError(f"{video_path} : {got} images decodees sur {len(idx)} attendues")

        # Masque du personnage : video niveaux de gris (blanc = personne). Absent (mode fast) -> plein cadre.
        if mask_path and os.path.exists(mask_path):
            m = torch.empty((len(idx), height, width, 3), dtype=dt, device=dev)
            buf = []
            for i, frame in decode_frames(mask_path, idx):
                buf.append((i, frame))
                if len(buf) == CHUNK:
                    flush(m, [f for _, f in buf], "bilinear")
                    buf.clear()
            if buf:
                flush(m, [f for _, f in buf], "bilinear")
            character_mask = m.mean(dim=-1).clamp(0, 1)  # (N,H,W)
        else:
            character_mask = torch.ones((len(idx), height, width), dtype=dt, device=dev)
        return (pose, character_mask, len(idx))


def _out_dir(prefix):
    d = os.path.join(folder_paths.get_output_directory(), prefix)
    os.makedirs(d, exist_ok=True)
    return d


def encode(images, path, fps):
    """IMAGE (N, H, W, 3) float -> H.264 (yuv420p, crf 16), envoye par paquets a ffmpeg."""
    n, h, w, _ = images.shape
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "16", "-preset", "medium",
                          "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    try:
        for s in range(0, n, CHUNK):
            p.stdin.write((images[s:s + CHUNK].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes())
    finally:
        p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg a echoue sur {path}")


class FakinfluSaveSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "fakinflu/animation"}),
            "index": ("INT", {"default": 0, "min": 0, "max": 1000}),
            "skip": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "images de recouvrement a retirer en tete"}),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0}),
        }, "optional": {"previous": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("segments",)
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, images, filename_prefix, index, skip, fps, previous=None):
        path = os.path.join(_out_dir(filename_prefix), f"segment_{index:03d}.mp4")
        encode(images[skip:], path, fps)
        return ("\n".join([p for p in (previous or "").split("\n") if p] + [path]),)


class FakinfluConcatVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segments": ("STRING", {"forceInput": True}),
                             "filename_prefix": ("STRING", {"default": "fakinflu/animation"})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "fakinflu"

    def run(self, segments, filename_prefix):
        paths = [p for p in segments.split("\n") if p]
        d = _out_dir(filename_prefix)
        lst = os.path.join(d, "segments.txt")
        with open(lst, "w") as f:
            f.writelines(f"file '{p}'\n" for p in paths)
        out = os.path.join(d, "animation_00001_.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        "-movflags", "+faststart", out], check=True, timeout=600)
        for p in paths + [lst]:
            os.remove(p)
        return {"ui": {"text": [out]}}


NODE_CLASS_MAPPINGS = {"FakinfluAnimatePrep": FakinfluAnimatePrep,
                       "FakinfluSaveSegment": FakinfluSaveSegment, "FakinfluConcatVideo": FakinfluConcatVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"FakinfluAnimatePrep": "fakinflu wananimate : video de controle + masque",
                              "FakinfluSaveSegment": "fakinflu wananimate : encodage d'un morceau",
                              "FakinfluConcatVideo": "fakinflu wananimate : assemblage de la video"}
