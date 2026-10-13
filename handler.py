#!/usr/bin/env python3
"""fakinflu — worker RunPod : wananimate (Comfy-Org/Wan-Animate-2).

Entree : {"items": [ {...}, ... ]}  (ou un item seul a plat dans "input")
Sortie : {"results": [ {...} | {"error": "..."} ]}

Plusieurs items par job = un seul cold start paye ; un item en echec ne fait
pas echouer le lot. Le modele est charge UNE fois, au cold start.
"""
import base64, io, os, time, traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # poids bakes : jamais de telechargement au runtime

import runpod
import torch
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "Comfy-Org/Wan-Animate-2")

# --- chargement unique au cold start ------------------------------------------
# TODO : charger le modele, ex. _model = Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
_model = None


def _b64_to_image(b64: str) -> Image.Image:
    if b64.strip().startswith("data:") and "," in b64[:64]:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _process(item: dict) -> dict:
    """TODO : une inference. Valider les champs, renvoyer un dict serialisable."""
    t = time.time()
    # ex. image = _b64_to_image(item["image"]); out = _model(image=image, prompt=item.get("prompt", ""))
    return {"echo": {k: v for k, v in item.items() if k != "image"},
            "stats": {"seconds": round(time.time() - t, 2)}}


def handler(job):
    inp = job.get("input", {}) or {}
    items = inp.get("items") or [inp]
    results = []
    for item in items:
        res = {"id": item.get("id")}
        try:
            res.update(_process(item))
        except Exception as e:
            traceback.print_exc()
            res["error"] = f"{type(e).__name__}: {e}"
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(res)
    return {"results": results}


runpod.serverless.start({"handler": handler})
