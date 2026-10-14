# fakinflu — worker `wananimate`

Wan 2.2 Animate 14B (ComfyUI natif `WanAnimateToVideo`), générateur de la route « remplacement de
personnage » (voir `docs/plan-replace.md`). Modèles bakés, aucun secret dans l'image.

## Entrées (contrat `contracts/wananimate.input.schema.json`)

| rôle | source | usage |
|---|---|---|
| `control` | `control_video.mp4` du worker **prep** (pose / depth / facemesh) | `pose_video` |
| `mask` (option) | `mask.mp4` du prep (hors mode fast) | `character_mask` |
| `ref` | photo du personnage | `reference_image` + CLIP vision |

`params` : `{preset: draft|final, resolution, prompt, negative, seed, precision: int8|bf16, probe}`.
Sortie : `animation.mp4`.

## Modèles (bakés, `download_models.sh`, sha256 pinnés)

Tout depuis `Comfy-Org/Wan-Animate-2` : `wan_animate_2_int8_convrot` (défaut) ou `wan_animate_2_bf16`,
`umt5_xxl_fp8_e4m3fn_scaled`, `clip_vision_h`, `Wan2_1_VAE_bf16`, LoRA `lightx2v` (distillation 6 étapes).

## Cycle de vie

```bash
make worker-bump W=wananimate [BUMP=minor] PR=1   # VERSION + worker.json dans la PR
# merge de la PR = build RunPod ; puis :
FORCE=1 make worker-release W=wananimate NOTES="..."
make wanimate-check     # /health, 0 crédit
```

Première fois : lier le dépôt à un endpoint dans la console RunPod (Serverless → New Endpoint →
Import Git Repository, branche `main`, Dockerfile `Dockerfile`), puis renseigner
`RUNPOD_WANANIMATE_ENDPOINT_ID` dans `.env`.

## À valider avant tout build payant

- **`WORKER_TAG`** (Dockerfile) doit pointer un `runpod/worker-comfyui:*-base` dont ComfyUI embarque
  `WanAnimateToVideo` : le garde-fou `grep` du Dockerfile échoue le build sinon (choisir un tag plus récent).
- **Graphe** : `python3 workflow.py --validate http://127.0.0.1:8188 <control.mp4> ref.png` contre un
  ComfyUI local (0 crédit) valide noms de nœuds et branchements avant de dépenser.
- **v1 = une seule passe** (`max_length`, défaut 77 images). Les clips longs peuvent dépasser la VRAM :
  le découpage par morceaux (`continue_motion` / `video_frame_offset`) est un TODO à valider sur GPU.
