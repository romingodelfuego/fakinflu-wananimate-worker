# fakinflu — worker `wananimate`

Wan 2.2 Animate 14B (ComfyUI natif `WanAnimateToVideo`), générateur de la route « remplacement de
personnage » (voir `docs/plan-replace.md`). Modèles bakés, aucun secret dans l'image.

## Entrées, signal `control` (contrat `contracts/wananimate.input.schema.json`)

| rôle | source | usage |
|---|---|---|
| `control` | `control_video.mp4` du worker **prep** (pose / depth / facemesh) | `pose_video` |
| `mask` (option) | `mask.mp4` du prep (hors mode fast) | `character_mask` |
| `ref` | photo du personnage | `reference_image` + CLIP vision |

`params` : `{preset: draft|final, resolution, prompt, negative, seed, precision: int8|bf16, signal: control|pose,
mask_expand, fps, max_length, probe}`.
Sortie : `animation.mp4`.

## Signal `pose` (v1.2.0)

`params.signal` : `control` (défaut, comportement v1.1.0 inchangé) ou `pose`. En `pose`, le worker fait
le **mode remplacement** natif de `WanAnimateToVideo` : la personne est remplacée **dans sa scène**.

| rôle | source | usage |
|---|---|---|
| `source` | `input.mp4` du prep (vidéo d'origine) | squelette DWPose → `pose_video` ; décor → `background_video` |
| `mask` (obligatoire) | `mask.mp4` du prep (mode `advanced` ou `bulletproof`) | `ThresholdMask` 0,5 → `GrowMask(mask_expand)` → `character_mask` |
| `ref` | photo du personnage | `reference_image` + CLIP vision |

- `background_video` = images source, zone du masque élargi remplie de gris 127 : le décor d'origine est gardé.
- `mask_expand` : 0 à 64 pixels (défaut 12). Cadence conseillée : `fps: 16` (envoyé par le client).
- Passes chaînées, LoRA lightx2v, encodage : inchangés.

**Crédits.** Le choix DWPose (réglages du nœud `DWPreprocessor`) et le remplissage gris de la zone du
personnage sont repris d'[Openjutsu](https://github.com/SydFloyd/openjutsu) (SydFloyd, Apache-2.0,
commit `f783a001ff3ceae4711a592623587224bf87f46c`, graphes `recast_wan22_animate.json` et `swap_wan_vace.json`).
Nœud `DWPreprocessor` : [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)
(Fannovel16, Apache-2.0), épinglé sur `a587a4c1594524c1d2de5cd1a86cd868ff448e1c` (commit testé par Openjutsu).

**Poids DWPose** (bakés dans `custom_nodes/comfyui_controlnet_aux/ckpts/`, sha256 pinnés) :
`hr16/yolox-onnx` → `yolox_l.torchscript.pt` (carte HF : `license: apache-2.0`, réupload de YOLOX, Megvii) ;
`hr16/DWPose-TorchScript-BatchSize5` → `dw-ll_ucoco_384_bs5.torchscript.pt` (carte HF : `license: apache-2.0`,
conversion TorchScript de DWPose, IDEA Research). À revoir avant usage commercial, comme les poids Wan.

Validation locale (0 crédit) :

```bash
python3 workflow.py --validate http://127.0.0.1:8188 --signal pose source.mp4 mask.mp4 ref.png
```

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
