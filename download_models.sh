#!/usr/bin/env bash
# fakinflu — poids de Wan 2.2 Animate (route remplacement, docs/plan-replace.md), BAKES dans l'image.
# Memes fichiers que le workflow natif ComfyUI « Wan Animate 2 ». Tout vient d'un seul depot
# (Comfy-Org/Wan-Animate-2), epingle sur un commit ; chaque fichier verifie par son sha256 (= oid LFS
# lu via l'API HF le 2026-10-14). Les encodeurs/VAE/LoRA ont les memes oid que le worker scail.
#   diffusion Wan 2.2 Animate 14B : Comfy-Org/Wan-Animate-2 -> diffusion_models/
#   umT5 XXL fp8, CLIP vision H, VAE Wan 2.1, LoRA lightx2v : meme depot -> text_encoders/ clip_vision/ vae/ loras/
#
#   MODELS_DIR=/comfyui/models ./download_models.sh core   # ~10 Go (encodeurs, CLIP vision, VAE, LoRA)
#   MODELS_DIR=/comfyui/models ./download_models.sh int8   # ~17 Go (preset precision=int8, defaut)
#   MODELS_DIR=/comfyui/models ./download_models.sh bf16   # ~33 Go (preset precision=bf16)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/comfyui/models}"

# DL <depot> <commit> <chemin dans le depot> <sous-dossier local> <sha256>
DL() {
  local repo="$1" rev="$2" src="$3" sub="$4" sha="$5" name
  name="$(basename "$src")"
  local dest="$MODELS_DIR/$sub/$name" url="https://huggingface.co/${repo}/resolve/${rev}/${src}"
  mkdir -p "$MODELS_DIR/$sub"
  echo ">> $sub/$name"
  local t0=$SECONDS
  if command -v aria2c >/dev/null; then
    aria2c -q -c -x 16 -s 16 -k 64M --max-tries=10 --retry-wait=5 --timeout=30 \
           --lowest-speed-limit=1M --file-allocation=none --console-log-level=warn \
           -d "$MODELS_DIR/$sub" -o "$name" "$url"
  else
    wget -q -c --tries=10 --timeout=30 --waitretry=5 -O "$dest" "$url"
  fi
  echo "   telecharge en $((SECONDS - t0)) s"
  echo "$sha  $dest" | sha256sum -c --quiet - || { echo "!! sha256 invalide : $dest"; rm -f "$dest"; exit 1; }
}

WA=e7181fe1896b7f2ce34120d1a3663413548240d7   # Comfy-Org/Wan-Animate-2 @ 2026-10-14

case "${1:-}" in
  core)
    DL Comfy-Org/Wan-Animate-2 "$WA" text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors text_encoders \
       c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68
    DL Comfy-Org/Wan-Animate-2 "$WA" clip_vision/clip_vision_h.safetensors clip_vision \
       64a7ef761bfccbadbaa3da77366aac4185a6c58fa5de5f589b42a65bcc21f161
    DL Comfy-Org/Wan-Animate-2 "$WA" vae/Wan2_1_VAE_bf16.safetensors vae \
       1ab9a32cc2c740f6e39d80d367ce5dcc28db8c71b79b28670546b8973e9d75f9
    DL Comfy-Org/Wan-Animate-2 "$WA" loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors loras \
       85c4a61c30e0497aa44b91d93a893b624708461a56fe5485183b28fa07e2dfb3 ;;
  int8)
    DL Comfy-Org/Wan-Animate-2 "$WA" diffusion_models/wan_animate_2_int8_convrot.safetensors diffusion_models \
       0580ecdd65e47e97c30df9670d13a6c4a131d26de5a1faf2ccc78392d5167584 ;;
  bf16)
    DL Comfy-Org/Wan-Animate-2 "$WA" diffusion_models/wan_animate_2_bf16.safetensors diffusion_models \
       642e906c7deaa27072a90ea433319b18f274da438ba75e350373467a0421f177 ;;
  dwpose)
    # signal pose : MODELS_DIR = <comfyui_controlnet_aux>/ckpts, sous-dossier = depot HF (custom_hf_download).
    # sha256 = oid LFS lu via l'API HF le 2026-10-22. Licence des deux depots : apache-2.0 (cartes HF).
    DL hr16/yolox-onnx a124b32c3b7c5cebda1c7cd96178f0f9d2050125 yolox_l.torchscript.pt hr16/yolox-onnx \
       80bc14b13c260c24b3014cd42c02994bf52296ab8fa2d80a60b6afe08c93ef42
    DL hr16/DWPose-TorchScript-BatchSize5 359d662a9b33b73f6d0f21732baf8845f17bb4be dw-ll_ucoco_384_bs5.torchscript.pt \
       hr16/DWPose-TorchScript-BatchSize5 d86a0b2b59fddc0901a7076e9f59c9f8602602133ed72511c693fd11eea23d91 ;;
  *) echo "usage: $0 core|int8|bf16|dwpose" >&2; exit 2 ;;
esac
