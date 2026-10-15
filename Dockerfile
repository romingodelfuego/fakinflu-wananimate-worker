# fakinflu — worker RunPod : wananimate (Wan 2.2 Animate 14B via ComfyUI). Route remplacement.
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=wananimate
#
# Base : runpod/worker-comfyui. WanAnimateToVideo est NATIF (comfy_extras/nodes_wan.py) a partir de la
# version de ComfyUI qui a introduit Wan 2.2 Animate. Le worker construit son graphe lui-meme
# (workflow.py) ; seul ajout : fakinflu_nodes (chargement de la video de controle + encodage par morceau).
#
# IMPORTANT : WORKER_TAG doit pointer un tag `-base` dont le ComfyUI (a) embarque WanAnimateToVideo ET
# (b) sait DETECTER le checkpoint Wan Animate 2 int8 'convrot' (model_type animate2, in_dim 36). Le
# `latest-base` initial livrait ComfyUI 0.17.2 : le noeud existait mais model_config_from_unet faisait
# "no match" sur l'unet int8 -> crash quant_config=None au 1er job. On epingle 5.10.0-base = ComfyUI
# 0.34.0 (comme scail, qui fait tourner Wan 2.1). Les deux garde-fous ci-dessous echouent le BUILD si
# 0.34.0 n'a pas le noeud -> on le sait sans gaspiller un job GPU.
ARG WORKER_TAG=5.10.0-base
FROM runpod/worker-comfyui:${WORKER_TAG}

# Garde-fou de parite de version : echoue ICI si ComfyUI n'a pas le noeud attendu.
RUN grep -q '"WanAnimateToVideo"' /comfyui/comfy_extras/nodes_wan.py \
 && grep -q 'character_mask' /comfyui/comfy_extras/nodes_wan.py \
 && grep -q '"SamplerCustom"' /comfyui/comfy_extras/nodes_custom_sampler.py

# ComfyUI dimensionne sa memoire epinglee sur la RAM de l'HOTE, pas sur la limite du conteneur : en
# chargeant les poids, le noyau peut tuer ComfyUI sans trace (cf. worker scail). On ajoute
# ${COMFY_EXTRA_ARGS} au start.sh de worker-comfyui (defaut ci-dessous, modifiable dans l'env sans rebuild).
RUN sed -i 's#--log-stdout &#--log-stdout ${COMFY_EXTRA_ARGS} \&#' /start.sh \
 && test "$(grep -c 'COMFY_EXTRA_ARGS} &' /start.sh)" -ge 1
ENV COMFY_EXTRA_ARGS="--disable-pinned-memory"

RUN apt-get update && apt-get install -y --no-install-recommends aria2 \
 && rm -rf /var/lib/apt/lists/*

# --- Poids BAKES : core ~10 Go + int8 ~17 Go (WITH_INT8) + bf16 ~33 Go (WITH_BF16) ------------------
# int8 par defaut (modele resident en VRAM sur 48 Go) ; bf16 en option (plus fidele, plus lourd).
ARG WITH_INT8=1
ARG WITH_BF16=0
COPY download_models.sh /tmp/download_models.sh
RUN MODELS_DIR=/comfyui/models bash /tmp/download_models.sh core
RUN if [ "${WITH_INT8}" = "1" ]; then MODELS_DIR=/comfyui/models bash /tmp/download_models.sh int8; fi
RUN if [ "${WITH_BF16}" = "1" ]; then MODELS_DIR=/comfyui/models bash /tmp/download_models.sh bf16; fi \
 && rm -f /tmp/download_models.sh

COPY fakinflu_nodes /comfyui/custom_nodes/fakinflu_nodes

# Garde-fou API ComfyUI, 100% STATIQUE (aucun import torch/comfy -> aucune init CUDA, donc OK sans GPU
# au build). Echoue le BUILD si un noeud reference une API comfy.<module>.<attr> inexistante (ex.
# `intermediate_dtype`) : le bug est attrape ici, pas sur un job GPU facture. NE PAS importer
# comfy.model_management au build : son code de module appelle torch.cuda et casse sans driver.
COPY check_comfy_api.py /tmp/check_comfy_api.py
RUN python3 /tmp/check_comfy_api.py /comfyui/custom_nodes/fakinflu_nodes/__init__.py /comfyui/comfy

# --- Handler fakinflu (remplace /handler.py de worker-comfyui) ---------------
COPY transfer.py workflow.py VERSION* /fakinflu/
COPY handler.py /handler.py
COPY test_input.json /test_input.json
ENV PYTHONPATH=/fakinflu \
    COMFY_LOG_LEVEL=INFO

CMD ["/start.sh"]
