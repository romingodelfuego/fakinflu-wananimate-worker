# fakinflu — worker RunPod : wananimate (Comfy-Org/Wan-Animate-2)
# Build par RunPod (integration GitHub) a chaque GitHub Release : make worker-release W=wananimate
ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
 && pip3 install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# TODO : deps du modele (epingler les versions = builds reproductibles)
RUN pip3 install "huggingface_hub>=0.27" pillow runpod

# Poids BAKES dans l'image : pas de volume (0 $ de stockage), pas de telechargement
# au cold start. Restreindre allow_patterns au strict necessaire (limite image : 80 Go).
ARG MODEL_ID=Comfy-Org/Wan-Animate-2
ENV MODEL_ID=${MODEL_ID}
RUN python3 -c "from huggingface_hub import snapshot_download as d; \
d('${MODEL_ID}', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])"

COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]
