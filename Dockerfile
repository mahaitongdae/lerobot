FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    LIBERO_CONFIG_PATH=/libero_data/libero_config \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git curl libglib2.0-0 libegl1-mesa-dev ffmpeg \
    libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
    cmake pkg-config ninja-build htop pipx vim parallel \
    && ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pipx install nvitop

WORKDIR /app

# RUN python -m pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir robosuite==1.4.0 && \
#     pip install --no-cache-dir libero==0.1.1 --use-pep517 --no-build-isolation

RUN python -m pip install --no-cache-dir --upgrade pip && \
    git clone --branch dev --single-branch --depth 1 \
    https://github.com/mahaitongdae/lerobot.git . && \
    pip install --no-cache-dir -e . && \
    pip install --no-cache-dir libero==0.1.1 --use-pep517 --no-build-isolation && \
    pip install --no-cache-dir huggingface_hub==0.36.2 transformers==4.57.3 && \
    pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir git+https://github.com/facebookresearch/r3m.git --no-deps && \
    pip install --no-cache-dir voltron-robotics

RUN mkdir -p /libero_data/libero_config \
    && printf "assets: /usr/local/lib/python3.10/dist-packages/libero/libero/./assets\n\
bddl_files: /usr/local/lib/python3.10/dist-packages/libero/libero/./bddl_files\n\
benchmark_root: /usr/local/lib/python3.10/dist-packages/libero/libero/./benchmark\n\
datasets: /usr/local/lib/python3.10/dist-packages/libero/libero/../datasets\n\
init_states: /usr/local/lib/python3.10/dist-packages/libero/libero/./init_files\n" \
    > /libero_data/libero_config/config.yaml

CMD ["/bin/bash"]
