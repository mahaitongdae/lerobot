FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    LIBERO_DATASET_PATH=/app/libero_dataset \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential git curl libglib2.0-0 libegl1-mesa-dev ffmpeg \
    libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
    cmake pkg-config ninja-build htop pipx vim \
    && ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pipx install nvitop

WORKDIR /app

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir lerobot==0.4.4 && \
    pip install --no-cache-dir libero==0.1.1 --use-pep517 --no-build-isolation

RUN git clone --branch dev --single-branch --depth 1 \
    https://github.com/mahaitongdae/lerobot.git . && \
    pip install --no-cache-dir -e .

CMD ["/bin/bash"]
