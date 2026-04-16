FROM node:22-bookworm-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-tk \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    software-properties-common \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

COPY requirements.txt ./

RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install uv

RUN --mount=type=cache,target=/root/.cache/uv /bin/sh -c 'set -e; \
    python3 -m uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
        torch torchvision "triton>=2.1.0"; \
    if echo "${TORCH_CUDA_ARCH_LIST}" | grep -q "12\.0"; then \
        echo "Detected compute capability 12.0 (RTX 50 series). Skipping xformers install."; \
    else \
        python3 -m uv pip install --system xformers; \
    fi'

RUN --mount=type=cache,target=/root/.cache/uv python3 -m uv pip install --system "numpy<2.0.0"
RUN --mount=type=cache,target=/root/.cache/uv python3 -m uv pip install --system -r requirements.txt

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

ARG INSTALL_STABLE_FAST=0
ENV INSTALL_STABLE_FAST=${INSTALL_STABLE_FAST}

ARG INSTALL_OLLAMA=0
ENV INSTALL_OLLAMA=${INSTALL_OLLAMA}

ARG INSTALL_SAGEATTENTION=0
ENV INSTALL_SAGEATTENTION=${INSTALL_SAGEATTENTION}

ARG INSTALL_SPARGEATTN=0
ENV INSTALL_SPARGEATTN=${INSTALL_SPARGEATTN}

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/build-cache/stablefast,sharing=locked /bin/sh -c ' \
    if [ "${INSTALL_STABLE_FAST}" = "1" ]; then \
        echo "Installing stable-fast for CUDA architectures: ${TORCH_CUDA_ARCH_LIST}"; \
        export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"; \
        export FORCE_CUDA=1; \
        mkdir -p /build-cache/stablefast; \
        python3 -m pip wheel --no-build-isolation --wheel-dir /build-cache/stablefast \
            git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast; \
        python3 -m pip install --no-build-isolation --no-index --find-links /build-cache/stablefast stable-fast; \
    else \
        echo "Skipping stable-fast installation (INSTALL_STABLE_FAST=${INSTALL_STABLE_FAST})"; \
    fi'

RUN --mount=type=cache,target=/build-cache/ollama,sharing=locked /bin/sh -c ' \
    if [ "${INSTALL_OLLAMA}" = "1" ]; then \
        echo "Installing Ollama and pulling qwen3:0.6b"; \
        mkdir -p /build-cache/ollama; \
        curl -fsSL https://ollama.com/install.sh -o /build-cache/ollama/install.sh; \
        sh /build-cache/ollama/install.sh; \
        export OLLAMA_HOME=/build-cache/ollama; \
        ollama serve >/tmp/ollama.log 2>&1 & \
        OLLAMA_PID=$!; \
        attempts=0; \
        until curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; do \
            attempts=$((attempts + 1)); \
            if [ ${attempts} -gt 20 ]; then \
                echo "Ollama failed to start"; \
                kill ${OLLAMA_PID} >/dev/null 2>&1 || true; \
                exit 1; \
            fi; \
            sleep 1; \
        done; \
        ollama pull qwen3:0.6b; \
        kill ${OLLAMA_PID} >/dev/null 2>&1 || true; \
        wait ${OLLAMA_PID} 2>/dev/null || true; \
    else \
        echo "Skipping Ollama installation (INSTALL_OLLAMA=${INSTALL_OLLAMA})"; \
    fi'

COPY . .
COPY --from=frontend-builder /frontend/dist ./frontend/dist

RUN --mount=type=cache,target=/root/.cache/torch_extensions,sharing=locked \
    --mount=type=cache,target=/build-cache/sageattention,sharing=locked /bin/sh -c ' \
    if [ "${INSTALL_SAGEATTENTION}" = "1" ]; then \
        if [ -d "SageAttention" ]; then \
            echo "Found SageAttention - applying patch"; \
            cd SageAttention; \
            python3 ../docker/patch_sageattention.py; \
            python3 -m pip wheel --no-build-isolation --wheel-dir /build-cache/sageattention .; \
            python3 -m pip install --no-index /build-cache/sageattention/*.whl; \
            cd ..; \
            rm -rf SageAttention/build SageAttention/*.egg-info; \
        else \
            echo "SageAttention directory not found - cloning and applying patch"; \
            git clone --depth 1 https://github.com/thu-ml/SageAttention /tmp/SageAttention; \
            cd /tmp/SageAttention; \
            python3 /app/docker/patch_sageattention.py; \
            python3 -m pip wheel --no-build-isolation --wheel-dir /build-cache/sageattention .; \
            python3 -m pip install --no-index /build-cache/sageattention/*.whl; \
            rm -rf /tmp/SageAttention/build /tmp/SageAttention/*.egg-info; \
            rm -rf /tmp/SageAttention; \
        fi; \
    else \
        echo "Skipping SageAttention installation (INSTALL_SAGEATTENTION=${INSTALL_SAGEATTENTION})"; \
    fi'

RUN --mount=type=cache,target=/root/.cache/torch_extensions,sharing=locked \
    --mount=type=cache,target=/build-cache/spargeattn,sharing=locked /bin/sh -c ' \
    if [ "${INSTALL_SPARGEATTN}" = "1" ]; then \
        if [ -d "SpargeAttn" ]; then \
            cd SpargeAttn; \
            if echo "${TORCH_CUDA_ARCH_LIST}" | grep -qE "(8\.0|8\.6|8\.7|8\.9|9\.0)"; then \
                echo "Building SpargeAttn for supported architectures: ${TORCH_CUDA_ARCH_LIST}"; \
                python3 -m pip wheel --no-build-isolation --wheel-dir /build-cache/spargeattn .; \
                python3 -m pip install --no-index /build-cache/spargeattn/*.whl; \
                rm -rf build *.egg-info; \
            else \
                echo "Skipping SpargeAttn - architecture ${TORCH_CUDA_ARCH_LIST} not supported (requires 8.0-9.0)"; \
            fi; \
            cd ..; \
        else \
            echo "SpargeAttn directory not found - cloning and attempting build if supported"; \
            git clone --depth 1 https://github.com/thu-ml/SpargeAttn /tmp/SpargeAttn; \
            cd /tmp/SpargeAttn; \
            if echo "${TORCH_CUDA_ARCH_LIST}" | grep -qE "(8\.0|8\.6|8\.7|8\.9|9\.0)"; then \
                echo "Building cloned SpargeAttn for supported architectures: ${TORCH_CUDA_ARCH_LIST}"; \
                python3 -m pip wheel --no-build-isolation --wheel-dir /build-cache/spargeattn .; \
                python3 -m pip install --no-index /build-cache/spargeattn/*.whl; \
                rm -rf build *.egg-info; \
            else \
                echo "Skipping cloned SpargeAttn - architecture ${TORCH_CUDA_ARCH_LIST} not supported (requires 8.0-9.0)"; \
            fi; \
            cd /app; \
            rm -rf /tmp/SpargeAttn; \
        fi; \
    else \
        echo "Skipping SpargeAttn installation (INSTALL_SPARGEATTN=${INSTALL_SPARGEATTN})"; \
    fi'

RUN mkdir -p ./output/classic \
    ./output/Flux \
    ./output/HiresFix \
    ./output/Img2Img \
    ./output/Adetailer \
    ./include/checkpoints \
    ./include/clip \
    ./include/embeddings \
    ./include/ESRGAN \
    ./include/loras \
    ./include/sd1_tokenizer \
    ./include/text_encoder \
    ./include/unet \
    ./include/vae \
    ./include/vae_approx \
    ./include/yolos

RUN echo "42" > ./include/last_seed.txt
RUN echo "A beautiful landscape" > ./include/prompt.txt

EXPOSE 7860

ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD if [ "${INSTALL_OLLAMA}" = "1" ]; then \
        echo "Starting Ollama server"; \
        ollama serve >/tmp/ollama_runtime.log 2>&1 & \
        for attempt in $(seq 1 20); do \
            if curl -fsS http://127.0.0.1:11434/api/version >/dev/null 2>&1; then \
                break; \
            fi; \
            sleep 1; \
        done; \
    fi; \
    exec python3 server.py --host 0.0.0.0 --port "${PORT}"
