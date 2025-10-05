# Use NVIDIA CUDA base image with development tools for building extensions
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
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

# Set python3.10 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1


# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install uv for faster package installation
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install --upgrade pip
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install uv

# Install PyTorch with CUDA support
RUN --mount=type=cache,target=/root/.cache/uv python3 -m uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision "triton>=2.1.0"

# Install numpy with version constraint
RUN --mount=type=cache,target=/root/.cache/uv python3 -m uv pip install --system "numpy<2.0.0"

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache/uv python3 -m uv pip install --system -r requirements.txt

# Provide cuDNN/cuBLAS python packages so native builds can discover headers without overriding CUDA libs
RUN --mount=type=cache,target=/root/.cache/uv python3 -m uv pip install --system \
    nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-cublas-cu12==12.5.3.2

# Allow overriding CUDA architectures later in the build without busting earlier layers
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Toggle stable-fast installation (set to 0 to skip during docker build)
ARG INSTALL_STABLE_FAST=0
ENV INSTALL_STABLE_FAST=${INSTALL_STABLE_FAST}

# Toggle Ollama installation (set to 1 to install and pre-pull qwen3:0.6b)
ARG INSTALL_OLLAMA=0
ENV INSTALL_OLLAMA=${INSTALL_OLLAMA}

# Build and install stable-fast with matching CUDA architectures
RUN --mount=type=cache,target=/root/.cache/pip /bin/sh -c ' \
    if [ "${INSTALL_STABLE_FAST}" = "1" ]; then \
        echo "Installing stable-fast for CUDA architectures: ${TORCH_CUDA_ARCH_LIST}"; \
        export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"; \
        export FORCE_CUDA=1; \
        python3 -m pip install --no-build-isolation \
            git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast; \
    else \
        echo "Skipping stable-fast installation (INSTALL_STABLE_FAST=${INSTALL_STABLE_FAST})"; \
    fi'

# Optionally install Ollama with the qwen3:0.6b model for prompt enhancement
RUN /bin/sh -c ' \
    if [ "${INSTALL_OLLAMA}" = "1" ]; then \
        echo "Installing Ollama and pulling qwen3:0.6b"; \
        curl -fsSL https://ollama.com/install.sh | sh; \
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

# Copy the entire project (including SageAttention and SpargeAttn directories)
COPY . .

# Patch SageAttention setup.py to support TORCH_CUDA_ARCH_LIST environment variable
# Only attempt to patch if the SageAttention directory exists in the build context
RUN if [ -d "SageAttention" ]; then \
        echo "Found SageAttention - applying patch" && \
        cd SageAttention && \
        python3 ../docker/patch_sageattention.py && \
        cd ..; \
    else \
        echo "SageAttention directory not found - cloning and applying patch" && \
        git clone --depth 1 https://github.com/thu-ml/SageAttention /tmp/SageAttention && \
        cd /tmp/SageAttention && \
        python3 /app/docker/patch_sageattention.py && \
        cd /app; \
    fi

# Build and install SageAttention from source (only if present)
# Limit parallel jobs to prevent out-of-memory errors during compilation
ENV MAX_JOBS=2
RUN if [ -d "SageAttention" ]; then \
        echo "Building SageAttention (local copy)" && \
        cd SageAttention && \
        python3 setup.py build_ext --parallel 2 && \
        python3 -m pip install -e . --no-build-isolation && \
        cd .. && \
        rm -rf SageAttention/build SageAttention/*.egg-info; \
    else \
        echo "Building SageAttention (cloned)" && \
        cd /tmp/SageAttention && \
        python3 setup.py build_ext --parallel 2 && \
        python3 -m pip install -e . --no-build-isolation && \
        rm -rf /tmp/SageAttention/build /tmp/SageAttention/*.egg-info && \
        rm -rf /tmp/SageAttention; \
    fi

# Build and install SpargeAttention from source
# Note: SpargeAttn only supports compute capabilities: 8.0, 8.6, 8.7, 8.9, 9.0
# Skip if building for unsupported architectures (e.g., 12.0 for RTX 50 series)
RUN if [ -d "SpargeAttn" ]; then \
        cd SpargeAttn && \
        if echo "${TORCH_CUDA_ARCH_LIST}" | grep -qE '(8\.0|8\.6|8\.7|8\.9|9\.0)'; then \
            echo "Building SpargeAttn for supported architectures: ${TORCH_CUDA_ARCH_LIST}"; \
            python3 setup.py build_ext --parallel 2 && \
            python3 -m pip install -e . --no-build-isolation && \
            rm -rf build *.egg-info; \
        else \
            echo "Skipping SpargeAttn - architecture ${TORCH_CUDA_ARCH_LIST} not supported (requires 8.0-9.0)"; \
        fi && \
        cd ..; \
    else \
        echo "SpargeAttn directory not found - cloning and attempting build if supported" && \
        git clone --depth 1 https://github.com/thu-ml/SpargeAttn /tmp/SpargeAttn && \
        cd /tmp/SpargeAttn && \
        if echo "${TORCH_CUDA_ARCH_LIST}" | grep -qE '(8\.0|8\.6|8\.7|8\.9|9\.0)'; then \
            echo "Building cloned SpargeAttn for supported architectures: ${TORCH_CUDA_ARCH_LIST}"; \
            python3 setup.py build_ext --parallel 2 && \
            python3 -m pip install -e . --no-build-isolation && \
            rm -rf build *.egg-info; \
        else \
            echo "Skipping cloned SpargeAttn - architecture ${TORCH_CUDA_ARCH_LIST} not supported (requires 8.0-9.0)"; \
        fi && \
        cd /app && rm -rf /tmp/SpargeAttn; \
    fi

# Create necessary directories
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
    ./include/unet \
    ./include/vae \
    ./include/vae_approx \
    ./include/yolos

# Create last_seed.txt if it doesn't exist
RUN echo "42" > ./include/last_seed.txt

# Create prompt.txt if it doesn't exist
RUN echo "A beautiful landscape" > ./include/prompt.txt

# Expose the ports for both Gradio and Streamlit
EXPOSE 7860
EXPOSE 8501

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV UI_FRAMEWORK=${UI_FRAMEWORK:-streamlit}

# Health check (supports both UIs)
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${UI_FRAMEWORK:+8501}${UI_FRAMEWORK:-7860}/ || exit 1

# Run the app based on UI_FRAMEWORK environment variable
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
    if [ "$UI_FRAMEWORK" = "gradio" ]; then \
        python3 app.py; \
    else \
        streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501; \
    fi
