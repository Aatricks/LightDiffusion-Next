# Use NVIDIA CUDA base image with development tools for building extensions
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

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
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install uv

# Install PyTorch with CUDA support
RUN python3 -m uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision "triton>=2.1.0"

# Install numpy with version constraint
RUN python3 -m uv pip install --system "numpy<2.0.0"

# Install Python dependencies
RUN python3 -m uv pip install --system -r requirements.txt

# Copy the entire project (including SageAttention and SpargeAttn directories)
COPY . .

# Set target GPU architectures for building CUDA extensions
# Common architectures: 8.0 (A100), 8.6 (RTX 30xx), 8.9 (RTX 40xx), 9.0 (H100), 12.0 (RTX 50xx/Blackwell)
# You can customize this via build arg: --build-arg TORCH_CUDA_ARCH_LIST="12.0"
ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Patch SageAttention setup.py to support TORCH_CUDA_ARCH_LIST environment variable
RUN cd SageAttention && \
    python3 ../docker/patch_sageattention.py && \
    cd ..

# Build and install SageAttention from source
# Limit parallel jobs to prevent out-of-memory errors during compilation
ENV MAX_JOBS=2
RUN cd SageAttention && \
    python3 setup.py build_ext --parallel 2 && \
    python3 -m pip install -e . --no-build-isolation && \
    cd .. && \
    rm -rf SageAttention/build SageAttention/*.egg-info

# Build and install SpargeAttention from source
# Note: SpargeAttn only supports compute capabilities: 8.0, 8.6, 8.7, 8.9, 9.0
# Skip if building for unsupported architectures (e.g., 12.0 for RTX 50 series)
RUN cd SpargeAttn && \
    if echo "${TORCH_CUDA_ARCH_LIST}" | grep -qE '(8\.0|8\.6|8\.7|8\.9|9\.0)'; then \
        echo "Building SpargeAttn for supported architectures: ${TORCH_CUDA_ARCH_LIST}"; \
        python3 setup.py build_ext --parallel 2 && \
        python3 -m pip install -e . --no-build-isolation && \
        rm -rf build *.egg-info; \
    else \
        echo "Skipping SpargeAttn - architecture ${TORCH_CUDA_ARCH_LIST} not supported (requires 8.0-9.0)"; \
    fi && \
    cd ..

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
CMD if [ "$UI_FRAMEWORK" = "gradio" ]; then \
        python3 app.py; \
    else \
        streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501; \
    fi
