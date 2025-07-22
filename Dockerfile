# Use Python 3.10 base image
FROM python:3.10-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-venv \
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
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install uv for faster package installation
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install uv

# Install PyTorch with CUDA support
RUN python3 -m uv pip install --system --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision  "triton>=2.1.0"

# Install numpy with version constraint
RUN python3 -m uv pip install --system "numpy<2.0.0"

# Install Python dependencies
RUN python3 -m uv pip install --system -r requirements.txt

# Copy the entire project
COPY . .

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

# Expose the port that Gradio will run on
EXPOSE 7860

# Set environment variable to indicate this is running in a container
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the Gradio app
CMD ["python3", "app.py"]
