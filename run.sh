#!/bin/bash

VENV_DIR=.venv

sudo apt-get install python3.12 python3.12-venv python3.12-full

# Check if .venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install build dependencies first
echo "Installing build dependencies..."
pip install setuptools wheel ninja

# Check for CUDA installation
if nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install xformers torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # Install stable-fast without CUDA
    echo "Installing stable-fast without CUDA support..."
    # Create a temporary script to set environment variable and run pip
    cat > temp_install.sh << 'EOF'
#!/bin/bash
export WITH_CUDA=0
pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@main#egg=stable-fast
EOF

    chmod +x temp_install.sh
    ./temp_install.sh
    rm temp_install.sh
fi
# Install tkinter
echo "Installing tkinter..."
sudo apt-get install python3.12-tk

# Install additional requirements
if [ -f requirements.txt ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping..."
fi

# Launch the script
echo "Launching LightDiffusion..."
python3.12 ./modules/user/GUI.py

# Deactivate the virtual environment
deactivate
