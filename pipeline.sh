#!/bin/bash

VENV_DIR=.venv

sudo apt-get install python3.10 python3.10-venv

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

# Install specific packages
echo "Installing required packages..."
pip install xformers==0.0.26post1 torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# Install tkinter
echo "Installing tkinter..."
sudo apt-get install python3.10-tk

# Install additional requirements
if [ -f requirements.txt ]; then
    echo "Installing additional requirements..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found, skipping..."
fi

REM Check for enhance-prompt argument
echo Checking for enhance-prompt argument...
if [[ " $* " == *" --enhance-prompt "* ]]; then
    echo "Installing ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    ollama pull llama3.2
fi

# Launch the script
echo "Launching LightDiffusion..."
python3.10 "./modules/user/pipeline.py" %*

# Deactivate the virtual environment
deactivate