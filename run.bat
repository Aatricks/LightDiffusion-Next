@echo off
SET VENV_DIR=.venv

REM Check if .venv exists
IF NOT EXIST %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Activate the virtual environment
CALL %VENV_DIR%\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install specific packages
echo Installing required packages...
pip install uv

REM Check GPU type
SET TORCH_URL=https://download.pytorch.org/whl/cpu
nvidia-smi >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU detected
    SET TORCH_URL=https://download.pytorch.org/whl/cu124
    uv pip install xformers torch torchvision --index-url %TORCH_URL%
) ELSE (
    echo No compatible GPU detected, using CPU
    uv pip install torch torchvision --index-url %TORCH_URL%
)

REM Install additional requirements
IF EXIST requirements.txt (
    echo Installing additional requirements...
    uv pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found, skipping...
)

REM Launch the script
echo Launching LightDiffusion...
python .\modules\user\GUI.py

REM Deactivate the virtual environment
deactivate
