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
pip install xformers==0.0.26post1 torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

REM Install additional requirements
IF EXIST requirements.txt (
    echo Installing additional requirements...
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found, skipping...
)

REM Launch the script
echo Launching LightDiffusion...
python .\modules\pipeline.py

REM Deactivate the virtual environment
deactivate
