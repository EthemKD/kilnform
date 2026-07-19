@echo off
rem Kilnform AI backend setup (one-time). Safe to re-run as a repair tool.
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
  echo Installing uv...
  winget install --id=astral-sh.uv -e --silent --accept-source-agreements --accept-package-agreements
)

if not exist .venv (
  uv venv --python 3.12 .venv
)

echo Installing PyTorch cu128 (large download)...
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo Installing dependencies...
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

if not exist vendor\TripoSR\tsr (
  echo Fetching TripoSR...
  git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git vendor\TripoSR
)

echo Downloading / verifying models (first run pulls ~5GB)...
.venv\Scripts\python.exe test_pipeline.py

echo Setup complete.
pause
