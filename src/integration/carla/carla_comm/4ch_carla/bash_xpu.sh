#!/usr/bin/env bash
python3 -m venv .venv_xpu
source .venv_xpu/bin/activate
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu
pip install -r requirements_xpu.txt