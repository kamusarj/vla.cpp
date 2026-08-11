#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/aloha_uv/.venv"

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv is required: https://docs.astral.sh/uv/" >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    mkdir -p "$(dirname "$VENV_DIR")"
    uv venv "$VENV_DIR" --python 3.10
fi

"$VENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 || true
uv pip install --python "$VENV_DIR/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1
uv pip install --python "$VENV_DIR/bin/python" \
    numpy==2.2.6 mujoco==3.3.3 dm-control==1.0.31 \
    av==14.2.0 pandas==2.3.3 pyarrow==19.0.1 \
    opencv-python-headless==4.10.0.84 \
    transformers==4.51.3 pyzmq==26.2.0 protobuf==5.29.5 pillow==11.1.0 \
    "aloha_sim @ git+https://github.com/google-deepmind/aloha_sim.git@d02904607cca1bf6dfb72f30b522506ac7ca0f91"
# Match the tested MuJoCo runtime. aloha_sim accepts dm-control>=1.0.31, and
# 1.0.31 is the latest release whose binary interface supports MuJoCo 3.3.3.
MUJOCO_GL=egl "$VENV_DIR/bin/python" -c \
    'import aloha_sim, cv2, dm_control, mujoco, torch, transformers, zmq'

echo
echo "Official google-deepmind/aloha_sim environment is ready."
echo "Run: eval/run_turbovla_aloha.sh --episodes 1"
