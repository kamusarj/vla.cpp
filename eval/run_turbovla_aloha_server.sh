#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GGUF="${GGUF:-$REPO/checkpoints/turbovla/DuyBao44DOCer-TurboVLA/finetune_step_5000-bf16.gguf}"
BIND="${VLA_BIND:-tcp://*:5555}"
SERVER_BIN="${SERVER_BIN:-$REPO/build-turbovla/vla-server}"

if [[ ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: vla-server not found: $SERVER_BIN" >&2
    exit 1
fi
if [[ ! -f "$GGUF" ]]; then
    echo "ERROR: TurboVLA GGUF not found: $GGUF" >&2
    exit 1
fi

exec "$SERVER_BIN" --bind "$BIND" --timing-detail phase "$GGUF"

