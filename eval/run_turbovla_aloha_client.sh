#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Physical ALOHA deployment for the exact left-arm carrot-to-cup checkpoint.
# Source ROS2 + the Interbotix workspace before running this script.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATS="${STATS:-$REPO/checkpoints/turbovla/DuyBao44DOCer-TurboVLA/finetune_step_5000-bf16.stats.json}"
VLA_ADDR="${VLA_ADDR:-tcp://localhost:5555}"

if [[ ! -f "$STATS" ]]; then
    echo "ERROR: TurboVLA stats not found: $STATS" >&2
    exit 1
fi
if [[ -z "${TOKENIZER:-}" ]]; then
    if [[ -f "$HOME/Desktop/TurboVLA/pretrained/bert-base-uncased/tokenizer.json" ]]; then
        TOKENIZER="$HOME/Desktop/TurboVLA/pretrained/bert-base-uncased"
    else
        TOKENIZER="google-bert/bert-base-uncased"
    fi
fi

python3 "$REPO/eval/client/run_ALOHA_client_direct.py" \
    --arch turbovla_aloha \
    --vla-addr "$VLA_ADDR" \
    --stats-json "$STATS" \
    --tokenizer "$TOKENIZER" \
    --image-size 256 \
    --max-state-dim 7 \
    --max-length 256 \
    --n-action-steps 12 \
    --executed-length 12 \
    --smooth-step 1 \
    --action-hz 15 \
    --task "Grasp the carrot from the plate, hold it, place it into the cup." \
    "$@"

