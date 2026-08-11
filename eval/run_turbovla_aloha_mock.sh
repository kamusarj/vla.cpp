#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Start a local TurboVLA ALOHA vla-server, run the closed-loop protocol mock,
# then stop the server. Override SERVER_BIN, GGUF, STATS, TOKENIZER, PORT, or
# CLIENT_PYTHON through the environment when needed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO/checkpoints/turbovla/DuyBao44DOCer-TurboVLA"
GGUF="${GGUF:-$MODEL_DIR/finetune_step_5000-bf16.gguf}"
STATS="${STATS:-$MODEL_DIR/finetune_step_5000-bf16.stats.json}"
PORT="${PORT:-5557}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/turbovla_aloha_mock}"
SERVER_LOG="$OUTPUT_DIR/server.log"

if [[ -z "${SERVER_BIN:-}" ]]; then
    for candidate in "$REPO/build-turbovla/vla-server" "$REPO/build/vla-server"; do
        if [[ -x "$candidate" ]]; then
            SERVER_BIN="$candidate"
            break
        fi
    done
fi
if [[ -z "${SERVER_BIN:-}" || ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: vla-server not found; set SERVER_BIN or build the vla-server target" >&2
    exit 1
fi
if [[ ! -f "$GGUF" ]]; then
    echo "ERROR: TurboVLA GGUF not found: $GGUF" >&2
    exit 1
fi
if [[ ! -f "$STATS" ]]; then
    echo "ERROR: TurboVLA stats sidecar not found: $STATS" >&2
    echo "       Re-run scripts/convert_turbovla_to_gguf.py to generate it." >&2
    exit 1
fi

if [[ -z "${CLIENT_PYTHON:-}" ]]; then
    if [[ -x "$REPO/eval/sim/libero/libero_uv/.venv/bin/python" ]]; then
        CLIENT_PYTHON="$REPO/eval/sim/libero/libero_uv/.venv/bin/python"
    else
        CLIENT_PYTHON="python3"
    fi
fi
TOKENIZER="${TOKENIZER:-google-bert/bert-base-uncased}"

mkdir -p "$OUTPUT_DIR"
"$SERVER_BIN" --bind "tcp://127.0.0.1:$PORT" --timing-detail phase "$GGUF" \
    >"$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
    if kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

ready=0
for _ in $(seq 1 300); do
    if grep -q "ready\." "$SERVER_LOG" 2>/dev/null; then
        ready=1
        break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: vla-server exited during startup" >&2
        sed -n '1,200p' "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 0.2
done
if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: timed out waiting for vla-server" >&2
    sed -n '1,200p' "$SERVER_LOG" >&2
    exit 1
fi

"$CLIENT_PYTHON" "$REPO/eval/client/run_turbovla_aloha_mock.py" \
    --vla-addr "tcp://127.0.0.1:$PORT" \
    --stats-json "$STATS" \
    --tokenizer "$TOKENIZER" \
    --output "$OUTPUT_DIR/summary.json" \
    "$@"

echo "server log: $SERVER_LOG"
