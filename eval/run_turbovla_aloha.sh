#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Start TurboVLA's C++ server and execute the real MuJoCo ALOHA
# carrot-from-plate-to-cup task. Override SERVER_BIN, CLIENT_PYTHON,
# GGUF, STATS, TOKENIZER, PORT, or OUTPUT_DIR when needed.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO/checkpoints/turbovla/DuyBao44DOCer-TurboVLA"
GGUF="${GGUF:-$MODEL_DIR/finetune_step_5000-bf16.gguf}"
STATS="${STATS:-$MODEL_DIR/finetune_step_5000-bf16.stats.json}"
port_was_explicit=0
if [[ -n "${PORT:-}" ]]; then
    port_was_explicit=1
else
    PORT=5557
fi
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/turbovla_aloha_sim}"
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
    echo "ERROR: vla-server not found; build target vla-server or set SERVER_BIN" >&2
    exit 1
fi
if [[ ! -f "$GGUF" || ! -f "$STATS" ]]; then
    echo "ERROR: TurboVLA GGUF/stats are missing under $MODEL_DIR" >&2
    exit 1
fi

if [[ -z "${CLIENT_PYTHON:-}" ]]; then
    for candidate in \
        "$REPO/eval/sim/aloha/aloha_uv/.venv/bin/python" \
        "$HOME/anaconda3/envs/octo_pt/bin/python" \
        "$REPO/eval/sim/libero/libero_uv/.venv/bin/python"; do
        if [[ -x "$candidate" ]] && "$candidate" -c \
            'import aloha_sim, cv2, dm_control, torch, transformers, zmq' \
            >/dev/null 2>&1; then
            CLIENT_PYTHON="$candidate"
            break
        fi
    done
fi
if [[ -z "${CLIENT_PYTHON:-}" ]]; then
    echo "ERROR: ALOHA simulator Python environment not found." >&2
    echo "       Run: bash eval/sim/aloha/setup_aloha_sim.sh" >&2
    exit 1
fi

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "ERROR: PORT must be an integer in [1, 65535], got: $PORT" >&2
    exit 1
fi

port_is_free() {
    "$CLIENT_PYTHON" - "$1" <<'PY'
import socket
import sys

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError:
        raise SystemExit(1)
PY
}

if ! port_is_free "$PORT"; then
    if (( port_was_explicit )); then
        echo "ERROR: requested PORT=$PORT is already in use." >&2
        echo "       Choose another one, for example: PORT=5560 $0 $*" >&2
        exit 1
    fi
    busy_port="$PORT"
    PORT="$($CLIENT_PYTHON - <<'PY'
import socket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    print(sock.getsockname()[1])
PY
)"
    echo "Default port $busy_port is busy; using free port $PORT."
fi

if [[ -z "${TOKENIZER:-}" ]]; then
    if [[ -f "$HOME/Desktop/TurboVLA/pretrained/bert-base-uncased/tokenizer.json" ]]; then
        TOKENIZER="$HOME/Desktop/TurboVLA/pretrained/bert-base-uncased"
    else
        TOKENIZER="google-bert/bert-base-uncased"
    fi
fi

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

MUJOCO_GL="${MUJOCO_GL:-egl}" "$CLIENT_PYTHON" \
    "$REPO/eval/client/run_turbovla_aloha_sim.py" \
    --vla-addr "tcp://127.0.0.1:$PORT" \
    --stats-json "$STATS" \
    --tokenizer "$TOKENIZER" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "server log: $SERVER_LOG"
