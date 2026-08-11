#!/usr/bin/env bash
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

# Start vla-server and reproduce TurboVLA's ALOHA val/train-sample open-loop
# evaluation against the real LeRobot recordings (no simulator).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$REPO/checkpoints/turbovla/DuyBao44DOCer-TurboVLA"
GGUF="${GGUF:-$MODEL_DIR/finetune_step_5000-bf16.gguf}"
STATS="${STATS:-$MODEL_DIR/finetune_step_5000-bf16.stats.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO/outputs/turbovla_aloha_openloop}"
SERVER_LOG="$OUTPUT_DIR/server.log"
port_was_explicit=0
if [[ -n "${PORT:-}" ]]; then port_was_explicit=1; else PORT=5558; fi

if [[ -z "${SERVER_BIN:-}" ]]; then
    for candidate in "$REPO/build-turbovla/vla-server" "$REPO/build/vla-server"; do
        if [[ -x "$candidate" ]]; then SERVER_BIN="$candidate"; break; fi
    done
fi
if [[ -z "${SERVER_BIN:-}" || ! -x "$SERVER_BIN" ]]; then
    echo "ERROR: vla-server not found; build it or set SERVER_BIN" >&2
    exit 1
fi
if [[ ! -f "$GGUF" || ! -f "$STATS" ]]; then
    echo "ERROR: TurboVLA GGUF/stats are missing under $MODEL_DIR" >&2
    exit 1
fi

CLIENT_PYTHON="${CLIENT_PYTHON:-$REPO/eval/sim/aloha/aloha_uv/.venv/bin/python}"
if [[ ! -x "$CLIENT_PYTHON" ]] || ! "$CLIENT_PYTHON" -c \
    'import av, matplotlib, pandas, pyarrow, torch, transformers, zmq' >/dev/null 2>&1; then
    echo "ERROR: open-loop environment is missing." >&2
    echo "       Run: bash eval/sim/aloha/setup_aloha_sim.sh" >&2
    exit 1
fi
if [[ ! "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
    echo "ERROR: PORT must be an integer in [1, 65535], got: $PORT" >&2
    exit 1
fi
port_is_free() {
    "$CLIENT_PYTHON" - "$1" <<'PY'
import socket, sys
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try: sock.bind(("127.0.0.1", int(sys.argv[1])))
    except OSError: raise SystemExit(1)
PY
}
if ! port_is_free "$PORT"; then
    if (( port_was_explicit )); then
        echo "ERROR: requested PORT=$PORT is already in use." >&2
        exit 1
    fi
    PORT="$($CLIENT_PYTHON - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0)); print(sock.getsockname()[1])
PY
)"
    echo "Default port 5558 is busy; using free port $PORT."
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
    if grep -q "ready\." "$SERVER_LOG" 2>/dev/null; then ready=1; break; fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "ERROR: vla-server exited during startup" >&2
        sed -n '1,200p' "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 0.2
done
if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: timed out waiting for vla-server" >&2
    exit 1
fi

data_args=()
if [[ -n "${DATA_ROOT:-}" ]]; then data_args=(--data-root "$DATA_ROOT"); fi
"$CLIENT_PYTHON" "$REPO/eval/client/eval_turbovla_aloha_openloop.py" \
    --vla-addr "tcp://127.0.0.1:$PORT" \
    --gguf "$GGUF" --stats-json "$STATS" --tokenizer "$TOKENIZER" \
    --output-dir "$OUTPUT_DIR" "${data_args[@]}" "$@"

echo "server log: $SERVER_LOG"
