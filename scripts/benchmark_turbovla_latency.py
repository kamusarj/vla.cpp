#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Benchmark TurboVLA PyTorch or vla.cpp on one exact parity fixture."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _metrics(samples: list[float]) -> dict[str, float | int]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
        "stdev": float(statistics.pstdev(samples)),
    }


def _load_fixture_arrays(fixture: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    images = []
    for index in range(2):
        raw = np.load(fixture / f"raw/view_{index}_rgb_u8.npy")
        images.append(np.ascontiguousarray(raw[::-1, ::-1], dtype=np.uint8))
    tokens = np.load(fixture / "exact_inputs/input_ids_i64.npy").reshape(-1)
    state = np.load(fixture / "exact_inputs/state_normalized_f32.npy").reshape(-1)
    return images, tokens, state


def _benchmark_pytorch(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(args.turbovla_root.resolve()))
    import torch

    from turbovla.debug import FixtureReader
    from turbovla.evaluation.model_loader import load_turbovla_for_inference

    loaded = load_turbovla_for_inference(
        checkpoint_path=args.checkpoint,
        dinov3_path=str(args.dinov3_path),
        bert_path=str(args.bert_path),
        device="cuda",
        precision="bf16",
        strict=True,
        deterministic=False,
    )
    reader = FixtureReader(args.fixture)
    pixels = reader.load_tensor(
        "exact_inputs/pixel_values_model_dtype.npy", bf16_bits=True,
    ).to(device=loaded.device)
    state = reader.load_tensor(
        "exact_inputs/state_model_dtype.npy", bf16_bits=True,
    ).to(device=loaded.device)
    instruction = reader.instruction
    loaded.model.set_parity_tracer(None)

    def forward():
        with torch.inference_mode():
            if args.pytorch_scope == "vision":
                return loaded.model.encode_vision(pixels)
            return loaded.model([instruction], {"dinov3": pixels}, state)

    for _ in range(args.warmup):
        output = forward()
    torch.cuda.synchronize()

    samples = []
    for _ in range(args.iterations):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        output = forward()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1.0e6)

    return {
        "backend": "pytorch",
        "scope": (
            "vision_encoder_and_projector_preprocessed_pixels"
            if args.pytorch_scope == "vision"
            else "model_forward_preprocessed_pixels_and_state_includes_bert_tokenization"
        ),
        "latency_ms": {"synchronized_wall": _metrics(samples)},
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "gpu": torch.cuda.get_device_name(0),
        "output_shape": list(output.shape),
    }


def _compile_proto(proto: Path, output: Path):
    subprocess.run(
        [
            "protoc",
            f"--proto_path={proto.parent}",
            f"--python_out={output}",
            str(proto),
        ],
        check=True,
    )
    sys.path.insert(0, str(output))
    import vla_pb2

    return vla_pb2


def _benchmark_cpp(args: argparse.Namespace) -> dict:
    import zmq

    images, tokens, state = _load_fixture_arrays(args.fixture)
    with tempfile.TemporaryDirectory(prefix="turbovla-latency-proto-") as directory:
        pb = _compile_proto(args.proto, Path(directory))
        request = pb.PredictRequest(request_id=1)
        for image in images:
            proto_image = request.images.add()
            proto_image.encoding = pb.Image.RGB_U8
            proto_image.height, proto_image.width = image.shape[:2]
            proto_image.data = image.tobytes()
        request.lang_tokens.extend(int(value) for value in tokens)
        request.state.extend(float(value) for value in state)
        payload = request.SerializeToString()

        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.RCVTIMEO, args.timeout_ms)
        socket.connect(args.addr)

        def forward():
            started = time.perf_counter_ns()
            # The immutable request payload remains alive for the whole run, so
            # pyzmq can hand it to libzmq without making a ~400 KiB copy on
            # every request.  Receive into a Frame for the same reason.
            socket.send(payload, copy=False)
            response = pb.PredictResponse()
            response_frame = socket.recv(copy=False)
            response.ParseFromString(response_frame.buffer)
            wall_ms = (time.perf_counter_ns() - started) / 1.0e6
            if response.error:
                raise RuntimeError(response.error)
            return response, wall_ms

        for _ in range(args.warmup):
            response, _ = forward()

        wall_samples = []
        total_samples = []
        vision_samples = []
        inference_samples = []
        for _ in range(args.iterations):
            response, wall_ms = forward()
            wall_samples.append(wall_ms)
            total_samples.append(float(response.latency_ms_total))
            vision_samples.append(float(response.latency_ms_vision))
            inference_samples.append(float(response.latency_ms_inference))
        socket.close()
        context.term()

    return {
        "backend": "vla.cpp",
        "scope": "server_predict_rotated_rgb_tokens_and_normalized_state",
        "latency_ms": {
            "client_wall_local_zmq": _metrics(wall_samples),
            "server_predict_total": _metrics(total_samples),
            "server_vision": _metrics(vision_samples),
            "server_inference_after_vision": _metrics(inference_samples),
        },
        "output_shape": [1, int(response.chunk_size), int(response.action_dim)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pytorch", "cpp"), required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turbovla-root", type=Path, default=Path("/home/linh/Desktop/TurboVLA"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dinov3-path", type=Path)
    parser.add_argument("--bert-path", type=Path)
    parser.add_argument("--pytorch-scope", choices=("full", "vision"), default="full")
    parser.add_argument("--addr", default="tcp://127.0.0.1:5567")
    parser.add_argument("--proto", type=Path, default=Path("src/serving/vla.proto"))
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must both be positive")
    if args.backend == "pytorch":
        missing = [
            name for name in ("checkpoint", "dinov3_path", "bert_path")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("PyTorch backend requires " + ", ".join(f"--{x.replace('_', '-')}" for x in missing))
        result = _benchmark_pytorch(args)
    else:
        result = _benchmark_cpp(args)
    result.update({
        "schema": "vla.cpp.turbovla_latency.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "fixture": str(args.fixture.resolve()),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "host": platform.node(),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
