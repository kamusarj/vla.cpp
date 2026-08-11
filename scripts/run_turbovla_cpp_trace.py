# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Replay a TurboVLA PyTorch parity fixture through vla-cli with tracing."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve_statistics(fixture: Path, stats_path: Path | None,
                        stats_key: str | None) -> tuple[Path, str]:
    if stats_path is not None and stats_key:
        return stats_path, stats_key
    baseline_metadata = fixture.parent / "baseline_metadata.json"
    if baseline_metadata.is_file():
        metadata = json.loads(baseline_metadata.read_text())
        command = metadata.get("environment", {}).get("command", [])
        cwd = Path(metadata.get("environment", {}).get("cwd", fixture.parent))
        try:
            inferred_path = Path(command[command.index("--stats-path") + 1])
            inferred_key = command[command.index("--stats-key") + 1]
            if not inferred_path.is_absolute():
                inferred_path = cwd / inferred_path
            return inferred_path, inferred_key
        except (ValueError, IndexError):
            pass
    raise ValueError("exhaustive replay requires --stats-path and --stats-key")


def _append_runtime_trace(output: Path, fixture: Path,
                          stats_path: Path, stats_key: str) -> None:
    manifest_path = output / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text().splitlines()
               if line.strip()]
    by_name = {record["semantic_name"]: record for record in records}
    next_trace_id = max(record["trace_id"] for record in records) + 1
    runtime_dir = output / "tensors" / "99_runtime_replay"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_summary_rows: list[list[object]] = []

    def emit(name: str, value: np.ndarray | float | list, shape: tuple[int, ...],
             layout: str, operation: str, level: str,
             source_dtype: str | None = None) -> None:
        nonlocal next_trace_id
        if name in by_name:
            return
        array = np.asarray(value)
        if array.dtype == np.bool_ or array.dtype == np.uint8:
            array = np.ascontiguousarray(array, dtype=np.uint8)
            storage, suffix = "uint8", "u8"
            source = source_dtype or ("torch.bool" if array.dtype == np.bool_ else "torch.uint8")
        elif np.issubdtype(array.dtype, np.integer):
            array = np.ascontiguousarray(array, dtype="<i8")
            storage, suffix, source = "int64", "i64le", source_dtype or "torch.int64"
        else:
            array = np.ascontiguousarray(array, dtype="<f4")
            storage, suffix, source = "float32", "f32le", source_dtype or "torch.float32"
        safe_name = name.replace("/", "_")
        relative = Path("tensors") / "99_runtime_replay" / (
            f"{next_trace_id:06d}__{safe_name}__call_00.{suffix}.bin")
        array.tofile(output / relative)
        record = {
            "trace_id": next_trace_id,
            "semantic_name": name,
            "call_index": 0,
            "module_path": None,
            "operation": operation,
            "io": "intermediate",
            "stage": "runtime_replay",
            "required_level": level,
            "shape": list(shape),
            "layout": layout,
            "source_dtype": source,
            "storage_dtype": storage,
            "endianness": "little",
            "contiguous": True,
            "numel": int(array.size),
            "file": str(relative),
        }
        records.append(record)
        by_name[name] = record
        numeric = array.astype(np.float64, copy=False).reshape(-1)
        runtime_summary_rows.append([
            next_trace_id, name, 0, level,
            "[" + ",".join(str(item) for item in shape) + "]", layout,
            storage, int(array.size),
            float(numeric.min()) if numeric.size else 0.0,
            float(numeric.max()) if numeric.size else 0.0,
            float(numeric.mean()) if numeric.size else 0.0,
            float(numeric.std()) if numeric.size else 0.0,
            float(np.abs(numeric).max()) if numeric.size else 0.0,
            float(np.linalg.norm(numeric)) if numeric.size else 0.0,
            int(np.isnan(numeric).sum()), int(np.isinf(numeric).sum()),
            str(relative),
        ])
        next_trace_id += 1

    metadata = json.loads((fixture / "metadata.json").read_text())
    preprocessing = metadata["preprocessing"]
    factor = np.float32(preprocessing.get("rescale_factor", 1.0 / 255.0))
    image_mean = np.asarray(preprocessing.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    image_std = np.asarray(preprocessing.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32)
    for index in range(2):
        raw = np.load(fixture / f"raw/view_{index}_rgb_u8.npy")
        rotated = np.ascontiguousarray(raw[::-1, ::-1])
        rescaled = rotated.astype(np.float32) * factor
        normalized = (rescaled.transpose(2, 0, 1) - image_mean[:, None, None]) / image_std[:, None, None]
        emit(f"input.raw.view_{index}_rgb_u8", raw, raw.shape, "H,W,C",
             "libero_observation", "boundary", "torch.uint8")
        emit(f"input.rescaled.view_{index}", rescaled, rescaled.shape, "H,W,C",
             "multiply", "exhaustive")
        emit(f"input.normalized.view_{index}", normalized, normalized.shape, "C,H,W",
             "normalize", "op")
    emit("input.rescale.factor", factor, (), "", "constant", "exhaustive")
    emit("input.normalize.mean", image_mean, (3,), "C", "constant", "op")
    emit("input.normalize.std", image_std, (3,), "C", "constant", "op")

    statistics = json.loads(stats_path.read_text())[stats_key]
    state_stats = statistics.get("state", statistics.get("proprio"))
    action_stats = statistics["action"]
    state_raw = np.load(fixture / "raw/state_raw_f32.npy").astype(np.float32)
    state_mean = np.asarray(state_stats["mean"], dtype=np.float32)
    state_std = np.asarray(state_stats["std"], dtype=np.float32)
    emit("state.raw", state_raw, state_raw.shape, "B,D", "libero_state", "boundary")
    emit("state.normalize.mean", state_mean, state_mean.shape, "D", "constant", "op")
    emit("state.normalize.std", state_std, state_std.shape, "D", "constant", "op")
    emit("state.centered", state_raw - state_mean, state_raw.shape, "B,D", "subtract", "exhaustive")
    instruction_bytes = np.frombuffer((fixture / "instruction.txt").read_text().encode("utf-8"), dtype=np.uint8)
    emit("text.instruction_utf8_bytes", instruction_bytes, instruction_bytes.shape,
         "BYTES", "utf8_encode", "boundary", "torch.uint8")

    unpadded = int(metadata["tokenizer"]["unpadded_length"])
    input_ids = np.load(fixture / "exact_inputs/input_ids_i64.npy")[:, :unpadded]
    attention = np.load(fixture / "exact_inputs/attention_mask_u8.npy")[:, :unpadded]
    self_attention = np.load(fixture / "exact_inputs/text_self_attention_mask_u8.npy")[:, :unpadded, :unpadded]
    positions = np.load(fixture / "exact_inputs/position_ids_i64.npy")[:, :unpadded]
    token_types = np.zeros_like(input_ids)
    emit("text.tokenizer.input_ids_unpadded", input_ids, input_ids.shape, "B,L", "tokenizer", "op")
    emit("text.tokenizer.token_type_ids_unpadded", token_types, token_types.shape, "B,L", "tokenizer", "op")
    emit("text.tokenizer.attention_mask_unpadded", attention, attention.shape, "B,L", "tokenizer", "op", "torch.bool")
    emit("text.tokenizer.self_attention_mask_unpadded", self_attention, self_attention.shape,
         "B,L,L", "tokenizer", "op", "torch.bool")
    emit("text.tokenizer.position_ids_unpadded", positions, positions.shape, "B,L", "tokenizer", "op")

    action_record = by_name["action.normalized"]
    action = np.fromfile(output / action_record["file"], dtype="<f4").reshape(action_record["shape"])
    minimum = np.asarray(action_stats["min"][:6], dtype=np.float32)
    maximum = np.asarray(action_stats["max"][:6], dtype=np.float32)
    source = action[..., 6]
    action_input = action[..., :6]
    plus_one = action_input + np.float32(1.0)
    action_range = maximum - minimum
    scaled = np.float32(0.5) * plus_one * action_range
    arm = scaled + minimum
    positive, negative = source > 0.0, source < 0.0
    gripper = np.where(positive, np.float32(1.0),
                       np.where(negative, np.float32(-1.0), np.float32(1.0)))
    denormalized = np.concatenate([arm, gripper[..., None]], axis=-1)
    emit("action.denormalize.action_min", minimum, minimum.shape, "A6", "constant", "op")
    emit("action.denormalize.action_max", maximum, maximum.shape, "A6", "constant", "op")
    emit("action.denormalize.input", action_input, action_input.shape, "B,T,A6", "slice", "op")
    emit("action.denormalize.plus_one", plus_one, plus_one.shape, "B,T,A6", "add", "exhaustive")
    emit("action.denormalize.range", action_range, action_range.shape, "A6", "subtract", "exhaustive")
    emit("action.denormalize.scaled", scaled, scaled.shape, "B,T,A6", "multiply", "exhaustive")
    emit("action.denormalize.arm_output", arm, arm.shape, "B,T,A6", "add", "op")
    emit("action.gripper.source", source, source.shape, "B,T", "slice", "op")
    emit("action.gripper.deadband", np.float32(0.0), (), "", "constant", "exhaustive")
    emit("action.gripper.positive_mask", positive, positive.shape, "B,T", "greater", "exhaustive", "torch.bool")
    emit("action.gripper.negative_mask", negative, negative.shape, "B,T", "less", "exhaustive", "torch.bool")
    emit("action.gripper.output", gripper, gripper.shape, "B,T", "where", "op")
    emit("action.denormalized", denormalized, denormalized.shape, "B,T,A", "concat", "boundary")
    emit("action.first_step", denormalized[:, :1], denormalized[:, :1].shape, "B,1,A", "slice", "boundary")
    emit("action.executed_steps", denormalized, denormalized.shape, "B,E,A", "slice", "boundary")

    manifest_path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n"
                                     for record in records))
    with (output / "summary.csv").open("a", newline="") as stream:
        csv.writer(stream).writerows(runtime_summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True,
                        help="fixture directory containing raw/ and exact_inputs/")
    parser.add_argument("--vla-cli", type=Path, default=Path("build/vla-cli"))
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--trace-level", choices=("boundary", "layer", "op", "exhaustive"),
                        default="boundary")
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--stats-key")
    args = parser.parse_args()

    for path in (args.vla_cli, args.gguf, args.fixture):
        if not path.exists():
            parser.error(f"path does not exist: {path}")
    output = args.trace_root / "forward_000000_rank_0"
    if output.exists():
        parser.error(f"trace already exists: {output}; choose a new --trace-root")

    ids = np.load(args.fixture / "exact_inputs/input_ids_i64.npy").reshape(-1)
    state = np.load(args.fixture / "exact_inputs/state_normalized_f32.npy").reshape(-1)
    args.trace_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="turbovla-cpp-trace-") as tmp:
        images = []
        for index in range(2):
            raw = np.load(args.fixture / f"raw/view_{index}_rgb_u8.npy")
            rotated = np.ascontiguousarray(raw[::-1, ::-1])
            image_path = Path(tmp) / f"view_{index}_rotated.png"
            Image.fromarray(rotated, mode="RGB").save(image_path)
            images.extend(("--image", str(image_path)))

        command = [
            str(args.vla_cli), "--ckpt", str(args.gguf), *images,
            "--tokens", ",".join(str(int(value)) for value in ids),
            "--state", ",".join(format(float(value), ".9g") for value in state),
            "--pretty",
        ]
        environment = os.environ.copy()
        environment.update(
            VLA_TURBOVLA_TRACE_ROOT=str(args.trace_root),
            VLA_TURBOVLA_TRACE_LEVEL=args.trace_level,
            VLA_TURBOVLA_TRACE_MAX_FORWARDS="1",
        )
        subprocess.run(command, env=environment, check=True)

    if args.trace_level == "exhaustive":
        stats_path, stats_key = _resolve_statistics(args.fixture, args.stats_path, args.stats_key)
        _append_runtime_trace(output, args.fixture, stats_path, stats_key)

    manifest = output / "manifest.jsonl"
    if not manifest.is_file():
        raise RuntimeError(f"vla.cpp did not produce {manifest}")
    count = sum(bool(line.strip()) for line in manifest.read_text().splitlines())
    print(f"wrote {count} C++ trace tensors to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
