# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Inspect or export one semantic tensor from a TurboVLA trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DTYPES = {
    "float32": np.dtype("<f4"),
    "int32": np.dtype("<i4"),
    "uint8": np.dtype("u1"),
}


def load_manifest(trace_dir: Path) -> list[dict]:
    manifest = trace_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--name", help="exact semantic_name to inspect")
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument("--list", action="store_true",
                        help="list matching semantic names instead of loading a tensor")
    parser.add_argument("--contains", default="",
                        help="filter used with --list")
    parser.add_argument("--full", action="store_true",
                        help="print every value (large tensors can produce huge output)")
    parser.add_argument("--precision", type=int, default=7)
    parser.add_argument("--output-npy", type=Path)
    parser.add_argument("--output-text", type=Path,
                        help="write values as text; tensors above 2-D flatten leading axes")
    args = parser.parse_args()

    records = load_manifest(args.trace_dir)
    if args.list:
        for record in records:
            name = record["semantic_name"]
            if args.contains in name:
                print(f"{name}  shape={record['shape']}  call={record['call_index']}")
        return 0
    if not args.name:
        parser.error("--name is required unless --list is used")

    matches = [record for record in records
               if record["semantic_name"] == args.name
               and int(record["call_index"]) == args.call_index]
    if not matches:
        parser.error(f"tensor not found: ({args.name!r}, {args.call_index})")
    if len(matches) != 1:
        parser.error(f"manifest contains {len(matches)} duplicate matches")
    record = matches[0]
    storage_dtype = record["storage_dtype"]
    if storage_dtype not in DTYPES:
        parser.error(f"unsupported storage dtype: {storage_dtype}")

    raw_path = args.trace_dir / record["file"]
    values = np.fromfile(raw_path, dtype=DTYPES[storage_dtype])
    expected = int(np.prod(record["shape"], dtype=np.int64))
    if values.size != expected:
        raise RuntimeError(f"raw tensor has {values.size} values, expected {expected}")
    tensor = values.reshape(record["shape"])

    finite = np.isfinite(tensor)
    finite_values = tensor[finite]
    print(f"semantic_name: {record['semantic_name']}")
    print(f"shape: {tuple(record['shape'])}  layout: {record.get('layout', '')}")
    print(f"dtype: {storage_dtype}  operation: {record.get('operation', '')}")
    print(f"file: {raw_path}")
    if finite_values.size:
        print("stats: "
              f"min={finite_values.min():.{args.precision}g} "
              f"max={finite_values.max():.{args.precision}g} "
              f"mean={finite_values.mean():.{args.precision}g} "
              f"std={finite_values.std():.{args.precision}g}")
    print("values:")
    threshold = tensor.size if args.full else 256
    with np.printoptions(precision=args.precision, suppress=False,
                         threshold=threshold, edgeitems=3, linewidth=160):
        print(tensor)

    if args.output_npy:
        args.output_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_npy, tensor)
        print(f"wrote npy: {args.output_npy}")
    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        matrix = tensor if tensor.ndim <= 2 else tensor.reshape(-1, tensor.shape[-1])
        np.savetxt(args.output_text, matrix, fmt=f"%.{args.precision}g")
        print(f"wrote text: {args.output_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
