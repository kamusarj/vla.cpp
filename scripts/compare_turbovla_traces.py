# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compare overlapping PyTorch and vla.cpp TurboVLA semantic tensor traces."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
from pathlib import Path

import numpy as np


DTYPES = {
    "float32": "<f4", "float16": "<f2", "bfloat16": "<u2",
    "int64": "<i8", "int32": "<i4", "int16": "<i2",
    "int8": "i1", "uint8": "u1", "bool": "u1",
}

TRACE_LEVEL_ORDER = {"boundary": 0, "layer": 1, "op": 2, "exhaustive": 3}

# These values live outside the C++ model API: the replay helper already passes
# rotated/model-ready images, token IDs, normalized state, and consumes normalized
# actions.  They are deliberately excluded from the model-op parity contract.
MODEL_OP_V1_RUNTIME_IO = (
    "input.raw.*",
    "input.normalized.*",
    "input.normalize.*",
    "state.raw",
    "state.normalize.*",
    "text.instruction_*",
    "text.tokenizer.*",
    "action.denormalize.*",
    "action.denormalized",
    "action.gripper.*",
    "action.first_step",
    "action.executed_steps",
)

ACTION_OUTPUT_V1 = frozenset(("action.normalized", "action.denormalized"))


def load_manifest(directory: Path) -> list[dict]:
    path = directory / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(record: dict) -> tuple[str, int]:
    return record["semantic_name"], int(record.get("call_index", 0))


def in_contract(record: dict, contract: str | None) -> bool:
    if contract is None:
        return True
    if contract == "action-output-v1":
        return record["semantic_name"] in ACTION_OUTPUT_V1
    if contract != "model-op-v1":
        raise ValueError(f"unsupported trace contract {contract!r}")
    level = record.get("required_level", "boundary")
    if level not in TRACE_LEVEL_ORDER:
        raise ValueError(
            f"invalid required_level {level!r} for {record.get('semantic_name')!r}"
        )
    if TRACE_LEVEL_ORDER[level] > TRACE_LEVEL_ORDER["op"]:
        return False
    name = record["semantic_name"]
    return not any(fnmatch.fnmatchcase(name, pattern) for pattern in MODEL_OP_V1_RUNTIME_IO)


def read_tensor(directory: Path, record: dict) -> np.ndarray:
    storage = record.get("storage_dtype", "float32").removeprefix("torch.")
    if storage not in DTYPES:
        raise TypeError(f"unsupported storage dtype {storage!r}")
    value = np.fromfile(directory / record["file"], dtype=DTYPES[storage])
    if storage == "bfloat16":
        value = (value.astype(np.uint32) << 16).view(np.float32)
    return value


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)
    finite_pair = np.isfinite(reference) & np.isfinite(candidate)
    equal_nonfinite = ~finite_pair & (reference == candidate)
    incompatible_nonfinite = ~(finite_pair | equal_nonfinite)
    error = np.zeros(reference.shape, dtype=np.float64)
    error[finite_pair] = np.abs(candidate[finite_pair] - reference[finite_pair])
    error[incompatible_nonfinite] = np.inf
    finite_reference = reference[finite_pair]
    finite_candidate = candidate[finite_pair]
    centered_ref = finite_reference - finite_reference.mean() if finite_reference.size else finite_reference
    centered_cand = finite_candidate - finite_candidate.mean() if finite_candidate.size else finite_candidate
    cosine_den = np.linalg.norm(finite_reference) * np.linalg.norm(finite_candidate)
    corr_den = np.linalg.norm(centered_ref) * np.linalg.norm(centered_cand)
    return {
        "max_abs_error": finite(error.max(initial=0.0)),
        "mean_abs_error": finite(error.mean()) if error.size else 0.0,
        "rmse": finite(np.sqrt(np.mean(error ** 2))) if error.size else 0.0,
        "cosine_similarity": finite(np.dot(finite_reference, finite_candidate) / cosine_den)
        if cosine_den else 1.0,
        "pearson_correlation": finite(np.dot(centered_ref, centered_cand) / corr_den)
        if corr_den else 1.0,
    }


def compare_values(reference: np.ndarray, candidate: np.ndarray,
                   shape: list[int] | tuple[int, ...], atol: float, rtol: float,
                   max_examples: int = 8) -> dict:
    """Compare every scalar and retain bounded, coordinate-level diagnostics.

    The tolerance order deliberately matches the historical ``np.allclose``
    call in this script: ``reference`` is the first operand and ``candidate``
    is the second, so the relative term is based on ``abs(candidate)``.
    """
    reference = np.asarray(reference).reshape(-1)
    candidate = np.asarray(candidate).reshape(-1)
    if reference.size != candidate.size:
        raise ValueError(
            f"value count differs: reference={reference.size}, "
            f"candidate={candidate.size}"
        )
    tensor_shape = tuple(int(value) for value in shape)
    expected = int(np.prod(tensor_shape, dtype=np.int64)) if tensor_shape else 1
    if reference.size != expected:
        raise ValueError(
            f"shape {tensor_shape} contains {expected} values, got {reference.size}"
        )

    close = np.isclose(reference, candidate, atol=atol, rtol=rtol,
                       equal_nan=False)
    mismatch_flat = np.flatnonzero(~close)
    examples = []
    for flat_index in mismatch_flat[:max_examples]:
        flat_index = int(flat_index)
        coordinate = tuple(
            int(value) for value in np.unravel_index(flat_index, tensor_shape)
        ) if tensor_shape else ()
        ref_value = float(reference[flat_index])
        cand_value = float(candidate[flat_index])
        examples.append(
            {
                "flat_index": flat_index,
                "index": list(coordinate),
                "reference": ref_value,
                "candidate": cand_value,
                "delta": cand_value - ref_value,
                "allowed_error": float(atol + rtol * abs(cand_value)),
            }
        )
    mismatch_count = int(mismatch_flat.size)
    return {
        "value_count": int(reference.size),
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / reference.size if reference.size else 0.0,
        "examples": examples,
    }


def compare(reference_dir: Path, candidate_dir: Path, output_dir: Path,
            atol: float, rtol: float, require_all: bool,
            contract: str | None = None) -> dict:
    reference = load_manifest(reference_dir)
    candidate = load_manifest(candidate_dir)
    ref_map = {key(record): record for record in reference}
    cand_map = {key(record): record for record in candidate}
    expected_reference = [record for record in reference if in_contract(record, contract)]
    expected_keys = {key(record) for record in expected_reference}
    overlap = [record for record in expected_reference if key(record) in cand_map]
    missing_candidate = [key(record) for record in expected_reference if key(record) not in cand_map]
    missing_reference = [name for name in cand_map if name not in ref_map]
    if not overlap:
        raise ValueError("the two traces have no matching semantic tensor names")

    rows = []
    failed = 0
    for ref_record in overlap:
        semantic_key = key(ref_record)
        cand_record = cand_map[semantic_key]
        row = {
            "semantic_name": semantic_key[0],
            "call_index": semantic_key[1],
            "reference_trace_id": ref_record["trace_id"],
            "candidate_trace_id": cand_record["trace_id"],
            "canonical_required_level": ref_record.get("required_level", "boundary"),
            "candidate_required_level": cand_record.get("required_level", "boundary"),
        }
        if ref_record.get("shape") != cand_record.get("shape"):
            row["status"] = "FAIL_SHAPE"
        elif ref_record.get("layout", "") != cand_record.get("layout", ""):
            row["status"] = "FAIL_LAYOUT"
        else:
            ref_value = read_tensor(reference_dir, ref_record)
            cand_value = read_tensor(candidate_dir, cand_record)
            if ref_value.size != cand_value.size:
                row["status"] = "FAIL_SHAPE"
            else:
                row.update(metrics(ref_value, cand_value))
                value_comparison = compare_values(
                    ref_value, cand_value, ref_record["shape"], atol, rtol,
                )
                row.update(
                    {
                        "value_count": value_comparison["value_count"],
                        "mismatch_count": value_comparison["mismatch_count"],
                        "mismatch_ratio": value_comparison["mismatch_ratio"],
                    }
                )
                if value_comparison["examples"]:
                    first = value_comparison["examples"][0]
                    row.update(
                        {
                            "first_mismatch_flat_index": first["flat_index"],
                            "first_mismatch_index": json.dumps(first["index"]),
                            "first_mismatch_reference": first["reference"],
                            "first_mismatch_candidate": first["candidate"],
                            "first_mismatch_delta": first["delta"],
                            "first_mismatch_allowed_error": first["allowed_error"],
                        }
                    )
                exact = np.array_equal(ref_value, cand_value)
                close = value_comparison["mismatch_count"] == 0
                row["status"] = "PASS_EXACT" if exact else (
                    "PASS_TOLERANCE" if close else "FAIL_TOLERANCE"
                )
        failed += not row["status"].startswith("PASS")
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with (output_dir / "compare.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "compare.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output_dir / "missing_in_cpp.txt").write_text(
        "\n".join(f"{name}::call_{index:02d}" for name, index in missing_candidate)
    )
    (output_dir / "missing_in_python.txt").write_text(
        "\n".join(f"{name}::call_{index:02d}" for name, index in missing_reference)
    )
    status = "FAIL" if failed or (require_all and missing_candidate) else "PASS"
    coverage_status = "PASS" if not missing_candidate else "FAIL"
    summary = {
        "status": status,
        "contract": contract,
        "require_all": require_all,
        "coverage_status": coverage_status,
        "expected_count": len(expected_keys),
        "overlap_count": len(overlap),
        "passing_count": len(overlap) - failed,
        "failing_count": failed,
        "missing_in_cpp_count": len(missing_candidate),
        "missing_in_python_count": len(missing_reference),
        "candidate_outside_contract_count": len(set(cand_map) - expected_keys),
        "atol": atol,
        "rtol": rtol,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="PyTorch forward_* trace directory")
    parser.add_argument("--candidate", type=Path, required=True,
                        help="vla.cpp forward_* trace directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--require-all", action="store_true",
                        help="also fail when an expected PyTorch tensor is absent in C++")
    parser.add_argument(
        "--contract", choices=("model-op-v1", "action-output-v1"), default=None,
        help=("Use a fixed PyTorch-authoritative parity scope. model-op-v1 includes "
              "boundary/layer/op model tensors and excludes runtime preprocessing/"
              "postprocessing that is outside the C++ model API; action-output-v1 "
              "checks normalized and denormalized action outputs only."),
    )
    args = parser.parse_args()
    summary = compare(args.reference, args.candidate, args.output,
                      args.atol, args.rtol, args.require_all, args.contract)
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
