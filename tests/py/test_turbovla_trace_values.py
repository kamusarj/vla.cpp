# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Element-by-element parity test for PyTorch and vla.cpp TurboVLA dumps.

Set ``VLA_TURBOVLA_REFERENCE_TRACE`` and ``VLA_TURBOVLA_CANDIDATE_TRACE`` to
the two ``forward_000000_rank_0`` directories to enable the integration test.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from compare_turbovla_traces import (  # noqa: E402
    compare_values,
    in_contract,
    key,
    load_manifest,
    metrics,
    read_tensor,
)


def _float_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if value < 0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _patterns() -> list[str]:
    value = os.environ.get("VLA_TURBOVLA_TRACE_PATTERN", "*")
    patterns = [item.strip() for item in value.split(",") if item.strip()]
    if not patterns:
        raise ValueError("VLA_TURBOVLA_TRACE_PATTERN contains no patterns")
    return patterns


def _selected(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _record_map(directory: Path) -> dict[tuple[str, int], dict]:
    records = load_manifest(directory)
    result = {}
    duplicates = []
    for record in records:
        record_key = key(record)
        if record_key in result:
            duplicates.append(record_key)
        result[record_key] = record
    assert not duplicates, f"duplicate semantic tensor keys in {directory}: {duplicates[:8]}"
    return result


def _format_examples(name: str, call_index: int, comparison: dict) -> list[str]:
    lines = [
        f"{name}::call_{call_index:02d}: "
        f"{comparison['mismatch_count']}/{comparison['value_count']} values differ"
    ]
    for example in comparison["examples"]:
        lines.append(
            f"  index={tuple(example['index'])} flat={example['flat_index']} "
            f"PT={example['reference']:.9g} C++={example['candidate']:.9g} "
            f"delta={example['delta']:+.9g} "
            f"allowed={example['allowed_error']:.9g}"
        )
    return lines


def test_compare_values_checks_all_values_and_reports_coordinates():
    reference = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    candidate = reference.copy()
    candidate[0, 1, 2] += 0.25
    candidate[1, 1, 1] -= 0.5

    result = compare_values(reference, candidate, reference.shape, atol=0.1, rtol=0.0)

    assert result["value_count"] == 12
    assert result["mismatch_count"] == 2
    assert result["examples"][0]["index"] == [0, 1, 2]
    assert result["examples"][1]["index"] == [1, 1, 1]
    assert result["examples"][0]["reference"] == pytest.approx(5.0)
    assert result["examples"][0]["candidate"] == pytest.approx(5.25)


@pytest.mark.parametrize(
    ("name", "level", "expected"),
    [
        ("vision.view_0.block_00.attn.logits", "op", True),
        ("action.normalized", "boundary", True),
        ("vision.view_0.block_00.attn.softmax.exp", "exhaustive", False),
        ("text.tokenizer.input_ids_unpadded", "op", False),
        ("state.normalize.mean", "op", False),
        ("action.denormalized", "boundary", False),
    ],
)
def test_model_op_v1_contract(name, level, expected):
    record = {"semantic_name": name, "required_level": level}
    assert in_contract(record, "model-op-v1") is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("action.normalized", True),
        ("action.denormalized", True),
        ("action.before_tanh", False),
        ("vision_projection.output", False),
    ],
)
def test_action_output_v1_contract(name, expected):
    record = {"semantic_name": name, "required_level": "boundary"}
    assert in_contract(record, "action-output-v1") is expected


def test_metrics_handles_structural_infinities_without_nan():
    reference = np.asarray([1.0, -np.inf, 3.0], dtype=np.float32)
    candidate = np.asarray([1.25, -np.inf, 3.0], dtype=np.float32)
    result = metrics(reference, candidate)
    assert result["max_abs_error"] == pytest.approx(0.25)
    assert result["mean_abs_error"] == pytest.approx(1.0 / 12.0)


def test_cpp_dump_attention_layout_is_canonical():
    candidate_value = os.environ.get("VLA_TURBOVLA_CANDIDATE_TRACE")
    if not candidate_value:
        pytest.skip("set VLA_TURBOVLA_CANDIDATE_TRACE to validate C++ trace layouts")
    candidate_dir = Path(candidate_value).expanduser().resolve()
    assert candidate_dir.is_dir(), f"candidate trace directory not found: {candidate_dir}"
    candidate = _record_map(candidate_dir)

    failures = []
    checked_value_heads = 0
    checked_contexts = 0
    for (semantic_name, call_index), record in sorted(candidate.items()):
        source_name = None
        if semantic_name.endswith(".v_heads"):
            prefix = semantic_name.removesuffix("v_heads")
            source_name = next(
                (name for name in (prefix + "v_linear", prefix + "v")
                 if (name, call_index) in candidate),
                None,
            )
        elif semantic_name.endswith(".v.transpose"):
            source_name = semantic_name.removesuffix(".v.transpose") + ".v_linear.bias_add"
        elif (semantic_name.endswith(".v_text.transpose")
              or semantic_name.endswith(".v_visual.transpose")):
            source_name = semantic_name.removesuffix(".transpose") + ".bias_add"

        if source_name and (source_name, call_index) in candidate:
            source_record = candidate[(source_name, call_index)]
            source = read_tensor(candidate_dir, source_record).reshape(source_record["shape"])
            dumped = read_tensor(candidate_dir, record).reshape(record["shape"])
            batch, length, dimension = source.shape
            heads = dumped.shape[1]
            expected = source.reshape(
                batch, length, heads, dimension // heads,
            ).transpose(0, 2, 1, 3)
            checked_value_heads += 1
            if not np.array_equal(expected, dumped):
                failures.append(
                    f"{semantic_name}: value-head layout differs from {source_name}; "
                    f"max_abs={np.max(np.abs(expected - dumped)):.9g}"
                )

        if semantic_name.endswith(".context_merged"):
            heads_name = semantic_name.removesuffix("context_merged") + "context_heads"
            if (heads_name, call_index) not in candidate:
                continue
            heads_record = candidate[(heads_name, call_index)]
            heads_value = read_tensor(candidate_dir, heads_record).reshape(
                heads_record["shape"]
            )
            merged = read_tensor(candidate_dir, record).reshape(record["shape"])
            if heads_value.ndim == 4:
                expected = heads_value.transpose(0, 2, 1, 3).reshape(merged.shape)
            else:
                batch, length, dimension = merged.shape
                batch_heads, heads_length, head_dimension = heads_value.shape
                assert heads_length == length and batch_heads % batch == 0
                expected = heads_value.reshape(
                    batch, batch_heads // batch, length, head_dimension,
                ).transpose(0, 2, 1, 3).reshape(batch, length, dimension)
            checked_contexts += 1
            if not np.array_equal(expected, merged):
                failures.append(
                    f"{semantic_name}: merged layout differs from {heads_name}; "
                    f"max_abs={np.max(np.abs(expected - merged)):.9g}"
                )

    assert checked_value_heads > 0, "candidate trace has no value-head tensors"
    assert checked_contexts > 0, "candidate trace has no merged-context tensors"
    assert not failures, "non-canonical C++ attention trace:\n" + "\n".join(failures[:32])


def test_pytorch_and_cpp_dump_values():
    reference_value = os.environ.get("VLA_TURBOVLA_REFERENCE_TRACE")
    candidate_value = os.environ.get("VLA_TURBOVLA_CANDIDATE_TRACE")
    if not reference_value and not candidate_value:
        pytest.skip(
            "set VLA_TURBOVLA_REFERENCE_TRACE and "
            "VLA_TURBOVLA_CANDIDATE_TRACE to compare real dumps"
        )
    assert reference_value and candidate_value, (
        "VLA_TURBOVLA_REFERENCE_TRACE and VLA_TURBOVLA_CANDIDATE_TRACE "
        "must be set together"
    )

    reference_dir = Path(reference_value).expanduser().resolve()
    candidate_dir = Path(candidate_value).expanduser().resolve()
    assert reference_dir.is_dir(), f"reference trace directory not found: {reference_dir}"
    assert candidate_dir.is_dir(), f"candidate trace directory not found: {candidate_dir}"

    reference = _record_map(reference_dir)
    candidate = _record_map(candidate_dir)
    patterns = _patterns()
    reference_keys = {item for item in reference if _selected(item[0], patterns)}
    candidate_keys = {item for item in candidate if _selected(item[0], patterns)}
    overlap = sorted(reference_keys & candidate_keys)
    assert overlap, f"no overlapping tensors match patterns {patterns}"

    require_all = _bool_env("VLA_TURBOVLA_TRACE_REQUIRE_ALL")
    if require_all:
        assert reference_keys == candidate_keys, (
            f"trace coverage differs; missing in C++={sorted(reference_keys - candidate_keys)[:20]}, "
            f"missing in PyTorch={sorted(candidate_keys - reference_keys)[:20]}"
        )

    atol = _float_env("VLA_TURBOVLA_TRACE_ATOL", 0.0)
    rtol = _float_env("VLA_TURBOVLA_TRACE_RTOL", 0.0)
    max_failed_tensors = int(os.environ.get("VLA_TURBOVLA_TRACE_MAX_FAILURES", "32"))
    assert max_failed_tensors > 0, "VLA_TURBOVLA_TRACE_MAX_FAILURES must be positive"

    failures = []
    failed_tensor_count = 0
    reported_tensor_count = 0
    checked_value_count = 0
    mismatch_value_count = 0
    for semantic_name, call_index in overlap:
        ref_record = reference[(semantic_name, call_index)]
        cand_record = candidate[(semantic_name, call_index)]
        if ref_record.get("shape") != cand_record.get("shape"):
            failed_tensor_count += 1
            if reported_tensor_count < max_failed_tensors:
                reported_tensor_count += 1
                failures.append(
                    f"{semantic_name}::call_{call_index:02d}: shape differs: "
                    f"PT={ref_record.get('shape')} C++={cand_record.get('shape')}"
                )
            continue
        if ref_record.get("layout", "") != cand_record.get("layout", ""):
            failed_tensor_count += 1
            if reported_tensor_count < max_failed_tensors:
                reported_tensor_count += 1
                failures.append(
                    f"{semantic_name}::call_{call_index:02d}: layout differs: "
                    f"PT={ref_record.get('layout', '')!r} "
                    f"C++={cand_record.get('layout', '')!r}"
                )
            continue

        ref_value = read_tensor(reference_dir, ref_record)
        cand_value = read_tensor(candidate_dir, cand_record)
        comparison = compare_values(
            ref_value, cand_value, ref_record["shape"], atol, rtol,
        )
        checked_value_count += comparison["value_count"]
        mismatch_value_count += comparison["mismatch_count"]
        if comparison["mismatch_count"]:
            failed_tensor_count += 1
            if reported_tensor_count < max_failed_tensors:
                reported_tensor_count += 1
                failures.extend(_format_examples(semantic_name, call_index, comparison))

    assert failed_tensor_count == 0, (
        f"element-wise trace parity failed: {failed_tensor_count}/{len(overlap)} tensors, "
        f"{mismatch_value_count}/{checked_value_count} checked values exceed "
        f"atol={atol}, rtol={rtol}. Showing at most {max_failed_tensors} tensor failures:\n"
        + "\n".join(failures)
    )
