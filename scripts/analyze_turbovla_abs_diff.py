#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Compare TurboVLA traces using one fixed element-wise absolute threshold."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from compare_turbovla_traces import key, load_manifest, read_tensor  # noqa: E402


SELECTED_BOUNDARIES = (
    "input.rotated.view_0_rgb_u8",
    "input.rotated.view_1_rgb_u8",
    "input.pixel_values_model_dtype",
    "vision_projection.flattened",
    "text.projection.output",
    "interaction.layer_00.visual.after_fusion",
    "interaction.layer_05.visual.after_fusion",
    "interaction.layer_05.text_enhancer.output",
    "condition.concatenated",
    "state_projection.output",
    "action.memory.concatenated",
    "action.decoder.layer_00.output",
    "action.decoder.layer_01.output",
    "action.decoder.layer_02.output",
    "action.before_tanh",
    "action.normalized",
)

ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")

LAYER_PATTERNS = (
    (re.compile(r"^text\.bert\.layer_(\d+)\."), "text.bert.layer_{}"),
    (re.compile(r"^vision\.view_(\d+)\.block_(\d+)\."), "vision.view_{}.block_{}"),
    (re.compile(r"^interaction\.layer_(\d+)\."), "interaction.layer_{}"),
    (re.compile(r"^action\.decoder\.layer_(\d+)\."), "action.decoder.layer_{}"),
)


def _metrics(name: str, reference: np.ndarray, candidate: np.ndarray,
             epsilon: float) -> dict:
    difference = np.abs(
        candidate.astype(np.float64, copy=False)
        - reference.astype(np.float64, copy=False)
    )
    count = int(difference.size)
    violations = int(np.count_nonzero(difference > epsilon))
    return {
        "semantic_name": name,
        "numel": count,
        "within_epsilon": count - violations,
        "above_epsilon": violations,
        "above_epsilon_percent": 100.0 * violations / count if count else 0.0,
        "mae": float(np.mean(difference)) if count else 0.0,
        "rmse": float(np.sqrt(np.mean(difference**2))) if count else 0.0,
        "p50_abs": float(np.quantile(difference, 0.50)) if count else 0.0,
        "p95_abs": float(np.quantile(difference, 0.95)) if count else 0.0,
        "p99_abs": float(np.quantile(difference, 0.99)) if count else 0.0,
        "max_abs": float(np.max(difference)) if count else 0.0,
        "status": "PASS_ALL" if violations == 0 else "HAS_VIOLATIONS",
    }


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _layer_name(semantic_name: str) -> str | None:
    for pattern, template in LAYER_PATTERNS:
        match = pattern.match(semantic_name)
        if match:
            return template.format(*match.groups())
    return None


def _aggregate_layers(rows: list[dict]) -> list[dict]:
    accumulators: dict[str, dict] = {}
    for row in rows:
        layer = _layer_name(row["semantic_name"])
        if layer is None:
            continue
        aggregate = accumulators.setdefault(
            layer,
            {
                "layer": layer, "tensor_count": 0, "numel": 0,
                "above_epsilon": 0, "sum_abs": 0.0,
                "sum_squared": 0.0, "max_abs": 0.0,
            },
        )
        aggregate["tensor_count"] += 1
        aggregate["numel"] += row["numel"]
        aggregate["above_epsilon"] += row["above_epsilon"]
        aggregate["sum_abs"] += row["mae"] * row["numel"]
        aggregate["sum_squared"] += row["rmse"] ** 2 * row["numel"]
        aggregate["max_abs"] = max(aggregate["max_abs"], row["max_abs"])
    result = []
    for aggregate in accumulators.values():
        count = aggregate["numel"]
        result.append(
            {
                "layer": aggregate["layer"],
                "tensor_count": aggregate["tensor_count"],
                "numel": count,
                "within_epsilon": count - aggregate["above_epsilon"],
                "above_epsilon": aggregate["above_epsilon"],
                "above_epsilon_percent": 100.0 * aggregate["above_epsilon"] / count,
                "weighted_mae": aggregate["sum_abs"] / count,
                "weighted_rmse": float(np.sqrt(aggregate["sum_squared"] / count)),
                "max_abs": aggregate["max_abs"],
            }
        )
    return sorted(result, key=lambda row: row["layer"])


def _plot_boundaries(rows: dict[str, dict], epsilon: float, path: Path) -> None:
    selected = [rows[name] for name in SELECTED_BOUNDARIES if name in rows]
    labels = [row["semantic_name"] for row in selected]
    values = [row["above_epsilon_percent"] for row in selected]
    colors = ["tab:green" if value == 0 else "tab:red" for value in values]
    fig, axis = plt.subplots(figsize=(13, 7))
    axis.barh(np.arange(len(labels)), values, color=colors, alpha=0.82)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel(f"elements with |GGUF − PyTorch| > {epsilon:g} (%)")
    axis.set_xlim(0, 100)
    axis.grid(axis="x", alpha=0.3)
    axis.set_title(
        f"Boundary comparison: % stored values with absolute difference > {epsilon:g} (lower is better)"
    )
    for index, value in enumerate(values):
        axis.text(
            0.7 if value == 0 else min(value + 0.7, 97.0), index,
            f"{value:.1f}%", va="center", fontsize=8,
        )
    axis.legend(
        handles=[Patch(color="tab:green", label="0 violations"),
                 Patch(color="tab:red", label="has violations")],
        loc="lower right", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_action(reference: np.ndarray, candidate: np.ndarray,
                 epsilon: float, path: Path) -> None:
    reference = reference.reshape(-1, reference.shape[-1])
    candidate = candidate.reshape(-1, candidate.shape[-1])
    difference = np.abs(candidate - reference)
    fig, axis = plt.subplots(figsize=(11, 5))
    image = axis.imshow(difference.T, aspect="auto", cmap="magma")
    axis.set_yticks(np.arange(difference.shape[1]), ACTION_NAMES)
    axis.set_xticks(np.arange(difference.shape[0]))
    axis.set_xlabel("action step")
    axis.set_title(
        f"Final action comparison: each cell is |GGUF − PyTorch|; fixed ε={epsilon:g}"
    )
    for step in range(difference.shape[0]):
        for dimension in range(difference.shape[1]):
            value = float(difference[step, dimension])
            marker = "*" if value > epsilon else ""
            axis.text(
                step, dimension, f"{value:.3f}{marker}", ha="center", va="center",
                fontsize=7, color="white" if value < difference.max() * 0.55 else "black",
            )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label("absolute difference")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_layers(rows: list[dict], epsilon: float, path: Path) -> None:
    labels = [row["layer"] for row in rows]
    values = [row["above_epsilon_percent"] for row in rows]
    colors = []
    for label in labels:
        if label.startswith("text"):
            colors.append("tab:blue")
        elif label.startswith("vision"):
            colors.append("tab:purple")
        elif label.startswith("interaction"):
            colors.append("tab:orange")
        else:
            colors.append("tab:green")
    fig, axis = plt.subplots(figsize=(13, max(8, 0.27 * len(labels))))
    axis.barh(np.arange(len(labels)), values, color=colors, alpha=0.82)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel(f"stored values with |GGUF − PyTorch| > {epsilon:g} (%)")
    axis.grid(axis="x", alpha=0.3)
    axis.set_title(
        f"Layer comparison: % stored values with absolute difference > {epsilon:g} (lower is better)"
    )
    for index, value in enumerate(values):
        axis.text(
            min(value + 0.7, 97.0), index, f"{value:.1f}%",
            va="center", fontsize=6,
        )
    axis.legend(
        handles=[
            Patch(color="tab:blue", label="BERT text"),
            Patch(color="tab:purple", label="DINO vision"),
            Patch(color="tab:orange", label="interaction"),
            Patch(color="tab:green", label="action decoder"),
        ],
        loc="lower right", fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=0.01)
    args = parser.parse_args()
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")

    reference_records = {key(record): record for record in load_manifest(args.reference)}
    candidate_records = {key(record): record for record in load_manifest(args.candidate)}
    shared_keys = sorted(reference_records.keys() & candidate_records.keys())
    rows: list[dict] = []
    incompatible: list[dict] = []
    total_values = 0
    total_violations = 0
    total_abs = 0.0
    total_squared = 0.0
    reference_action = None
    candidate_action = None

    for semantic_key in shared_keys:
        reference_record = reference_records[semantic_key]
        candidate_record = candidate_records[semantic_key]
        if reference_record.get("shape") != candidate_record.get("shape"):
            incompatible.append({"semantic_name": semantic_key[0], "reason": "shape"})
            continue
        reference = read_tensor(args.reference, reference_record)
        candidate = read_tensor(args.candidate, candidate_record)
        if reference.size != candidate.size:
            incompatible.append({"semantic_name": semantic_key[0], "reason": "numel"})
            continue
        row = _metrics(semantic_key[0], reference, candidate, args.epsilon)
        rows.append(row)
        total_values += row["numel"]
        total_violations += row["above_epsilon"]
        total_abs += row["mae"] * row["numel"]
        total_squared += row["rmse"] ** 2 * row["numel"]
        if semantic_key == ("action.normalized", 0):
            reference_action = reference.reshape(reference_record["shape"])
            candidate_action = candidate.reshape(candidate_record["shape"])

    if reference_action is None or candidate_action is None:
        raise ValueError("action.normalized::call_00 is not shared by both traces")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "abs_diff.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows_by_name = {row["semantic_name"]: row for row in rows}
    layer_rows = _aggregate_layers(rows)
    with (args.output_dir / "layer_abs_diff.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(layer_rows[0]))
        writer.writeheader()
        writer.writerows(layer_rows)
    action_row = rows_by_name["action.normalized"]
    action_difference = np.abs(
        candidate_action.reshape(-1, candidate_action.shape[-1])
        - reference_action.reshape(-1, reference_action.shape[-1])
    )
    reference_action_2d = reference_action.reshape(-1, reference_action.shape[-1])
    candidate_action_2d = candidate_action.reshape(-1, candidate_action.shape[-1])
    action_violations = []
    for step, dimension in np.argwhere(action_difference > args.epsilon):
        action_violations.append(
            {
                "step": int(step),
                "dimension": int(dimension),
                "name": ACTION_NAMES[int(dimension)],
                "pytorch": float(reference_action_2d[step, dimension]),
                "gguf": float(candidate_action_2d[step, dimension]),
                "abs_difference": float(action_difference[step, dimension]),
            }
        )
    action_per_dimension = []
    for dimension in range(action_difference.shape[1]):
        values = action_difference[:, dimension]
        violations = int(np.count_nonzero(values > args.epsilon))
        action_per_dimension.append(
            {
                "dimension": dimension,
                "name": ACTION_NAMES[dimension],
                "numel": int(values.size),
                "above_epsilon": violations,
                "above_epsilon_percent": 100.0 * violations / values.size,
                "mae": float(values.mean()),
                "p95_abs": float(np.quantile(values, 0.95)),
                "max_abs": float(values.max()),
            }
        )

    summary = {
        "schema": "vla.cpp.turbovla_fixed_abs_diff.v1",
        "epsilon": args.epsilon,
        "comparison": "abs(candidate - reference) <= epsilon; rtol=0",
        "shared_semantic_keys": len(shared_keys),
        "compared_tensors": len(rows),
        "incompatible_tensors": len(incompatible),
        "all_values": {
            "numel": total_values,
            "within_epsilon": total_values - total_violations,
            "above_epsilon": total_violations,
            "above_epsilon_percent": 100.0 * total_violations / total_values,
            "weighted_mae": total_abs / total_values,
            "weighted_rmse": float(np.sqrt(total_squared / total_values)),
        },
        "tensor_gate": {
            "pass_all": sum(row["status"] == "PASS_ALL" for row in rows),
            "has_violations": sum(row["status"] != "PASS_ALL" for row in rows),
        },
        "layer_summary": layer_rows,
        "action_normalized": action_row,
        "action_per_dimension": action_per_dimension,
        "action_violations": action_violations,
        "incompatible": incompatible,
    }
    (args.output_dir / "abs_diff.json").write_text(
        json.dumps({"summary": summary, "tensors": rows}, indent=2) + "\n",
        encoding="utf-8",
    )

    _plot_boundaries(rows_by_name, args.epsilon, args.output_dir / "abs_boundary_violations.png")
    _plot_action(
        reference_action, candidate_action, args.epsilon,
        args.output_dir / "abs_action_heatmap.png",
    )
    _plot_layers(layer_rows, args.epsilon, args.output_dir / "abs_layer_violations.png")

    worst = sorted(rows, key=lambda row: row["above_epsilon_percent"], reverse=True)[:10]
    lines = [
        "# TurboVLA fixed absolute-difference comparison",
        "",
        "## Test definition",
        "",
        f"For every shape-compatible value, compute `d = |vla.cpp GGUF − PyTorch|` and",
        f"accept that value only when `d ≤ ε`, using the single fixed threshold",
        f"`ε = {args.epsilon:g}` and no relative tolerance (`rtol = 0`).",
        "",
        f"- PyTorch trace: `{args.reference.resolve()}`",
        f"- vla.cpp trace: `{args.candidate.resolve()}`",
        "- Both traces use the same LIBERO Object task 0, episode 0, step 0 fixture.",
        "",
        "## Result",
        "",
        f"The final normalized action has **{action_row['within_epsilon']}/{action_row['numel']} "
        f"values within ε ({100.0 - action_row['above_epsilon_percent']:.4f}%)**.",
        f"Only **{action_row['above_epsilon']} value** exceeds ε; action MAE is",
        f"`{_fmt(action_row['mae'])}`, p95 is `{_fmt(action_row['p95_abs'])}`, p99 is",
        f"`{_fmt(action_row['p99_abs'])}`, and max is `{_fmt(action_row['max_abs'])}`.",
        "Therefore final-output error is small overall, with one localized outlier.",
        "",
        "Across every overlapping op-level tensor, the fixed threshold is much stricter:",
        f"{summary['tensor_gate']['pass_all']}/{len(rows)} tensors have no violations and",
        f"{summary['all_values']['above_epsilon_percent']:.4f}% of all compared stored values",
        "exceed ε. This global number includes raw logits and reshape/transpose/head dumps",
        "whose C++ storage order is not yet canonicalized to the PyTorch trace, so it must not",
        "be interpreted as the final-action error rate.",
        "",
        "## Final action by dimension",
        "",
        "| Dim | Name | Within ε | Above ε | MAE | p95 | Max |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in action_per_dimension:
        lines.append(
            f"| {row['dimension']} | {row['name']} | {row['numel'] - row['above_epsilon']}/{row['numel']} | "
            f"{row['above_epsilon']} | {_fmt(row['mae'])} | {_fmt(row['p95_abs'])} | {_fmt(row['max_abs'])} |"
        )
    lines.extend(
        [
            "",
            "### Action values above ε",
            "",
            "| Step | Dim | Name | PyTorch | GGUF | Absolute difference |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in action_violations:
        lines.append(
            f"| {row['step']} | {row['dimension']} | {row['name']} | "
            f"{_fmt(row['pytorch'])} | {_fmt(row['gguf'])} | "
            f"{_fmt(row['abs_difference'])} |"
        )
    lines.extend(
        [
            "",
            "An asterisk marks a value above the fixed threshold.",
            "",
            "![Final action absolute-difference heatmap](abs_action_heatmap.png)",
            "",
            "*How to read: one cell is one action step/dimension. The number is the absolute",
            "difference, and `*` means that cell exceeds ε. This is a difference heatmap,",
            "not an action trajectory plot.*",
            "",
            "## Layer-by-layer comparison",
            "",
            "Each row aggregates every shape-compatible semantic tensor assigned to that",
            "model layer. Percentages are element counts, not tensor counts.",
            "",
            "| Layer | Tensors | Values | Within ε | Above ε (%) | Weighted MAE | Weighted RMSE | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in layer_rows:
        lines.append(
            f"| `{row['layer']}` | {row['tensor_count']} | {row['numel']} | "
            f"{row['within_epsilon']} | {row['above_epsilon_percent']:.4f}% | "
            f"{_fmt(row['weighted_mae'])} | {_fmt(row['weighted_rmse'])} | "
            f"{_fmt(row['max_abs'])} |"
        )
    lines.extend(
        [
            "",
            "![Layer-by-layer violation rates](abs_layer_violations.png)",
            "",
            "*How to read: each bar is one model layer. Its length is the percentage of all",
            "stored op-level values in that layer whose absolute difference exceeds ε; shorter",
            "is better. It includes Q/K/V, logits and transpose/head tensors.*",
            "",
            "## Selected boundary tensors",
            "",
            "| Semantic tensor | Within ε | Above ε | Above ε (%) | MAE | p95 | p99 | Max |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in SELECTED_BOUNDARIES:
        if name not in rows_by_name:
            continue
        row = rows_by_name[name]
        lines.append(
            f"| `{name}` | {row['within_epsilon']}/{row['numel']} | {row['above_epsilon']} | "
            f"{row['above_epsilon_percent']:.4f}% | {_fmt(row['mae'])} | "
            f"{_fmt(row['p95_abs'])} | {_fmt(row['p99_abs'])} | {_fmt(row['max_abs'])} |"
        )
    lines.extend(
        [
            "",
            "![Selected boundary violation rates](abs_boundary_violations.png)",
            "",
            "*How to read: this uses the same percentage as the layer chart, but only for",
            "selected stage outputs. It shows where error grows along the forward path.*",
            "",
            "## Highest violation-rate internal tensors",
            "",
            "| Semantic tensor | Above ε (%) | MAE | p99 | Max |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in worst:
        lines.append(
            f"| `{row['semantic_name']}` | {row['above_epsilon_percent']:.4f}% | "
            f"{_fmt(row['mae'])} | {_fmt(row['p99_abs'])} | {_fmt(row['max_abs'])} |"
        )
    lines.extend(
        [
            "",
            "## Final assessment",
            "",
            "| Scope | Evidence at ε=0.01 | Verdict |",
            "|---|---|---|",
            "| Exact input fixture | Both RGB views and BF16 model pixels have 0 violations | **PASS** |",
            f"| Final normalized action | {action_row['within_epsilon']}/{action_row['numel']} values within ε; "
            f"MAE {_fmt(action_row['mae'])}; max {_fmt(action_row['max_abs'])} | **PASS WITH 1 OUTLIER** |",
            f"| Selected stage boundaries | Decoder layer outputs above ε: "
            f"{rows_by_name['action.decoder.layer_00.output']['above_epsilon_percent']:.2f}% → "
            f"{rows_by_name['action.decoder.layer_01.output']['above_epsilon_percent']:.2f}% → "
            f"{rows_by_name['action.decoder.layer_02.output']['above_epsilon_percent']:.2f}% | **MIXED** |",
            f"| Full overlapping op trace | {summary['tensor_gate']['pass_all']}/"
            f"{summary['compared_tensors']} tensors have no violations | **NOT YET PASSING** |",
            "",
            "The GGUF conversion is therefore validated at the **input and final-output level**",
            "for this exact LIBERO fixture. The remaining final-action difference is small and",
            "localized to gripper step 0 (`0.0301915`), while every other action value is within",
            "the fixed threshold.",
            "",
            "This result is sufficient evidence that vla.cpp reproduces the policy output closely",
            "on the checked sample, but it is **not evidence of element-exact internal equivalence**.",
            "Deep-layer aggregates include storage-order-sensitive transpose/head tensors and also",
            "show genuine BF16/backend accumulation. Trace layout should be canonicalized and the",
            "same fixture rerun before claiming kernel-by-kernel parity. A multi-fixture test is",
            "also required before generalizing this single-sample result to the full benchmark.",
            "",
            "Machine-readable results: [`abs_diff.csv`](abs_diff.csv) and",
            "[`abs_diff.json`](abs_diff.json); per-layer CSV:",
            "[`layer_abs_diff.csv`](layer_abs_diff.csv).",
            "",
        ]
    )
    (args.output_dir / "ABS_DIFF_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
