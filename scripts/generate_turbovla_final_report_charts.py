#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Generate the retained figures for the consolidated TurboVLA report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _latency_comparison(root: Path, output: Path) -> None:
    latency = root / "artifacts/latency"
    pytorch = _json(latency / "pytorch.json")["latency_ms"]["synchronized_wall"]
    cpp = _json(latency / "cpp_final_tcp_zerocopy_borrowed.json")["latency_ms"]
    rows = (
        ("PyTorch\nmodel", pytorch),
        ("C++\npredict", cpp["server_predict_total"]),
        ("C++ local\nTCP/ZMQ", cpp["client_wall_local_zmq"]),
    )
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    for offset, metric, color in ((-0.18, "median", "#2878b5"), (0.18, "p95", "#f28e2b")):
        values = [row[1][metric] for row in rows]
        bars = axis.bar(x + offset, values, width=0.34, label=metric, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set_xticks(x, [row[0] for row in rows])
    axis.set_ylabel("Latency (ms)")
    axis.set_ylim(0, 27)
    axis.set_title("TurboVLA final latency: PyTorch vs vla.cpp")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(fig, output / "latency_comparison.png")


def _trace_overview(root: Path, output: Path) -> None:
    rows = _json(root / "artifacts/parity/exhaustive/compare.json")
    counts = {
        "Exact": sum(row["status"] == "PASS_EXACT" for row in rows),
        "Within tolerance": sum(row["status"] == "PASS_TOLERANCE" for row in rows),
        "Failure": sum(row["status"].startswith("FAIL") for row in rows),
    }
    fig, axis = plt.subplots(figsize=(7.4, 5.4))
    colors = ["#59a14f", "#4e79a7", "#e15759"]
    bars = axis.bar(counts.keys(), counts.values(), color=colors)
    axis.bar_label(bars, padding=4, fontweight="bold")
    axis.set_ylabel("Semantic tensor records")
    axis.set_title("Exhaustive trace result: 2968/2968 PASS")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output / "trace_match_overview.png")


def _error_distribution(root: Path, output: Path) -> None:
    rows = _json(root / "artifacts/parity/exhaustive/compare.json")
    mean_error = np.asarray([row["mean_abs_error"] for row in rows], dtype=np.float64)
    max_error = np.asarray([row["max_abs_error"] for row in rows], dtype=np.float64)
    positive_mean = np.maximum(mean_error, 1e-12)
    positive_max = np.maximum(max_error, 1e-12)
    bins = np.logspace(-12, 1, 55)
    fig, axis = plt.subplots(figsize=(9.2, 5.5))
    axis.hist(positive_mean, bins=bins, alpha=0.72, label="mean absolute error")
    axis.hist(positive_max, bins=bins, alpha=0.58, label="max absolute error")
    axis.set_xscale("log")
    axis.set_xlabel("Absolute error (zeros displayed at 1e-12)")
    axis.set_ylabel("Tensor count")
    axis.set_title("Error distribution across 2968 semantic tensors")
    axis.grid(alpha=0.22)
    axis.legend()
    _save(fig, output / "trace_error_distribution.png")


def _top_tensor_errors(root: Path, output: Path) -> None:
    rows = _json(root / "artifacts/parity/exhaustive/compare.json")
    selected = sorted(rows, key=lambda row: row["max_abs_error"], reverse=True)[:15]
    selected.reverse()
    labels = [row["semantic_name"] for row in selected]
    values = [row["max_abs_error"] for row in selected]
    colors = ["#4e79a7" if row["status"].startswith("PASS") else "#e15759" for row in selected]
    fig, axis = plt.subplots(figsize=(13.5, 7.2))
    bars = axis.barh(np.arange(len(labels)), values, color=colors)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
    axis.set_xlabel("Maximum absolute error")
    axis.set_title("Largest per-tensor absolute deltas (all displayed rows pass tolerance)")
    axis.grid(axis="x", alpha=0.25)
    axis.bar_label(bars, fmt="%.3g", padding=3, fontsize=7)
    _save(fig, output / "trace_top_tensor_errors.png")


def _stage(name: str) -> str:
    if name.startswith("vision."):
        return "vision"
    if name.startswith("text."):
        return "text"
    if name.startswith("interaction."):
        return "interaction"
    if name.startswith("action."):
        return "action"
    if name.startswith("state"):
        return "state"
    if name.startswith("input."):
        return "input/runtime"
    if name.startswith("condition."):
        return "condition"
    return "other"


def _stage_summary(root: Path, output: Path) -> None:
    rows = _json(root / "artifacts/parity/exhaustive/compare.json")
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[_stage(row["semantic_name"])].append(row)
    labels = sorted(groups)
    weighted_mae = []
    maxima = []
    for label in labels:
        count = sum(row["value_count"] for row in groups[label])
        weighted_mae.append(
            sum(row["mean_abs_error"] * row["value_count"] for row in groups[label]) / count
        )
        maxima.append(max(row["max_abs_error"] for row in groups[label]))
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    axis.bar(x - 0.18, np.maximum(weighted_mae, 1e-12), 0.36,
             label="value-weighted MAE", color="#4e79a7")
    axis.bar(x + 0.18, np.maximum(maxima, 1e-12), 0.36,
             label="largest max abs", color="#f28e2b")
    axis.set_yscale("log")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel("Absolute error (log scale)")
    axis.set_title("Exhaustive trace error by semantic stage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(fig, output / "trace_stage_error_summary.png")


def _action_dimensions(root: Path, output: Path) -> None:
    metrics = _json(root / "artifacts/parity/report_data.json")["action_normalized"]["per_dimension"]
    labels = [row["name"] for row in metrics]
    x = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10.2, 5.5))
    width = 0.25
    for offset, key, label, color in (
        (-width, "mae", "MAE", "#4e79a7"),
        (0.0, "rmse", "RMSE", "#59a14f"),
        (width, "max_abs", "max abs", "#e15759"),
    ):
        axis.bar(x + offset, [row[key] for row in metrics], width, label=label, color=color)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Absolute error")
    axis.set_title("Normalized action error by dimension")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save(fig, output / "action_dimension_errors.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_root", type=Path)
    args = parser.parse_args()
    root = args.report_root.resolve()
    output = root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _latency_comparison(root, output)
    _trace_overview(root, output)
    _error_distribution(root, output)
    _top_tensor_errors(root, output)
    _stage_summary(root, output)
    _action_dimensions(root, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
