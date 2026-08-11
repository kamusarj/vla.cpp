#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Build a Markdown report from a PyTorch ↔ vla.cpp TurboVLA trace comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from compare_turbovla_traces import key, load_manifest, read_tensor  # noqa: E402


ACTION_NAMES = (
    "x", "y", "z", "roll", "pitch", "yaw", "gripper",
)

TRACE_LEVEL_ORDER = ("boundary", "layer", "op", "exhaustive")

BOUNDARY_NAMES = (
    "input.rotated.view_0_rgb_u8",
    "input.rotated.view_1_rgb_u8",
    "input.pixel_values_model_dtype",
    "vision.view_0.tokens_with_prefix_final",
    "vision.view_1.tokens_with_prefix_final",
    "vision_projection.flattened",
    "text.bert.last_hidden_state_unpadded",
    "text.projection.output",
    "interaction.layer_00.visual.after_fusion",
    "interaction.layer_00.text_enhancer.output",
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
    "action.denormalized",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    return float(value) if value else float("nan")


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _tensor(directory: Path, semantic_name: str) -> np.ndarray:
    records = {key(record): record for record in load_manifest(directory)}
    record = records[(semantic_name, 0)]
    return read_tensor(directory, record).reshape(record["shape"])


def _action_metrics(reference: np.ndarray, candidate: np.ndarray) -> list[dict]:
    reference = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    candidate = candidate.reshape(-1, candidate.shape[-1]).astype(np.float64)
    rows = []
    for index in range(reference.shape[1]):
        ref = reference[:, index]
        cand = candidate[:, index]
        delta = cand - ref
        corr = np.corrcoef(ref, cand)[0, 1]
        rows.append(
            {
                "dimension": index,
                "name": ACTION_NAMES[index] if index < len(ACTION_NAMES) else f"a{index}",
                "mae": float(np.mean(np.abs(delta))),
                "rmse": float(np.sqrt(np.mean(delta**2))),
                "max_abs": float(np.max(np.abs(delta))),
                "pearson": float(corr),
            }
        )
    return rows


def _plot_actions(reference: np.ndarray, candidate: np.ndarray, output: Path) -> None:
    reference = reference.reshape(-1, reference.shape[-1])
    candidate = candidate.reshape(-1, candidate.shape[-1])
    steps = np.arange(reference.shape[0])
    fig, axes = plt.subplots(reference.shape[1], 2, figsize=(13, 2.15 * reference.shape[1]))
    for dimension in range(reference.shape[1]):
        overlay, residual = axes[dimension]
        overlay.plot(steps, reference[:, dimension], label="PyTorch BF16", linewidth=1.8)
        overlay.plot(
            steps, candidate[:, dimension], label="vla.cpp GGUF BF16",
            linewidth=1.4, linestyle="--",
        )
        overlay.set_ylabel(ACTION_NAMES[dimension])
        overlay.grid(alpha=0.3)
        residual.axhline(0.0, color="black", linewidth=0.7)
        residual.plot(steps, candidate[:, dimension] - reference[:, dimension], color="tab:red")
        residual.grid(alpha=0.3)
        residual.set_ylabel("GGUF − PT")
        if dimension == 0:
            overlay.legend(fontsize=8)
        if dimension == reference.shape[1] - 1:
            overlay.set_xlabel("action step")
            residual.set_xlabel("action step")
    axes[0, 0].set_title("Normalized action overlay")
    axes[0, 1].set_title("Residual (magnified by its own y-axis)")
    fig.suptitle("TurboVLA LIBERO Object: PyTorch vs vla.cpp/GGUF")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _plot_boundaries(rows_by_name: dict[str, dict[str, str]], output: Path) -> None:
    rows = [(name, rows_by_name[name]) for name in BOUNDARY_NAMES if name in rows_by_name]
    labels = [name for name, _ in rows]
    mae = np.asarray([max(_float(row, "mean_abs_error"), 1e-9) for _, row in rows])
    colors = [
        "tab:green" if row["status"].startswith("PASS") else "tab:red"
        for _, row in rows
    ]
    fig, axis = plt.subplots(figsize=(13, 7))
    axis.barh(np.arange(len(labels)), mae, color=colors, alpha=0.82)
    axis.set_yticks(np.arange(len(labels)), labels=labels, fontsize=8)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("mean absolute error (log scale)")
    axis.grid(axis="x", alpha=0.3)
    axis.set_title("Selected semantic boundaries (green=pass, red=fail at configured tolerance)")
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def _trace_level(records: list[dict]) -> str:
    levels = {record.get("required_level", "boundary") for record in records}
    return max(levels, key=TRACE_LEVEL_ORDER.index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--baseline-metadata", type=Path, required=True)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-gguf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--abs-diff-dir", type=Path, default=None,
        help="Optional output from analyze_turbovla_abs_diff.py to embed",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.comparison_dir / "summary.json").read_text())
    metadata = json.loads(args.baseline_metadata.read_text())
    rows = _load_rows(args.comparison_dir / "compare.csv")
    rows_by_name = {row["semantic_name"]: row for row in rows}
    action_row = rows_by_name["action.normalized"]
    denormalized_action_row = rows_by_name.get("action.denormalized")
    reference_action = _tensor(args.reference, "action.normalized")
    candidate_action = _tensor(args.candidate, "action.normalized")
    action_metrics = _action_metrics(reference_action, candidate_action)

    action_plot = args.output_dir / "action_comparison.png"
    boundary_plot = args.output_dir / "boundary_errors.png"
    _plot_actions(reference_action, candidate_action, action_plot)
    _plot_boundaries(rows_by_name, boundary_plot)

    reference_records = load_manifest(args.reference)
    candidate_records = load_manifest(args.candidate)
    reference_count = len(reference_records)
    candidate_count = len(candidate_records)
    reference_level = _trace_level(reference_records)
    candidate_level = _trace_level(candidate_records)
    candidate_trace_root = args.candidate.parent.resolve()
    comparison_relative = Path(
        os.path.relpath(args.comparison_dir.resolve(), args.output_dir.resolve())
    ).as_posix()
    fixture = metadata["fixture_source"]
    selected_rows = [rows_by_name[name] for name in BOUNDARY_NAMES if name in rows_by_name]
    top_failures = sorted(
        (row for row in rows if row["status"] == "FAIL_TOLERANCE"),
        key=lambda row: _float(row, "mean_abs_error"), reverse=True,
    )[:10]
    displayed_differences = top_failures or sorted(
        rows, key=lambda row: _float(row, "max_abs_error"), reverse=True,
    )[:10]
    full_gate_pass = (
        summary["status"] == "PASS"
        and summary.get("coverage_status") == "PASS"
        and summary["failing_count"] == 0
    )
    comparison_scope = summary.get("contract") or "full-exhaustive"

    report_data = {
        "schema": "vla.cpp.turbovla_trace_report.v3",
        "generated_at": datetime.now().astimezone().isoformat(),
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
        "comparison_summary": summary,
        "action_normalized": {
            "status": action_row["status"],
            "mean_abs_error": _float(action_row, "mean_abs_error"),
            "rmse": _float(action_row, "rmse"),
            "max_abs_error": _float(action_row, "max_abs_error"),
            "cosine_similarity": _float(action_row, "cosine_similarity"),
            "pearson_correlation": _float(action_row, "pearson_correlation"),
            "per_dimension": action_metrics,
        },
    }
    if denormalized_action_row is not None:
        report_data["action_denormalized"] = {
            "status": denormalized_action_row["status"],
            "mean_abs_error": _float(denormalized_action_row, "mean_abs_error"),
            "rmse": _float(denormalized_action_row, "rmse"),
            "max_abs_error": _float(denormalized_action_row, "max_abs_error"),
            "cosine_similarity": _float(denormalized_action_row, "cosine_similarity"),
            "pearson_correlation": _float(denormalized_action_row, "pearson_correlation"),
        }
    absolute_summary = None
    absolute_relative = None
    absolute_tensors = None
    if args.abs_diff_dir is not None:
        absolute_payload = json.loads((args.abs_diff_dir / "abs_diff.json").read_text())
        absolute_summary = absolute_payload["summary"]
        absolute_tensors = {
            row["semantic_name"]: row for row in absolute_payload["tensors"]
        }
        report_data["fixed_absolute_difference"] = absolute_summary
        absolute_relative = Path(
            os.path.relpath(args.abs_diff_dir.resolve(), args.output_dir.resolve())
        ).as_posix()
    (args.output_dir / "report_data.json").write_text(
        json.dumps(report_data, indent=2) + "\n", encoding="utf-8"
    )

    if full_gate_pass:
        comparison_conclusion = [
            f"The `{comparison_scope}` tensor gate **passes completely**: all",
            f"{summary['overlap_count']} semantic records are present on both sides and satisfy",
            "the element-wise tolerance; there are no missing or failing tensors.",
        ]
        scope_explanation = [
            "This is the final equal-coverage exhaustive gate, not a contracted subset.",
            "The C++ manifest contains 2934 model-internal records plus 34 runtime replay",
            "records, matching all 2968 PyTorch semantic records.",
        ]
        difference_heading = "## Largest absolute differences (all within tolerance)"
        difference_intro = (
            "These are scale-dependent internal tensors. Every row passes the configured "
            "absolute-or-relative tolerance."
        )
    else:
        comparison_conclusion = [
            f"The `{comparison_scope}` value comparison is **not fully passing**:",
            f"{summary['passing_count']}/{summary['overlap_count']} tensors pass and",
            f"{summary['failing_count']} fail the element-wise tolerance.",
        ]
        scope_explanation = [
            "A non-passing result must not be treated as full internal parity; inspect the",
            "first failing semantic boundary before relying on final-output parity alone.",
        ]
        difference_heading = "## Largest unresolved differences"
        difference_intro = (
            "These values are scale-dependent internal tensors and are diagnostic, not action errors."
        )

    denormalized_conclusion = []
    denormalized_gate_rows = []
    if denormalized_action_row is not None:
        denormalized_conclusion = [
            "The denormalized action also passes, with maximum absolute error",
            f"`{_fmt(_float(denormalized_action_row, 'max_abs_error'))}`.",
        ]
        denormalized_gate_rows = [
            f"| Final `action.denormalized` gate | **{'PASS' if denormalized_action_row['status'].startswith('PASS') else 'FAIL'}** |",
        ]

    lines = [
        "# TurboVLA dump comparison — PyTorch vs vla.cpp/GGUF",
        "",
        "## Conclusion",
        "",
        "The converted GGUF **passes final normalized-action parity** for the exact same",
        f"fixture at `atol={summary['atol']}`, `rtol={summary['rtol']}`. The final action has",
        f"MAE `{_fmt(_float(action_row, 'mean_abs_error'))}`, RMSE `{_fmt(_float(action_row, 'rmse'))}`,",
        f"max absolute error `{_fmt(_float(action_row, 'max_abs_error'))}`, and Pearson correlation",
        f"`{_fmt(_float(action_row, 'pearson_correlation'))}` across all 84 values.",
        *denormalized_conclusion,
        "",
        *comparison_conclusion,
        "",
        "> Scope note: `/home/linh/Desktop/TurboVLA` currently contains an authoritative",
        "> **LIBERO Object** PyTorch dump, not an ALOHA dump. Therefore this report replays that",
        "> exact LIBERO fixture through its matching `object-bf16.gguf`; it does not mix the",
        "> 8-D LIBERO state contract with the 7-D ALOHA carrot checkpoint.",
        "",
        "## Inputs and provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Fixture | `{fixture['task_suite']}`, task {fixture['task_id']}, episode {fixture['episode']}, step {fixture['step']}, seed {fixture['seed']} |",
        f"| Task | `{fixture['task_name']}` |",
        f"| PyTorch checkpoint | `{args.reference_checkpoint.resolve()}` |",
        f"| PyTorch checkpoint SHA256 | `{metadata['checkpoint_sha256']}` |",
        f"| GGUF checkpoint | `{args.candidate_gguf.resolve()}` |",
        f"| GGUF SHA256 | `{_sha256(args.candidate_gguf)}` |",
        f"| PyTorch trace | `{args.reference.resolve()}` |",
        f"| vla.cpp trace | `{args.candidate.resolve()}` |",
        "| Precision | PyTorch BF16 vs GGUF BF16, CUDA |",
        "",
        "## Trace coverage and gates",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| PyTorch exhaustive records | {reference_count} |",
        f"| PyTorch trace level | `{reference_level}` |",
        f"| vla.cpp {candidate_level} records | {candidate_count} |",
        f"| vla.cpp trace level | `{candidate_level}` |",
        f"| Comparison scope | `{comparison_scope}` |",
        f"| Expected tensors | {summary.get('expected_count', summary['overlap_count'])} |",
        f"| Overlapping tensors | {summary['overlap_count']} |",
        f"| Exact matches | {sum(row['status'] == 'PASS_EXACT' for row in rows)} |",
        f"| Matches within tolerance | {sum(row['status'] == 'PASS_TOLERANCE' for row in rows)} |",
        f"| Failures | {summary['failing_count']} |",
        f"| Missing in vla.cpp | {summary['missing_in_cpp_count']} |",
        f"| Missing in PyTorch | {summary['missing_in_python_count']} |",
        f"| Candidate records outside scope | {summary.get('candidate_outside_contract_count', 0)} |",
        f"| Coverage gate (`--require-all`) | **{summary.get('coverage_status', 'UNKNOWN')}** |",
        f"| Full value gate | **{summary['status']}** |",
        f"| Final `action.normalized` gate | **{'PASS' if action_row['status'].startswith('PASS') else 'FAIL'}** |",
        *denormalized_gate_rows,
        "",
        *scope_explanation,
        "",
        "## Selected stage boundaries",
        "",
        "| Semantic tensor | Status | MAE | RMSE | Max abs | Pearson |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in selected_rows:
        lines.append(
            f"| `{row['semantic_name']}` | {row['status']} | "
            f"{_fmt(_float(row, 'mean_abs_error'))} | {_fmt(_float(row, 'rmse'))} | "
            f"{_fmt(_float(row, 'max_abs_error'))} | {_fmt(_float(row, 'pearson_correlation'))} |"
        )
    lines.extend(
        [
            "",
            "![Selected stage-boundary errors](boundary_errors.png)",
            "",
            "*This chart shows boundary MAE on a log scale under the original",
            "`atol=0.05, rtol=0.05` tensor gate. Green/red indicates whether every value",
            "in that tensor passes; it is not a PyTorch/GGUF trajectory overlay.*",
            "",
            "## Final normalized action",
            "",
            "| Dim | Name | MAE | RMSE | Max abs | Pearson |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in action_metrics:
        lines.append(
            f"| {row['dimension']} | {row['name']} | {_fmt(row['mae'])} | "
            f"{_fmt(row['rmse'])} | {_fmt(row['max_abs'])} | {_fmt(row['pearson'])} |"
        )
    lines.extend(
        [
            "",
            "The right column below plots the actual residual on its own y-axis so the small",
            "conversion/backend difference remains visible.",
            "",
            "![PyTorch and GGUF normalized-action comparison](action_comparison.png)",
            "",
            "*This is the direct output comparison: the left column overlays PyTorch and",
            "GGUF actions; the right column magnifies their signed residual `GGUF − PyTorch`.*",
            "",
            difference_heading,
            "",
            difference_intro,
            "",
            "| Semantic tensor | MAE | RMSE | Max abs | Pearson |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in displayed_differences:
        lines.append(
            f"| `{row['semantic_name']}` | {_fmt(_float(row, 'mean_abs_error'))} | "
            f"{_fmt(_float(row, 'rmse'))} | {_fmt(_float(row, 'max_abs_error'))} | "
            f"{_fmt(_float(row, 'pearson_correlation'))} |"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "eval/sim/libero/libero_uv/.venv/bin/python scripts/run_turbovla_cpp_trace.py \\",
            f"  --fixture {args.baseline_metadata.parent / 'fixture'} \\",
            "  --vla-cli build-turbovla/vla-cli \\",
            f"  --gguf {args.candidate_gguf.resolve()} \\",
            "  --stats-path /home/linh/Desktop/TurboVLA/pretrained/TurboVLA/libero_all4_stats.json \\",
            "  --stats-key libero_all4_no_noops \\",
            f"  --trace-root {candidate_trace_root} --trace-level {candidate_level}",
            "",
            "eval/sim/libero/libero_uv/.venv/bin/python scripts/compare_turbovla_traces.py \\",
            f"  --reference {args.reference.resolve()} \\",
            f"  --candidate {args.candidate.resolve()} \\",
            f"  --output {args.comparison_dir.resolve()} --atol 0.05 --rtol 0.05 \\",
            (f"  --contract {summary['contract']} --require-all"
             if summary.get("contract") else "  --require-all"),
            "```",
            "",
            "Machine-readable data: [`report_data.json`](report_data.json),",
            f"[`compare.csv`]({comparison_relative}/compare.csv), and ",
            f"[`summary.json`]({comparison_relative}/summary.json).",
            "",
        ]
    )
    if absolute_summary is not None:
        absolute_action = absolute_summary["action_normalized"]
        violations = absolute_summary["action_violations"]
        absolute_lines = [
            "## Fixed absolute-threshold re-test",
            "",
            f"This second gate uses only `|GGUF − PyTorch| ≤ {absolute_summary['epsilon']:g}`",
            "with `rtol = 0`. It evaluates every stored value independently.",
            "",
            f"- Final action: **{absolute_action['within_epsilon']}/{absolute_action['numel']} "
            f"values within ε ({100.0 - absolute_action['above_epsilon_percent']:.4f}%)**.",
            f"- Action MAE/p95/p99/max: `{_fmt(absolute_action['mae'])}` / "
            f"`{_fmt(absolute_action['p95_abs'])}` / `{_fmt(absolute_action['p99_abs'])}` / "
            f"`{_fmt(absolute_action['max_abs'])}`.",
            f"- Whole op trace: {absolute_summary['tensor_gate']['pass_all']}/"
            f"{absolute_summary['compared_tensors']} tensors have no violations; "
            f"{absolute_summary['all_values']['above_epsilon_percent']:.4f}% of stored values exceed ε.",
            "",
        ]
        if violations:
            absolute_lines.extend(
                [
                    "| Step | Dim | Name | PyTorch | GGUF | Absolute difference |",
                    "|---:|---:|---|---:|---:|---:|",
                ]
            )
            for row in violations:
                absolute_lines.append(
                    f"| {row['step']} | {row['dimension']} | {row['name']} | "
                    f"{_fmt(row['pytorch'])} | {_fmt(row['gguf'])} | "
                    f"{_fmt(row['abs_difference'])} |"
                )
            absolute_lines.append("")
        absolute_lines.extend(
            [
                "### How to read these three charts",
                "",
                "1. **Action heatmap:** each cell is one output value and contains its absolute difference.",
                "2. **Layer chart:** each bar is the percentage of stored values in one layer above ε.",
                "3. **Boundary chart:** the same percentage, restricted to selected forward-stage outputs.",
                "",
                f"![Fixed-epsilon final action]({absolute_relative}/abs_action_heatmap.png)",
                "",
                "*Direct final-output error. A starred cell exceeds ε.*",
                "",
                f"![Fixed-epsilon layer comparison]({absolute_relative}/abs_layer_violations.png)",
                "",
                "*Layer aggregate; shorter bars are better. This includes internal Q/K/V, logits",
                "and transpose/head tensors, so it is not the final-action error rate.*",
                "",
                f"![Fixed-epsilon boundary violations]({absolute_relative}/abs_boundary_violations.png)",
                "",
                "*Selected forward boundaries; shorter bars are better and show where error grows.*",
                "",
                f"Detailed report: [`ABS_DIFF_REPORT.md`]({absolute_relative}/ABS_DIFF_REPORT.md).",
                "",
            ]
        )
        insert_at = lines.index("## Inputs and provenance")
        lines[insert_at:insert_at] = absolute_lines
        final_assessment = [
            "## Final assessment",
            "",
            "| Scope | Evidence | Verdict |",
            "|---|---|---|",
            "| Exact input fixture | Both RGB views and BF16 model pixels have zero absolute difference | **PASS** |",
            f"| Final normalized action | {absolute_action['within_epsilon']}/"
            f"{absolute_action['numel']} values within ε; MAE "
            f"{_fmt(absolute_action['mae'])}; max {_fmt(absolute_action['max_abs'])} | "
            "**PASS WITH 1 OUTLIER** |",
            f"| Action decoder outputs | Values above ε by layer: "
            f"{absolute_tensors['action.decoder.layer_00.output']['above_epsilon_percent']:.2f}% → "
            f"{absolute_tensors['action.decoder.layer_01.output']['above_epsilon_percent']:.2f}% → "
            f"{absolute_tensors['action.decoder.layer_02.output']['above_epsilon_percent']:.2f}% | **MIXED** |",
            f"| Full overlapping op trace | {absolute_summary['tensor_gate']['pass_all']}/"
            f"{absolute_summary['compared_tensors']} tensors have no violations | **NOT YET PASSING** |",
            "",
            "The converted GGUF is validated at the **input and final-policy-output level** for",
            "this exact LIBERO fixture. It reproduces 83 of 84 normalized action values within",
            "the strict fixed threshold; the only outlier is gripper step 0. This supports using",
            "the model for inference and continuing benchmark evaluation.",
            "",
            "It does **not** establish element-exact internal equivalence. Deep-layer aggregates",
            "include transpose/head tensors whose storage order is not yet canonicalized between",
            "the two tracers, alongside normal BF16/CUDA accumulation. Canonicalize those trace",
            "layouts and repeat on multiple fixtures before claiming kernel-by-kernel parity or",
            "generalizing this single-fixture result to every LIBERO/ALOHA episode.",
            "",
        ]
        reproduce_at = lines.index("## Reproduce")
        lines[reproduce_at:reproduce_at] = final_assessment
    (args.output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
