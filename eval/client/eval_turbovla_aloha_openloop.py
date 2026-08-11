#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Evaluate TurboVLA GGUF open-loop on the real ALOHA LeRobot dataset.

This mirrors DuyBaoDOCer's ``experiments/aloha/eval_openloop.py`` protocol:
at anchors spaced by 12 frames, predict a fixed [12, 7] chunk from the two
recorded camera frames and recorded state, unnormalize without clipping, and
compare every valid chunk element with the recorded raw absolute joint action.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from client.turbovla_aloha import TurboVlaAlohaStats  # noqa: E402
from client.vla_cpp_client import VlaCppClient  # noqa: E402


ACTION_DIM = 7
CHUNK_SIZE = 12
NMSE_GATE_THRESHOLD = 0.1
CAMERA_KEYS = (
    "observation.images.color.high",
    "observation.images.color.wrist_left",
)
JOINT_NAMES = (
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
)
VAL_EPISODES = (7, 9, 16, 21, 30, 32, 36, 47, 51, 52, 65, 67, 74, 79, 82, 85, 93, 95, 99, 126)
TRAIN_SAMPLE_EPISODES = (0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23)
PYTORCH_REFERENCE_URL = (
    "https://github.com/DuyBaoDOCer/TurboVLA/blob/main/notebooks/RESULTS.md"
)
PYTORCH_REFERENCE = {
    "val": {
        "n_points": 2883,
        "nmse_per_joint": [0.020959286019206047, 0.019992660731077194, 0.01870501972734928, 0.1096000000834465, 0.04919964820146561, 0.11473291367292404, 0.06671946495771408],
        "nmse_total": 0.05712985619902611,
        "mse_per_joint": [0.0022330868523567915, 0.009222717955708504, 0.003571211826056242, 0.004790329374372959, 0.004873286932706833, 0.007130431942641735, 0.0023344906512647867],
        "mse_total": 0.004879365209490061,
        "mae_per_joint": [0.02956974506378174, 0.048576805740594864, 0.03907235339283943, 0.025305423885583878, 0.047360703349113464, 0.03576558083295822, 0.016252079978585243],
        "mae_total": 0.0345575287938118,
    },
    "train_sample": {
        "n_points": 3139,
        "nmse_per_joint": [0.014891412109136581, 0.015099354088306427, 0.006870099809020758, 0.00638544699177146, 0.008317364379763603, 0.006260185968130827, 0.006423387676477432],
        "nmse_total": 0.009178178384900093,
        "mse_per_joint": [0.0015475802356377244, 0.006794858258217573, 0.0012895651161670685, 0.00051261973567307, 0.0007866406231187284, 0.0006010163924656808, 0.00021741437376476824],
        "mse_total": 0.0016785276820883155,
        "mae_per_joint": [0.01129063032567501, 0.024036221206188202, 0.013274386525154114, 0.011499638669192791, 0.013313277624547482, 0.012734307907521725, 0.005633379332721233],
        "mae_total": 0.013111690990626812,
    },
}


def _parse_episode_list(value: str) -> list[int]:
    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    if not result:
        raise ValueError("episode list is empty")
    return result


def _discover_data_root(explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    candidates.extend(
        [
            Path.home() / "Desktop/octo/octo-pytorch/data/aloha_carrot_easy",
            Path("data/aloha"),
        ]
    )
    for candidate in candidates:
        if (candidate / "meta/info.json").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "ALOHA LeRobot dataset not found; pass --data-root. Searched: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_episode(data_root: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    path = data_root / "data/chunk-000" / f"episode_{episode_id:06d}.parquet"
    table = pd.read_parquet(path, columns=["observation.state", "action"])
    state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
    action = np.stack(table["action"].to_numpy()).astype(np.float32)
    if state.shape != action.shape or state.shape[1:] != (ACTION_DIM,):
        raise ValueError(f"episode {episode_id}: invalid state/action shapes {state.shape}/{action.shape}")
    return state, action


def _decode_anchor_frames(
    video_path: Path,
    anchors: list[int],
    expected_length: int,
    image_size: int,
) -> dict[int, np.ndarray]:
    wanted = set(anchors)
    result: dict[int, np.ndarray] = {}
    count = 0
    container = av.open(str(video_path))
    try:
        for index, frame in enumerate(container.decode(video=0)):
            count = index + 1
            if index not in wanted:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            resized = Image.fromarray(rgb).resize(
                (image_size, image_size), Image.Resampling.BILINEAR
            )
            chw = np.transpose(np.asarray(resized, dtype=np.float32) / 255.0, (2, 0, 1))
            result[index] = np.ascontiguousarray(chw)
    finally:
        container.close()
    if count != expected_length:
        raise ValueError(
            f"{video_path}: decoded {count} frames but parquet has {expected_length} rows"
        )
    missing = wanted.difference(result)
    if missing:
        raise ValueError(f"{video_path}: missing requested frames {sorted(missing)}")
    return result


def _compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    diff = pred - gt
    mse = np.mean(diff**2, axis=0)
    mae = np.mean(np.abs(diff), axis=0)
    variance = np.var(gt, axis=0)
    nmse = mse / np.clip(variance, 1e-8, None)
    return {
        "n_points": int(pred.shape[0]),
        "mse_per_joint": mse.tolist(),
        "mse_total": float(np.mean(mse)),
        "rmse_per_joint": np.sqrt(mse).tolist(),
        "rmse_total": float(np.sqrt(np.mean(diff**2))),
        "mae_per_joint": mae.tolist(),
        "mae_total": float(np.mean(mae)),
        "var_per_joint": variance.tolist(),
        "nmse_per_joint": nmse.tolist(),
        "nmse_total": float(np.mean(nmse)),
        "joints_under_gate": int(np.sum(nmse < NMSE_GATE_THRESHOLD)),
        "pred_min_per_joint": np.min(pred, axis=0).tolist(),
        "pred_max_per_joint": np.max(pred, axis=0).tolist(),
        "gt_min_per_joint": np.min(gt, axis=0).tolist(),
        "gt_max_per_joint": np.max(gt, axis=0).tolist(),
    }


def _plot_episode(
    episode_id: int,
    chunks: list[tuple[int, np.ndarray]],
    gt: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(ACTION_DIM, 1, figsize=(10, 2.2 * ACTION_DIM), sharex=True)
    for joint, axis in enumerate(axes):
        axis.plot(np.arange(len(gt)), gt[:, joint], color="tab:blue", linewidth=1.5, label="ground truth")
        for index, (start, prediction) in enumerate(chunks):
            axis.plot(
                np.arange(start, start + len(prediction)),
                prediction[:, joint],
                color="tab:orange",
                linewidth=1.2,
                linestyle="--",
                label="vla.cpp GGUF" if index == 0 else None,
            )
        axis.set_ylabel(JOINT_NAMES[joint], fontsize=9)
        axis.grid(alpha=0.3)
        if joint == 0:
            axis.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(f"episode {episode_id}: vla.cpp GGUF vs ground truth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _evaluate_split(
    *,
    split_name: str,
    episodes: list[int],
    data_root: Path,
    client: VlaCppClient,
    stats: TurboVlaAlohaStats,
    instruction: str,
    stride: int,
    image_size: int,
    output_dir: Path,
    write_plots: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    global_pred: list[np.ndarray] = []
    global_gt: list[np.ndarray] = []
    per_episode: dict[str, Any] = {}
    server_latency: list[float] = []
    wall_latency: list[float] = []

    for episode_id in episodes:
        states, actions = _load_episode(data_root, episode_id)
        length = len(actions)
        anchors = list(range(0, length, stride))
        view_frames = []
        for camera_key in CAMERA_KEYS:
            video_path = (
                data_root
                / "videos/chunk-000"
                / camera_key
                / f"episode_{episode_id:06d}.mp4"
            )
            view_frames.append(
                _decode_anchor_frames(video_path, anchors, length, image_size)
            )

        episode_pred: list[np.ndarray] = []
        episode_gt: list[np.ndarray] = []
        plot_chunks: list[tuple[int, np.ndarray]] = []
        for anchor in anchors:
            observation = {
                "observation.images.image": view_frames[0][anchor],
                "observation.images.image2": view_frames[1][anchor],
                "observation.state": states[anchor],
                "task": instruction,
            }
            start = time.perf_counter()
            client._predict_chunk(observation)
            wall_latency.append((time.perf_counter() - start) * 1000.0)
            response = client._last_response
            if response is None:
                raise RuntimeError("server returned no response")
            server_latency.append(float(response.latency_ms_total))
            normalized = np.asarray(response.action_chunk, dtype=np.float32).reshape(
                response.chunk_size, response.action_dim
            )
            if normalized.shape != (CHUNK_SIZE, ACTION_DIM):
                raise ValueError(f"unexpected action chunk shape {normalized.shape}")
            # Exact upstream eval behavior: inverse min/max mapping without clipping.
            prediction = (
                0.5 * (normalized + 1.0) * (stats.action_max - stats.action_min)
                + stats.action_min
            ).astype(np.float32)
            valid = min(CHUNK_SIZE, length - anchor)
            pred_valid = prediction[:valid]
            gt_valid = actions[anchor : anchor + valid]
            episode_pred.append(pred_valid)
            episode_gt.append(gt_valid)
            plot_chunks.append((anchor, pred_valid))

        pred_array = np.concatenate(episode_pred, axis=0)
        gt_array = np.concatenate(episode_gt, axis=0)
        metrics = _compute_metrics(pred_array, gt_array)
        per_episode[str(episode_id)] = metrics
        global_pred.append(pred_array)
        global_gt.append(gt_array)
        episode_payload = {
            "episode": episode_id,
            "split": split_name,
            "num_open_loop_steps": stride,
            "chunk_size": CHUNK_SIZE,
            **metrics,
        }
        (output_dir / f"openloop_{episode_id}.json").write_text(
            json.dumps(episode_payload, indent=2) + "\n", encoding="utf-8"
        )
        if write_plots:
            _plot_episode(
                episode_id,
                plot_chunks,
                actions,
                output_dir / f"openloop_{episode_id}.png",
            )
        print(
            f"[{split_name}] episode={episode_id:03d} points={metrics['n_points']} "
            f"MSE={metrics['mse_total']:.6f} MAE={metrics['mae_total']:.6f} "
            f"NMSE={metrics['nmse_total']:.6f}",
            flush=True,
        )

    overall = _compute_metrics(np.concatenate(global_pred), np.concatenate(global_gt))
    latency = {
        "requests": len(server_latency),
        "server_mean_ms": float(np.mean(server_latency)),
        "server_p50_ms": float(np.percentile(server_latency, 50)),
        "server_p95_ms": float(np.percentile(server_latency, 95)),
        "client_wall_mean_ms": float(np.mean(wall_latency)),
    }
    summary = {
        "split": split_name,
        "episodes": episodes,
        "num_open_loop_steps": stride,
        "chunk_size": CHUNK_SIZE,
        "nmse_gate_threshold": NMSE_GATE_THRESHOLD,
        "overall": overall,
        "latency": latency,
        "per_episode": per_episode,
    }
    (output_dir / "openloop_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _metric_row(split: str, implementation: str, metric: str, values: dict[str, Any]) -> list[Any]:
    return [split, implementation, metric, *values[f"{metric.lower()}_per_joint"], values[f"{metric.lower()}_total"]]


def _write_csv(path: Path, summaries: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "implementation", "metric", *[f"j{i}" for i in range(7)], "total"])
        for split, summary in summaries.items():
            for metric in ("NMSE", "MSE", "MAE"):
                writer.writerow(_metric_row(split, "vla.cpp GGUF", metric, summary["overall"]))
                writer.writerow(_metric_row(split, "PyTorch reference", metric, PYTORCH_REFERENCE[split]))


def _format_table_row(split: str, implementation: str, metric: str, values: dict[str, Any]) -> str:
    key = metric.lower()
    numbers = [*values[f"{key}_per_joint"], values[f"{key}_total"]]
    return "| " + " | ".join([split, implementation, metric, *[f"{value:.4f}" for value in numbers]]) + " |"


def _comparison_summary(summaries: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for split in ("val", "train_sample"):
        comparison[split] = {}
        for metric in ("nmse", "mse", "mae"):
            cpp = float(summaries[split]["overall"][f"{metric}_total"])
            reference = float(PYTORCH_REFERENCE[split][f"{metric}_total"])
            comparison[split][metric] = {
                "vla_cpp": cpp,
                "pytorch_reference": reference,
                "absolute_delta": cpp - reference,
                "relative_delta_percent": 100.0 * (cpp - reference) / reference,
            }
    return comparison


def _write_results_markdown(
    path: Path,
    summaries: dict[str, Any],
    *,
    gguf: Path,
    data_root: Path,
    stride: int,
    include_plots: bool = True,
) -> None:
    lines = [
        "# TurboVLA ALOHA open-loop — vla.cpp GGUF",
        "",
        "This evaluation mirrors the upstream TurboVLA open-loop protocol: real recorded",
        "camera frames and proprioception, fixed 12-step action chunks, raw joint units,",
        "and non-overlapping anchors. No simulator is used.",
        "",
        f"- GGUF: `{gguf}`",
        f"- Dataset: `{data_root}`",
        f"- Anchor stride: `{stride}`; action chunk: `{CHUNK_SIZE}`",
        f"- PyTorch reference: [{PYTORCH_REFERENCE_URL}]({PYTORCH_REFERENCE_URL})",
        "",
        "## Open-loop comparison",
        "",
        "| Split | Implementation | Metric | j0 | j1 | j2 | j3 | j4 | j5 | j6 (gripper) | **total** |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("val", "train_sample"):
        for metric in ("NMSE", "MSE", "MAE"):
            lines.append(_format_table_row(split, "vla.cpp GGUF", metric, summaries[split]["overall"]))
            lines.append(_format_table_row(split, "PyTorch reference", metric, PYTORCH_REFERENCE[split]))
    comparison = _comparison_summary(summaries)
    lines.extend(["", "## Aggregate delta from PyTorch", ""])
    for split in ("val", "train_sample"):
        fields = comparison[split]
        lines.append(
            f"- **{split}**: NMSE {fields['nmse']['relative_delta_percent']:+.3f}%, "
            f"MSE {fields['mse']['relative_delta_percent']:+.3f}%, "
            f"MAE {fields['mae']['relative_delta_percent']:+.3f}%."
        )
    lines.extend(["", "## Runtime", ""])
    for split in ("val", "train_sample"):
        overall = summaries[split]["overall"]
        latency = summaries[split]["latency"]
        lines.append(
            f"- **{split}**: {len(summaries[split]['episodes'])} episodes, "
            f"{overall['n_points']} chunk points, {latency['requests']} requests, "
            f"server mean/p50/p95 = {latency['server_mean_ms']:.2f}/"
            f"{latency['server_p50_ms']:.2f}/{latency['server_p95_ms']:.2f} ms."
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"The upstream gate is total NMSE < {NMSE_GATE_THRESHOLD}. "
            f"Val: **{'PASS' if summaries['val']['overall']['nmse_total'] < NMSE_GATE_THRESHOLD else 'FAIL'}**; "
            f"train sample: **{'PASS' if summaries['train_sample']['overall']['nmse_total'] < NMSE_GATE_THRESHOLD else 'FAIL'}**.",
            "",
        ]
    )
    if include_plots:
        lines.extend(
            [
                "## Representative single-episode plots",
                "",
                "Each plot shows all 7 joints, predicted versus recorded, for one episode.",
                "The full 20+20 per-episode JSON/PNG pairs are stored in `val/` and",
                "`train_sample/` next to this report.",
                "",
                "![val episode 7](val/openloop_7.png)",
                "",
                "*Val episode 7 — GGUF prediction versus recorded joint trajectories, held-out.*",
                "",
                "![train episode 0](train_sample/openloop_0.png)",
                "",
                "*Train-sample episode 0 — GGUF prediction versus recorded, training data.*",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-addr", default="tcp://127.0.0.1:5557")
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=CHUNK_SIZE)
    parser.add_argument("--splits", default="val,train_sample")
    parser.add_argument("--episodes", default=None, help="Custom episode list; requires --splits custom")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/turbovla_aloha_openloop"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be positive")
    data_root = _discover_data_root(args.data_root)
    tasks_path = data_root / "meta/tasks.jsonl"
    instruction = json.loads(tasks_path.read_text(encoding="utf-8").splitlines()[0])["task"]
    stats = TurboVlaAlohaStats.from_json(args.stats_json)
    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    split_episodes: dict[str, list[int]] = {}
    for split in split_names:
        if split == "val":
            split_episodes[split] = list(VAL_EPISODES)
        elif split == "train_sample":
            split_episodes[split] = list(TRAIN_SAMPLE_EPISODES)
        elif split == "custom" and args.episodes:
            split_episodes[split] = _parse_episode_list(args.episodes)
        else:
            parser.error(f"unsupported split {split!r}; use val, train_sample, or custom with --episodes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = VlaCppClient(
        vla_addr=args.vla_addr,
        arch="turbovla_aloha",
        tokenizer_name=args.tokenizer,
        image_size=256,
        max_state_dim=7,
        real_action_dim=7,
        max_length=256,
        recv_timeout_ms=args.recv_timeout_ms,
        n_action_steps=CHUNK_SIZE,
        stats_json=args.stats_json,
    )
    summaries: dict[str, Any] = {}
    try:
        for split, episodes in split_episodes.items():
            summaries[split] = _evaluate_split(
                split_name=split,
                episodes=episodes,
                data_root=data_root,
                client=client,
                stats=stats,
                instruction=instruction,
                stride=args.stride,
                image_size=256,
                output_dir=args.output_dir / split,
                write_plots=not args.no_plots,
            )
    finally:
        client.close()

    combined = {
        "schema": "vla.cpp.turbovla_aloha_openloop.v1",
        "protocol": "DuyBaoDOCer/TurboVLA experiments/aloha/eval_openloop.py",
        "reference_url": PYTORCH_REFERENCE_URL,
        "gguf": str(args.gguf.resolve()),
        "stats_json": str(args.stats_json.resolve()),
        "data_root": str(data_root),
        "instruction": instruction,
        "pytorch_reference": PYTORCH_REFERENCE,
        "comparison": (
            _comparison_summary(summaries)
            if set(summaries) == {"val", "train_sample"}
            else None
        ),
        "splits": summaries,
    }
    (args.output_dir / "openloop_summary.json").write_text(
        json.dumps(combined, indent=2) + "\n", encoding="utf-8"
    )
    if set(summaries) == {"val", "train_sample"}:
        _write_csv(args.output_dir / "openloop_metrics.csv", summaries)
        _write_results_markdown(
            args.output_dir / "RESULTS.md",
            summaries,
            gguf=args.gguf.resolve(),
            data_root=data_root,
            stride=args.stride,
            include_plots=not args.no_plots,
        )
    print(json.dumps({name: value["overall"] for name, value in summaries.items()}, indent=2))
    print(f"wrote {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
