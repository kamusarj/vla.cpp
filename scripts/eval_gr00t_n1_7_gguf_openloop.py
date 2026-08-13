#!/usr/bin/env python3
"""Prepare and score a recorded-dataset open-loop evaluation for GR00T-N1.7 GGUF.

The ``prepare`` command runs the checkpoint's Hugging Face processor but does
not load the PyTorch model.  It writes processor-exact tensors and resized RGB
frames for the persistent ``vla-openloop`` C++ runner.  The ``score`` command
decodes the runner's normalized action output, reproduces NVIDIA's trajectory
MSE/MAE calculation, and writes the trajectory plot.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.policy.gr00t_policy import _rec_to_dtype


def _processor_and_data(checkpoint: Path, dataset: Path, embodiment_name: str):
    embodiment = EmbodimentTag.resolve(embodiment_name)
    processor = Gr00tN1d7Processor.from_pretrained(str(checkpoint.resolve()))
    processor.eval()
    modality = {
        key: value
        for key, value in processor.modality_configs[embodiment.value].items()
        if key != "rl_info"
    }
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset.resolve()), modality_configs=modality
    )
    return processor, modality, loader, embodiment


def _extract_columns(trajectory, columns: list[str]) -> np.ndarray:
    arrays = {
        column: np.vstack([np.asarray(value) for value in trajectory[column]])
        for column in columns
    }
    return np.concatenate([arrays[column] for column in columns], axis=-1)


def _write_array(path: Path, array: np.ndarray, dtype: str) -> None:
    np.asarray(array).astype(dtype, copy=False).tofile(path)


def prepare(args: argparse.Namespace) -> int:
    processor, modality, loader, embodiment = _processor_and_data(
        args.checkpoint, args.dataset, args.embodiment
    )
    trajectory = loader[args.trajectory]
    actual_steps = min(args.steps, len(trajectory))
    request_steps = list(range(0, actual_steps, args.execution_horizon))
    modality_without_action = deepcopy(modality)
    modality_without_action.pop("action")

    all_tokens: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    all_images: list[np.ndarray] = []
    instructions: list[str] = []
    image_grids: list[list[list[int]]] = []
    expected_image_shape: tuple[int, int, int, int] | None = None
    expected_token_count: int | None = None

    image_processor = processor.collator.processor.image_processor
    for request_index, step in enumerate(request_steps):
        point = extract_step_data(
            trajectory, step, modality_without_action, embodiment
        )
        transformed = processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": point}]
        )
        collated = _rec_to_dtype(
            processor.collator([transformed]), dtype=torch.bfloat16
        )["inputs"]

        tokens = collated["input_ids"][0].cpu().numpy().astype(np.int32)
        state = (
            collated["state"][0]
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
            .astype(np.float32)
        )
        if expected_token_count is None:
            expected_token_count = int(tokens.size)
        elif tokens.size != expected_token_count:
            raise RuntimeError(
                f"request {request_index} has {tokens.size} tokens; expected "
                f"{expected_token_count}. The C++ batch fixture requires a fixed token count."
            )

        resized_views = []
        for image in transformed["vlm_content"]["images"]:
            height, width = image.shape[-2:]
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_processor.patch_size * image_processor.merge_size,
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
            )
            resized = tv_functional.resize(
                image,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            resized_views.append(
                resized.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            )
        images = np.stack(resized_views)
        if expected_image_shape is None:
            expected_image_shape = tuple(images.shape)
        elif tuple(images.shape) != expected_image_shape:
            raise RuntimeError(
                f"request {request_index} image shape {images.shape}; expected "
                f"{expected_image_shape}"
            )

        all_tokens.append(tokens)
        all_states.append(state)
        all_images.append(images)
        instructions.append(point.text)
        image_grids.append(collated["image_grid_thw"].cpu().tolist())
        print(
            f"prepared request {request_index + 1}/{len(request_steps)} "
            f"(trajectory step {step})",
            flush=True,
        )

    tokens_array = np.stack(all_tokens)
    states_array = np.stack(all_states)
    images_array = np.stack(all_images)
    action_horizon = int(processor.max_action_horizon)
    action_dim = int(processor.max_action_dim)
    generator = np.random.default_rng(args.seed)
    noise = generator.standard_normal(
        (len(request_steps), action_horizon, action_dim), dtype=np.float32
    )
    # GR00T-N1.7 samples BF16 noise; retain those exact values for GGUF input.
    noise = torch.from_numpy(noise).to(torch.bfloat16).float().numpy()

    state_keys = list(modality["state"].modality_keys)
    action_keys = list(modality["action"].modality_keys)
    state_trajectory = _extract_columns(
        trajectory, [f"state.{key}" for key in state_keys]
    )[:actual_steps]
    ground_truth = _extract_columns(
        trajectory, [f"action.{key}" for key in action_keys]
    )[:actual_steps]

    args.fixture.mkdir(parents=True, exist_ok=True)
    _, view_count, image_height, image_width, channels = images_array.shape
    if channels != 3:
        raise RuntimeError(f"expected RGB images, found {channels} channels")
    meta_values = (
        len(request_steps),
        tokens_array.shape[1],
        view_count,
        image_height,
        image_width,
        states_array.shape[1],
        action_horizon,
        action_dim,
    )
    (args.fixture / "meta.txt").write_text(
        " ".join(str(value) for value in meta_values) + "\n"
    )
    _write_array(args.fixture / "tokens.i32", tokens_array, "<i4")
    _write_array(args.fixture / "states.f32", states_array, "<f4")
    _write_array(args.fixture / "noise.f32", noise, "<f4")
    _write_array(args.fixture / "images.u8", images_array, "u1")
    _write_array(args.fixture / "ground_truth.f32", ground_truth, "<f4")
    _write_array(args.fixture / "state_trajectory.f32", state_trajectory, "<f4")

    metadata = {
        "format": "vla.cpp-gr00t-n1.7-openloop-v1",
        "checkpoint_processor": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "trajectory": args.trajectory,
        "trajectory_length": len(trajectory),
        "actual_steps": actual_steps,
        "execution_horizon": args.execution_horizon,
        "request_steps": request_steps,
        "request_count": len(request_steps),
        "seed": args.seed,
        "embodiment": embodiment.value,
        "embodiment_id": int(processor.embodiment_id_mapping[embodiment.value]),
        "instructions": instructions,
        "state_keys": state_keys,
        "action_keys": action_keys,
        "active_state_dim": int(state_trajectory.shape[1]),
        "active_action_dim": int(ground_truth.shape[1]),
        "model_state_dim": int(states_array.shape[1]),
        "model_action_horizon": action_horizon,
        "model_action_dim": action_dim,
        "token_count": int(tokens_array.shape[1]),
        "image_shape_vhwc": [view_count, image_height, image_width, channels],
        "image_grid_thw": image_grids,
    }
    (args.fixture / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2))
    return 0


def _read_timings(path: Path) -> dict[str, float | int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"request_count": 0}
    result: dict[str, float | int] = {"request_count": len(rows)}
    for column in ("total_ms", "vision_ms", "inference_ms"):
        values = np.asarray([float(row[column]) for row in rows])
        result[f"mean_{column}"] = float(values.mean())
        result[f"median_{column}"] = float(np.median(values))
        result[f"min_{column}"] = float(values.min())
        result[f"max_{column}"] = float(values.max())
    return result


def _plot(
    state: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    metadata: dict,
    output_path: Path,
) -> None:
    action_dim = ground_truth.shape[1]
    figure, axes = plt.subplots(
        nrows=action_dim, ncols=1, figsize=(8, 4 * action_dim)
    )
    if action_dim == 1:
        axes = [axes]
    figure.suptitle(
        f"Trajectory {metadata['trajectory']} - State: "
        f"{', '.join(metadata['state_keys'])} | Action: "
        f"{', '.join(metadata['action_keys'])}",
        fontsize=16,
        color="blue",
    )
    for action_index, axis in enumerate(axes):
        if state.shape == ground_truth.shape:
            axis.plot(state[:, action_index], label="state joints")
        axis.plot(ground_truth[:, action_index], label="gt action")
        axis.plot(prediction[:, action_index], label="pred action")
        points = range(0, len(ground_truth), metadata["execution_horizon"])
        for point_index, point in enumerate(points):
            axis.plot(
                point,
                ground_truth[point, action_index],
                "ro",
                label="inference point" if point_index == 0 else None,
            )
        axis.set_title(f"Action {action_index}")
        axis.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def score(args: argparse.Namespace) -> int:
    metadata = json.loads((args.fixture / "metadata.json").read_text())
    processor, _, _, embodiment = _processor_and_data(
        args.checkpoint, args.dataset, metadata["embodiment"]
    )
    normalized_shape = (
        metadata["request_count"],
        metadata["model_action_horizon"],
        metadata["model_action_dim"],
    )
    normalized = np.fromfile(args.actions, dtype="<f4")
    expected_values = int(np.prod(normalized_shape))
    if normalized.size != expected_values:
        raise RuntimeError(
            f"{args.actions} has {normalized.size} floats; expected {expected_values}"
        )
    normalized = normalized.reshape(normalized_shape)
    if not np.isfinite(normalized).all():
        raise RuntimeError("GGUF output contains non-finite values")

    decoded = processor.decode_action(normalized, embodiment, state=None)
    action_keys = metadata["action_keys"]
    chunks = np.concatenate([np.asarray(decoded[key]) for key in action_keys], axis=-1)
    prediction = chunks.reshape(-1, chunks.shape[-1])[: metadata["actual_steps"]]
    ground_truth = np.fromfile(args.fixture / "ground_truth.f32", dtype="<f4").reshape(
        metadata["actual_steps"], metadata["active_action_dim"]
    )
    state = np.fromfile(args.fixture / "state_trajectory.f32", dtype="<f4").reshape(
        metadata["actual_steps"], metadata["active_state_dim"]
    )
    if prediction.shape != ground_truth.shape:
        raise RuntimeError(
            f"prediction shape {prediction.shape} != ground truth shape {ground_truth.shape}"
        )

    mse = float(np.mean((ground_truth - prediction) ** 2))
    mae = float(np.mean(np.abs(ground_truth - prediction)))
    per_dimension_mse = np.mean((ground_truth - prediction) ** 2, axis=0)
    per_dimension_mae = np.mean(np.abs(ground_truth - prediction), axis=0)
    timings = _read_timings(args.fixture / "timings.csv")

    args.output.mkdir(parents=True, exist_ok=True)
    _write_array(args.output / "pred_actions.f32", prediction, "<f4")
    _plot(state, ground_truth, prediction, metadata, args.output / "traj_0.jpeg")
    metrics = {
        "evaluator": "vla.cpp GGUF recorded-dataset open-loop evaluation",
        "checkpoint_gguf": str(args.gguf.resolve()) if args.gguf else None,
        "checkpoint_processor": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "trajectory": metadata["trajectory"],
        "actual_steps": metadata["actual_steps"],
        "execution_horizon": metadata["execution_horizon"],
        "request_count": metadata["request_count"],
        "seed": metadata["seed"],
        "unnormalized_action_mse": mse,
        "unnormalized_action_mae": mae,
        "per_dimension_mse": per_dimension_mse.tolist(),
        "per_dimension_mae": per_dimension_mae.tolist(),
        "normalized_output_min": float(normalized.min()),
        "normalized_output_max": float(normalized.max()),
        "normalized_output_mean": float(normalized.mean()),
        "timing": timings,
        "plot": str((args.output / "traj_0.jpeg").resolve()),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output / "eval.log").write_text(
        f"Unnormalized Action MSE across single traj: {mse}\n"
        f"Unnormalized Action MAE across single traj: {mae}\n"
        f"state_joints vs time {state.shape}\n"
        f"gt_action_joints vs time {ground_truth.shape}\n"
        f"pred_action_joints vs time {prediction.shape}\n"
    )
    print(json.dumps(metrics, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--checkpoint", type=Path, required=True)
    prepare_parser.add_argument("--dataset", type=Path, required=True)
    prepare_parser.add_argument("--fixture", type=Path, required=True)
    prepare_parser.add_argument("--embodiment", default="new_embodiment")
    prepare_parser.add_argument("--trajectory", type=int, default=0)
    prepare_parser.add_argument("--steps", type=int, default=814)
    prepare_parser.add_argument("--execution-horizon", type=int, default=16)
    prepare_parser.add_argument("--seed", type=int, default=20260813)
    prepare_parser.set_defaults(func=prepare)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--checkpoint", type=Path, required=True)
    score_parser.add_argument("--dataset", type=Path, required=True)
    score_parser.add_argument("--fixture", type=Path, required=True)
    score_parser.add_argument("--actions", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--gguf", type=Path)
    score_parser.set_defaults(func=score)
    args = parser.parse_args()
    if getattr(args, "execution_horizon", 1) <= 0:
        parser.error("--execution-horizon must be positive")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
