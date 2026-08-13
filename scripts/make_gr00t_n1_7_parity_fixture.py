#!/usr/bin/env python3
"""Create a deterministic GR00T-N1.7 GGUF parity fixture from a LeRobot episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embodiment", default="new_embodiment")
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    embodiment = EmbodimentTag.resolve(args.embodiment)
    policy = Gr00tPolicy(
        embodiment_tag=embodiment,
        model_path=str(args.checkpoint.resolve()),
        device=args.device,
    )
    modality = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(dataset_path=str(args.dataset.resolve()), modality_configs=modality)
    trajectory = loader[args.trajectory]

    modality_without_action = dict(modality)
    modality_without_action.pop("action")
    point = extract_step_data(trajectory, args.step, modality_without_action, embodiment)
    observation = {}
    for key, value in point.states.items():
        observation[f"state.{key}"] = value
    for key, value in point.images.items():
        observation[f"video.{key}"] = np.asarray(value)
    for language_key in modality["language"].modality_keys:
        observation[language_key] = point.text
    parsed = parse_observation_gr00t(observation, modality)

    unbatched = policy._unbatch_observation(parsed)
    processed = []
    for item in unbatched:
        step_data = policy._to_vla_step_data(item)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": step_data}]
        processed.append(policy.processor(messages))
    collated = _rec_to_dtype(policy.collate_fn(processed), dtype=torch.bfloat16)

    inputs = collated["inputs"]
    action_horizon = int(policy.model.config.action_horizon)
    action_dim = int(policy.model.config.max_action_dim)
    generator = np.random.default_rng(args.seed)
    noise = generator.standard_normal((1, action_horizon, action_dim), dtype=np.float32)
    fixed_noise = torch.from_numpy(noise).to(policy.model.device, dtype=policy.model.dtype)
    noise = fixed_noise.float().cpu().numpy()

    original_randn = torch.randn

    def deterministic_randn(*positional, **kwargs):
        size = kwargs.get("size", positional[0] if positional else None)
        if size is not None and tuple(size) == tuple(fixed_noise.shape):
            return fixed_noise.clone()
        return original_randn(*positional, **kwargs)

    torch.randn = deterministic_randn
    try:
        with torch.inference_mode():
            prediction = policy.model.get_action(**collated)["action_pred"].float().cpu().numpy()
    finally:
        torch.randn = original_randn

    args.output.mkdir(parents=True, exist_ok=True)
    tokens = inputs["input_ids"][0].cpu().numpy().astype(np.int32)
    state = inputs["state"][0].float().cpu().numpy().reshape(-1).astype(np.float32)
    Path(args.output / "tokens.txt").write_text(",".join(map(str, tokens.tolist())) + "\n")
    Path(args.output / "state.txt").write_text(",".join(f"{x:.9g}" for x in state) + "\n")
    noise[0].astype("<f4").tofile(args.output / "noise.f32")
    prediction[0].astype("<f4").tofile(args.output / "pytorch_action.f32")

    images = processed[0]["vlm_content"]["images"]
    image_processor = policy.collate_fn.processor.image_processor
    image_paths = []
    resized_images = []
    for index, image in enumerate(images):
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
        resized_images.append(resized)
        array = resized.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        path = args.output / f"image_{index}.png"
        Image.fromarray(array, mode="RGB").save(path)
        image_paths.append(str(path))

    replay = policy.collate_fn.processor(
        text=[processed[0]["vlm_content"]["text"]],
        images=resized_images,
        return_tensors="pt",
        padding=True,
    )
    pixel_diff = (replay["pixel_values"] - inputs["pixel_values"].cpu()).abs()
    if not torch.equal(replay["input_ids"], inputs["input_ids"].cpu()):
        raise RuntimeError("smart-resized fixture changed input_ids")

    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset": str(args.dataset.resolve()),
        "trajectory": args.trajectory,
        "step": args.step,
        "seed": args.seed,
        "embodiment": embodiment.value,
        "embodiment_id": int(inputs["embodiment_id"][0]),
        "instruction": point.text,
        "token_count": int(tokens.size),
        "state_count": int(state.size),
        "noise_shape": list(noise.shape[1:]),
        "prediction_shape": list(prediction.shape[1:]),
        "images": image_paths,
        "image_grid_thw": inputs["image_grid_thw"].cpu().tolist(),
        "replayed_pixel_values_max_abs_diff": float(pixel_diff.max()),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
