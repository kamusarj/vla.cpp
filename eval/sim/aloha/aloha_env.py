#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TurboVLA adapter for the official google-deepmind ALOHA simulator.

The model checkpoint consumes the left-arm dataset contract: two RGB images,
seven joint positions, and a 7-D absolute joint action.  The upstream simulator
is bimanual and accepts 14-D controls, so this adapter applies the same mapping,
camera setup, reset layouts, and three-control-tick stepping used by the ALOHA
simulation in octo-pytorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .official_carrot import AlohaCarrotLayout
from .official_carrot import CONTROL_TIMESTEP
from .official_carrot import DATASET_GRIPPER_CLOSE
from .official_carrot import DATASET_GRIPPER_OPEN
from .official_carrot import DATASET_STEPS_PER_CONTROL
from .official_carrot import create_official_carrot_env
from .official_carrot import get_aloha_carrot_layout
from .official_carrot import jitter_aloha_carrot_layout
from .official_carrot import left_dataset_action_to_bimanual
from .official_carrot import official_gripper_to_dataset


FPS = 1.0 / (CONTROL_TIMESTEP * DATASET_STEPS_PER_CONTROL)
TASK_INSTRUCTION = "Grasp the carrot from the plate, hold it, place it into the cup."


def _resize_uint8(image: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(image, dtype=np.uint8),
        (int(image_size), int(image_size)),
        interpolation=cv2.INTER_AREA,
    ).astype(np.uint8)


def _json_metrics(metrics: dict[str, float | bool]) -> dict[str, float | bool]:
    return {
        key: bool(value) if isinstance(value, (bool, np.bool_)) else float(value)
        for key, value in metrics.items()
    }


class AlohaCarrotCupEnv:
    """7-D left-arm view of the official 14-D ALOHA composer environment."""

    metadata = {
        "render_fps": FPS,
        "simulator": "google-deepmind/aloha_sim",
    }

    def __init__(
        self,
        *,
        action_min: np.ndarray,
        action_max: np.ndarray,
        seed: int = 42,
        image_size: int = 256,
        max_steps: int = 160,
        initial_condition: str = "episode0",
        object_xy_jitter: float = 0.0,
        act_assets: str | Path | None = None,
        randomize: bool | None = None,
    ) -> None:
        # Kept only so older launch commands fail neither parsing nor startup.
        # Official aloha_sim owns its robot models and does not use ACT assets.
        del act_assets
        if randomize is not None:
            object_xy_jitter = 0.012 if randomize else 0.0
        self.action_min = np.asarray(action_min, dtype=np.float64).reshape(7)
        self.action_max = np.asarray(action_max, dtype=np.float64).reshape(7)
        if not np.all(np.isfinite(self.action_min)) or not np.all(
            np.isfinite(self.action_max)
        ):
            raise ValueError("action bounds must be finite")
        if not np.all(self.action_max > self.action_min):
            raise ValueError("action_max must exceed action_min")
        if object_xy_jitter < 0.0 or not np.isfinite(object_xy_jitter):
            raise ValueError("object_xy_jitter must be finite and non-negative")

        self.seed = int(seed)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.initial_condition = str(initial_condition)
        self.object_xy_jitter = float(object_xy_jitter)
        self._env = create_official_carrot_env(
            seed=self.seed,
            max_dataset_steps=self.max_steps,
        )
        spec = self._env.action_spec()
        self._clip_low = np.concatenate(
            [np.asarray(spec.minimum[:6]), [DATASET_GRIPPER_CLOSE]]
        )
        self._clip_high = np.concatenate(
            [np.asarray(spec.maximum[:6]), [DATASET_GRIPPER_OPEN]]
        )
        self._timestep = None
        self.step_count = 0
        self.max_reward = 0.0
        self.success = False
        self.layout: AlohaCarrotLayout | None = None

    @property
    def model(self):
        """Expose the MuJoCo model for the optional passive viewer."""

        return self._env.physics.model.ptr

    @property
    def data(self):
        """Expose the MuJoCo data for the optional passive viewer."""

        return self._env.physics.data.ptr

    def _raw_images(self) -> dict[str, np.ndarray]:
        if self._timestep is None:
            raise RuntimeError("environment must be reset before observation")
        return {
            "overhead_cam": np.asarray(
                self._timestep.observation["overhead_cam"], dtype=np.uint8
            ).copy(),
            "wrist_cam_left": np.asarray(
                self._timestep.observation["wrist_cam_left"], dtype=np.uint8
            ).copy(),
        }

    def state(self) -> np.ndarray:
        if self._timestep is None:
            raise RuntimeError("environment must be reset before reading state")
        joints = np.asarray(
            self._timestep.observation["joints_pos"][:7], dtype=np.float64
        ).copy()
        joints[6] = float(official_gripper_to_dataset(joints[6]))
        return joints.astype(np.float32)

    def observation(self) -> dict[str, Any]:
        raw = self._raw_images()
        high = _resize_uint8(raw["overhead_cam"], self.image_size)
        wrist = _resize_uint8(raw["wrist_cam_left"], self.image_size)
        return {
            "observation.images.image": np.ascontiguousarray(
                np.transpose(high, (2, 0, 1)), dtype=np.float32
            )
            / 255.0,
            "observation.images.image2": np.ascontiguousarray(
                np.transpose(wrist, (2, 0, 1)), dtype=np.float32
            )
            / 255.0,
            "observation.state": self.state(),
            "task": TASK_INSTRUCTION,
            "_render_high_u8": raw["overhead_cam"],
            "_render_wrist_u8": raw["wrist_cam_left"],
        }

    def _task_info(self) -> dict[str, Any]:
        task = self._env.task
        physics = self._env.physics
        carrot_body = task._pen_prop.mjcf_model.find_all("body")[0]
        cup_body = task._mug_prop.mjcf_model.find_all("body")[0]
        metrics = task.release_metrics(physics)
        carrot_position = np.asarray(physics.bind(carrot_body).xpos).copy()
        cup_position = np.asarray(physics.bind(cup_body).xpos).copy()
        inside = bool(metrics["inside_cup"])
        if self.success:
            phase = "released_in_cup"
        elif inside:
            phase = "in_cup_not_released"
        else:
            phase = "running"
        return {
            "reward": float(self.max_reward),
            "max_reward": float(self.max_reward),
            "is_success": bool(self.success),
            "object_phase": phase,
            "inside_cup": inside,
            "carrot_position": carrot_position.tolist(),
            "cup_position": cup_position.tolist(),
            "release_metrics": _json_metrics(metrics),
            "simulator": "google-deepmind/aloha_sim",
            "layout_id": self.layout.layout_id if self.layout is not None else "",
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        initial_condition: str | None = None,
        object_xy_jitter: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        reset_seed = self.seed if seed is None else int(seed)
        layout_id = initial_condition or self.initial_condition
        jitter = self.object_xy_jitter if object_xy_jitter is None else float(
            object_xy_jitter
        )
        layout = get_aloha_carrot_layout(layout_id)
        layout = jitter_aloha_carrot_layout(
            layout,
            xy_jitter=jitter,
            seed=reset_seed,
        )
        if jitter > 0.0:
            self._env.task.set_custom_layout(layout)
        else:
            self._env.task.set_layout(layout.layout_id)
        self.layout = layout
        self._timestep = self._env.reset()
        self.step_count = 0
        self.max_reward = 0.0
        self.success = False
        info = self._task_info()
        info.update(
            {
                "initial_condition": layout.layout_id,
                "layout_description": layout.description,
                "object_positions": {
                    "carrot_xy": list(layout.carrot_xy),
                    "cup_xy": list(layout.cup_xy),
                },
                "object_xy_jitter": jitter,
                "jitter_seed": reset_seed,
            }
        )
        return self.observation(), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        target = np.asarray(action, dtype=np.float64).reshape(-1)
        if target.size != 7 or not np.all(np.isfinite(target)):
            raise ValueError(f"ALOHA action must be finite [7], got {target}")
        clipped = np.clip(target, self._clip_low, self._clip_high)
        clipped = np.clip(clipped, self.action_min, self.action_max)
        official_action = left_dataset_action_to_bimanual(clipped)
        reward = 0.0
        video_frames: list[np.ndarray] = []
        for _ in range(DATASET_STEPS_PER_CONTROL):
            self._timestep = self._env.step(official_action)
            reward = max(reward, float(self._timestep.reward or 0.0))
            video_frames.append(self.render())
            if self._timestep.last():
                break

        self.step_count += 1
        self.max_reward = max(self.max_reward, reward)
        self.success = self.success or reward >= 1.0
        terminated = bool(self.success)
        truncated = bool(self.step_count >= self.max_steps and not terminated)
        info = self._task_info()
        info["video_frames"] = video_frames
        return self.observation(), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Return the Octo-style vertical overhead + left-wrist frame."""

        raw = self._raw_images()
        return np.concatenate(
            [raw["overhead_cam"], raw["wrist_cam_left"]], axis=0
        )

    def close(self) -> None:
        self._env.close()


__all__ = ["AlohaCarrotCupEnv", "FPS", "TASK_INSTRUCTION"]
