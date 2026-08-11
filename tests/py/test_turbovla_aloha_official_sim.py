# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pytest


os.environ.setdefault("MUJOCO_GL", "egl")
pytest.importorskip("aloha_sim")
pytest.importorskip("dm_control")

EVAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from sim.aloha import AlohaCarrotCupEnv  # noqa: E402
from sim.aloha.official_carrot import ALOHA_CARROT_LAYOUTS  # noqa: E402
from sim.aloha.official_carrot import DATASET_GRIPPER_CLOSE  # noqa: E402
from sim.aloha.official_carrot import DATASET_GRIPPER_OPEN  # noqa: E402
from sim.aloha.official_carrot import UPSTREAM_ALOHA_SIM_COMMIT  # noqa: E402
from sim.aloha.official_carrot import dataset_gripper_to_official  # noqa: E402
from sim.aloha.official_carrot import official_gripper_to_dataset  # noqa: E402


def _bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.asarray(
        [-3.0, -2.0, -2.0, -3.0, -2.0, -3.0, DATASET_GRIPPER_CLOSE]
    )
    high = np.asarray(
        [3.0, 2.0, 2.0, 3.0, 2.0, 3.0, DATASET_GRIPPER_OPEN]
    )
    return low, high


def test_official_backend_pin_and_gripper_round_trip():
    assert UPSTREAM_ALOHA_SIM_COMMIT == "d02904607cca1bf6dfb72f30b522506ac7ca0f91"
    assert len(ALOHA_CARROT_LAYOUTS) == 10
    dataset = np.linspace(DATASET_GRIPPER_CLOSE, DATASET_GRIPPER_OPEN, 11)
    np.testing.assert_allclose(
        official_gripper_to_dataset(dataset_gripper_to_official(dataset)),
        dataset,
        atol=1e-7,
    )


def test_turbovla_adapter_reset_and_three_tick_step():
    low, high = _bounds()
    env = AlohaCarrotCupEnv(
        action_min=low,
        action_max=high,
        image_size=64,
        max_steps=2,
        initial_condition="episode0",
    )
    try:
        observation, info = env.reset(seed=42)
        assert observation["observation.images.image"].shape == (3, 64, 64)
        assert observation["observation.images.image2"].shape == (3, 64, 64)
        assert observation["observation.state"].shape == (7,)
        assert info["simulator"] == "google-deepmind/aloha_sim"
        assert info["object_positions"] == {
            "carrot_xy": [0.118, 0.145],
            "cup_xy": [-0.0335, 0.15],
        }

        observation, reward, terminated, truncated, info = env.step(
            observation["observation.state"]
        )
        assert reward == 0.0
        assert not terminated
        assert not truncated
        assert len(info["video_frames"]) == 3
        assert info["video_frames"][0].shape == (960, 640, 3)
    finally:
        env.close()
