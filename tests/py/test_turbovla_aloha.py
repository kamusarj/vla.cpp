# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pathlib
import sys

import numpy as np
import pytest


CLIENT_DIR = pathlib.Path(__file__).resolve().parents[2] / "eval" / "client"
sys.path.insert(0, str(CLIENT_DIR))

from turbovla_aloha import TurboVlaAlohaStats  # noqa: E402


def _payload():
    return {
        "state_mean": [1.0] * 7,
        "state_std": [2.0] * 7,
        "action_min": [-2.0] * 7,
        "action_max": [4.0] * 7,
        "stats_key": "aloha_test",
    }


def test_turbovla_aloha_normalization_round_trip_contract():
    stats = TurboVlaAlohaStats.from_mapping(_payload())
    np.testing.assert_allclose(stats.normalize_state(np.ones(7)), np.zeros(7))

    normalized = np.array([[-1.0] * 7, [0.0] * 7, [2.0] * 7], dtype=np.float32)
    raw = stats.unnormalize_action(normalized)
    np.testing.assert_allclose(raw[0], [-2.0] * 7)
    np.testing.assert_allclose(raw[1], [1.0] * 7)
    np.testing.assert_allclose(raw[2], [4.0] * 7)


def test_turbovla_aloha_rejects_invalid_stats_and_state():
    payload = _payload()
    payload["state_std"][3] = 0.0
    with pytest.raises(ValueError, match="state_std"):
        TurboVlaAlohaStats.from_mapping(payload)

    stats = TurboVlaAlohaStats.from_mapping(_payload())
    with pytest.raises(ValueError, match="7-D"):
        stats.normalize_state(np.zeros(8, dtype=np.float32))
