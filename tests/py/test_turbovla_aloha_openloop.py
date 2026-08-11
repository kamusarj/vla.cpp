# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest


pytest.importorskip("av")
pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

EVAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "eval"
sys.path.insert(0, str(EVAL_DIR))

from client.eval_turbovla_aloha_openloop import CHUNK_SIZE  # noqa: E402
from client.eval_turbovla_aloha_openloop import TRAIN_SAMPLE_EPISODES  # noqa: E402
from client.eval_turbovla_aloha_openloop import VAL_EPISODES  # noqa: E402
from client.eval_turbovla_aloha_openloop import _compute_metrics  # noqa: E402


def test_published_split_and_chunk_contract():
    assert CHUNK_SIZE == 12
    assert len(VAL_EPISODES) == 20
    assert len(TRAIN_SAMPLE_EPISODES) == 20
    assert set(VAL_EPISODES).isdisjoint(TRAIN_SAMPLE_EPISODES)


def test_openloop_metrics_match_upstream_definition():
    gt = np.asarray([[0.0] * 7, [2.0] * 7], dtype=np.float32)
    pred = gt + 0.5
    metrics = _compute_metrics(pred, gt)
    np.testing.assert_allclose(metrics["mse_per_joint"], [0.25] * 7)
    np.testing.assert_allclose(metrics["mae_per_joint"], [0.5] * 7)
    np.testing.assert_allclose(metrics["nmse_per_joint"], [0.25] * 7)
    assert metrics["mse_total"] == pytest.approx(0.25)
    assert metrics["rmse_total"] == pytest.approx(0.5)
