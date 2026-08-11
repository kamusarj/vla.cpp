# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""TurboVLA ALOHA state/action transforms shared by mock and ROS2 clients."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _vector(payload: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in payload:
        raise KeyError(f"TurboVLA ALOHA stats are missing {name!r}")
    value = np.asarray(payload[name], dtype=np.float32).reshape(-1)
    if value.size != 7:
        raise ValueError(f"TurboVLA ALOHA stats {name!r} must be 7-D, got {value.size}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"TurboVLA ALOHA stats {name!r} contain NaN/Inf")
    return value


@dataclass(frozen=True)
class TurboVlaAlohaStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_min: np.ndarray
    action_max: np.ndarray
    stats_key: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TurboVlaAlohaStats":
        stats = cls(
            state_mean=_vector(payload, "state_mean"),
            state_std=_vector(payload, "state_std"),
            action_min=_vector(payload, "action_min"),
            action_max=_vector(payload, "action_max"),
            stats_key=str(payload.get("stats_key", "")),
        )
        if np.any(stats.state_std <= 0.0):
            raise ValueError("TurboVLA ALOHA state_std must be strictly positive")
        if np.any(stats.action_max <= stats.action_min):
            raise ValueError("TurboVLA ALOHA action_max must exceed action_min")
        return stats

    @classmethod
    def from_json(cls, path: str | Path) -> "TurboVlaAlohaStats":
        stats_path = Path(path).expanduser()
        if not stats_path.is_file():
            raise FileNotFoundError(f"TurboVLA ALOHA stats JSON not found: {stats_path}")
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"TurboVLA ALOHA stats must be a JSON object: {stats_path}")
        return cls.from_mapping(payload)

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        raw = np.asarray(state, dtype=np.float32).reshape(-1)
        if raw.size != 7:
            raise ValueError(f"TurboVLA ALOHA state must be 7-D, got {raw.size}")
        if not np.all(np.isfinite(raw)):
            raise ValueError("TurboVLA ALOHA state contains NaN/Inf")
        return ((raw - self.state_mean) / (self.state_std + 1e-6)).astype(np.float32)

    def unnormalize_action(self, normalized: np.ndarray) -> np.ndarray:
        action = np.asarray(normalized, dtype=np.float32)
        if action.ndim < 1 or action.shape[-1] != 7:
            raise ValueError(
                f"TurboVLA ALOHA normalized action must end in 7, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("TurboVLA ALOHA normalized action contains NaN/Inf")
        clipped = np.clip(action, -1.0, 1.0)
        return (
            0.5 * (clipped + 1.0) * (self.action_max - self.action_min)
            + self.action_min
        ).astype(np.float32)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "stats_key": self.stats_key,
        }
