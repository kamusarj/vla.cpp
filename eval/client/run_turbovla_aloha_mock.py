#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Closed-loop protocol simulation for a TurboVLA ALOHA vla-server.

This runner deliberately is not a physics benchmark. It renders deterministic
synthetic two-camera observations, sends raw 7-DoF ALOHA joint state through
the real client preprocessing/ZMQ/protobuf path, and applies predicted joint
targets to a first-order mock plant. A successful run proves that checkpoint
metadata, tokenization, normalization, action chunking, and client/server
transport agree end to end.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from turbovla_aloha import TurboVlaAlohaStats
from vla_cpp_client import VlaCppClient


class MockAlohaEnv:
    """Small deterministic 7-DoF plant with state-dependent RGB observations."""

    def __init__(self, stats: TurboVlaAlohaStats, response: float = 0.35):
        if not 0.0 < response <= 1.0:
            raise ValueError("response must be in (0, 1]")
        self.stats = stats
        self.response = float(response)
        self.state = stats.state_mean.copy()
        axis = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(axis, axis)
        self.step_index = 0

    def reset(self, initial_state: np.ndarray | None = None) -> dict:
        self.state = (
            self.stats.state_mean.copy()
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float32).reshape(7).copy()
        )
        if not np.all(np.isfinite(self.state)):
            raise ValueError("initial state contains NaN/Inf")
        self.step_index = 0
        return self.observation()

    def _render(self, view: int) -> np.ndarray:
        normalized = self.stats.normalize_state(self.state)
        phase = float(np.tanh(normalized.mean()))
        joint_wave = float(np.sin(normalized[:6]).mean())
        if view == 0:
            channels = (
                self.xx,
                self.yy,
                0.5 + 0.25 * phase + 0.15 * np.sin(2.0 * math.pi * self.xx),
            )
        else:
            channels = (
                self.yy,
                1.0 - self.xx,
                0.5 + 0.25 * joint_wave + 0.15 * np.cos(2.0 * math.pi * self.yy),
            )
        image = np.stack(channels, axis=0)
        return np.ascontiguousarray(np.clip(image, 0.0, 1.0), dtype=np.float32)

    def observation(self) -> dict:
        return {
            "observation.images.image": self._render(0),
            "observation.images.image2": self._render(1),
            "observation.state": self.state.copy(),
        }

    def step(self, joint_target: np.ndarray) -> dict:
        target = np.asarray(joint_target, dtype=np.float32).reshape(-1)
        if target.size != 7 or not np.all(np.isfinite(target)):
            raise ValueError(f"mock plant needs one finite 7-D target, got {target}")
        self.state += self.response * (target - self.state)
        self.step_index += 1
        return self.observation()


def _parse_state(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    result = np.fromstring(value, sep=",", dtype=np.float32)
    if result.size != 7:
        raise argparse.ArgumentTypeError("--initial-state needs 7 comma-separated values")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    parser.add_argument("--task", default="pick the object.")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--response", type=float, default=0.35)
    parser.add_argument("--initial-state", default=None)
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/turbovla_aloha_mock/summary.json"),
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be positive")
    if not 1 <= args.n_action_steps <= 12:
        parser.error("--n-action-steps must be in [1, 12]")
    try:
        initial_state = _parse_state(args.initial_state)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))

    stats = TurboVlaAlohaStats.from_json(args.stats_json)
    env = MockAlohaEnv(stats, response=args.response)
    obs = env.reset(initial_state)
    obs["task"] = args.task

    client = VlaCppClient(
        vla_addr=args.vla_addr,
        arch="turbovla_aloha",
        tokenizer_name=args.tokenizer,
        image_size=256,
        max_state_dim=7,
        real_action_dim=7,
        max_length=256,
        recv_timeout_ms=args.recv_timeout_ms,
        n_action_steps=args.n_action_steps,
        stats_json=args.stats_json,
    )
    client.reset()

    start = time.perf_counter()
    actions: list[np.ndarray] = []
    latencies_ms: list[float] = []
    protocol_ok = True
    try:
        for step in range(args.steps):
            action = client.get_action(obs)
            actions.append(action.copy())
            if not np.all(np.isfinite(action)):
                protocol_ok = False
                raise RuntimeError(f"non-finite action at step {step}: {action}")
            tolerance = 1e-5
            if np.any(action < stats.action_min - tolerance) or np.any(
                action > stats.action_max + tolerance
            ):
                protocol_ok = False
                raise RuntimeError(f"unnormalized action outside checkpoint bounds: {action}")
            obs = env.step(action)
            obs["task"] = args.task
            response = client._last_response
            if response is not None and len(latencies_ms) < client._step:
                latencies_ms.append(float(response.latency_ms_total))
            if not args.quiet:
                print(
                    f"step={step:02d} request_count={client._step} "
                    f"state={np.array2string(env.state, precision=4)}",
                    flush=True,
                )
    finally:
        client.close()

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    action_array = np.stack(actions)
    expected_requests = math.ceil(args.steps / args.n_action_steps)
    protocol_ok = protocol_ok and client._step == expected_requests
    summary = {
        "kind": "protocol_mock_not_physics_benchmark",
        "protocol_ok": bool(protocol_ok),
        "task": args.task,
        "steps": args.steps,
        "n_action_steps": args.n_action_steps,
        "requests_sent": client._step,
        "expected_requests": expected_requests,
        "elapsed_ms": elapsed_ms,
        "server_latency_ms": latencies_ms,
        "initial_state": (
            stats.state_mean.tolist() if initial_state is None else initial_state.tolist()
        ),
        "final_state": env.state.tolist(),
        "action_min_observed": action_array.min(axis=0).tolist(),
        "action_max_observed": action_array.max(axis=0).tolist(),
        "stats_json": str(args.stats_json),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.output.resolve()}", flush=True)
    return 0 if protocol_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
