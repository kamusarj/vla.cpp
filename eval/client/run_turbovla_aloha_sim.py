#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run the TurboVLA ALOHA checkpoint in its MuJoCo carrot-to-cup task."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

EVAL_ROOT = Path(__file__).resolve().parents[1]
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from client.turbovla_aloha import TurboVlaAlohaStats  # noqa: E402
from client.vla_cpp_client import VlaCppClient  # noqa: E402
from sim.aloha import AlohaCarrotCupEnv  # noqa: E402
from sim.aloha.aloha_env import FPS, TASK_INSTRUCTION  # noqa: E402
from sim.aloha.official_carrot import CONTROL_TIMESTEP  # noqa: E402


VIDEO_FPS = 1.0 / CONTROL_TIMESTEP


def _video_frame(observation: dict, *, step: int, reward: float) -> np.ndarray:
    high = observation["_render_high_u8"]
    wrist = observation["_render_wrist_u8"]
    frame = np.concatenate([high, wrist], axis=0)
    return _annotate_video_frame(frame, step=step, reward=reward)


def _annotate_video_frame(frame: np.ndarray, *, step: int, reward: float) -> np.ndarray:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.putText(
        frame,
        f"step {step:03d}  success reward {reward:.0f}/1",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return frame


def _open_video(path: Path, observation: dict) -> cv2.VideoWriter:
    sample = _video_frame(observation, step=0, reward=0.0)
    height, width = sample.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create rollout video: {path}")
    writer.write(sample)
    return writer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vla-addr", default="tcp://127.0.0.1:5555")
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--tokenizer", default="google-bert/bert-base-uncased")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--n-action-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-condition", default="episode0")
    parser.add_argument(
        "--object-xy-jitter", type=float, default=0.0,
        help="Seeded XY jitter in metres for both carrot and cup.",
    )
    parser.add_argument("--recv-timeout-ms", type=int, default=120_000)
    parser.add_argument(
        "--dump-request-dir", type=Path, default=None,
        help="Dump exact protobuf requests, decoded tokens, state, and input images.",
    )
    parser.add_argument("--dump-request-limit", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/turbovla_aloha_sim")
    )
    parser.add_argument(
        "--no-randomize", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="Also open a passive MuJoCo viewer and run at wall-clock 16.67 Hz.",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if not 1 <= args.n_action_steps <= 12:
        parser.error("--n-action-steps must be in [1, 12]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = TurboVlaAlohaStats.from_json(args.stats_json)
    env = AlohaCarrotCupEnv(
        action_min=stats.action_min,
        action_max=stats.action_max,
        seed=args.seed,
        max_steps=args.max_steps,
        initial_condition=args.initial_condition,
        object_xy_jitter=0.0 if args.no_randomize else args.object_xy_jitter,
    )
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
        dump_request_dir=args.dump_request_dir,
        dump_request_limit=args.dump_request_limit,
    )

    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(env.model, env.data)

    episode_results: list[dict] = []
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset(seed=args.seed + episode)
            client.reset()
            video_path = args.output_dir / f"episode_{episode:03d}.mp4"
            writer = _open_video(video_path, observation)
            inference_ms: list[float] = []
            server_latency_ms: list[float] = []
            actions: list[np.ndarray] = []
            states: list[np.ndarray] = []
            final_info = reset_info
            reward = 0.0
            terminated = False
            truncated = False
            episode_start = time.perf_counter()
            try:
                for step in range(args.max_steps):
                    control_start = time.perf_counter()
                    previous_requests = client._step
                    states.append(
                        np.asarray(observation["observation.state"], dtype=np.float32).copy()
                    )
                    action = client.get_action(observation)
                    actions.append(np.asarray(action, dtype=np.float32).copy())
                    inference_ms.append((time.perf_counter() - control_start) * 1000.0)
                    if client._step > previous_requests and client._last_response is not None:
                        server_latency_ms.append(
                            float(client._last_response.latency_ms_total)
                        )

                    observation, reward, terminated, truncated, final_info = env.step(action)
                    for raw_frame in final_info.pop("video_frames", []):
                        writer.write(
                            _annotate_video_frame(
                                raw_frame, step=step + 1, reward=reward
                            )
                        )
                    if viewer is not None:
                        viewer.sync()
                        elapsed = time.perf_counter() - control_start
                        time.sleep(max(0.0, 1.0 / FPS - elapsed))
                    if step % 10 == 0 or reward > 0 or terminated or truncated:
                        carrot = np.asarray(final_info["carrot_position"])
                        print(
                            f"episode={episode} step={step + 1:03d} "
                            f"reward={reward:.0f}/1 requests={client._step} "
                            f"carrot={np.array2string(carrot, precision=3)}",
                            flush=True,
                        )
                    if terminated or truncated:
                        break
            finally:
                writer.release()

            steps_run = step + 1
            trajectory_path = args.output_dir / f"episode_{episode:03d}.npz"
            np.savez_compressed(
                trajectory_path,
                state=np.stack(states),
                action=np.stack(actions),
                server_latency_ms=np.asarray(server_latency_ms, dtype=np.float32),
            )
            result = {
                "episode": episode,
                "seed": args.seed + episode,
                "success": bool(final_info["is_success"]),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "steps": steps_run,
                "requests_sent": int(client._step),
                "max_reward": float(final_info["max_reward"]),
                "final_info": final_info,
                "mean_control_call_ms": float(np.mean(inference_ms)),
                "server_latency_ms": server_latency_ms,
                "elapsed_s": time.perf_counter() - episode_start,
                "video": str(video_path.resolve()),
                "trajectory": str(trajectory_path.resolve()),
                "action_min_observed": np.min(actions, axis=0).tolist(),
                "action_max_observed": np.max(actions, axis=0).tolist(),
            }
            episode_results.append(result)
            print(json.dumps(result, indent=2), flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        client.close()
        env.close()

    successes = sum(int(item["success"]) for item in episode_results)
    summary = {
        "kind": "google_deepmind_aloha_sim_rollout",
        "simulator": "google-deepmind/aloha_sim",
        "task": TASK_INSTRUCTION,
        "control_fps": FPS,
        "video_fps": VIDEO_FPS,
        "episodes": args.episodes,
        "successes": successes,
        "success_rate": successes / args.episodes,
        "n_action_steps": args.n_action_steps,
        "stats_json": str(args.stats_json.resolve()),
        "initial_condition": args.initial_condition,
        "object_xy_jitter": 0.0 if args.no_randomize else args.object_xy_jitter,
        "results": episode_results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {summary_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
