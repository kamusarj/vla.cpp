# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Any
import numpy as np
import torch
from tree import map_structure

from lerobot.envs.utils import preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.processor.pipeline import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION

class BasePipelineAdapter:
    def __init__(self, client: Any = None):
        self._client = client

    def reset(self):
        return self._client.reset()

    def get_action(self, obs: dict[str, Any]) -> np.ndarray:
        parsed_obs = self.parse_observation(obs)
        action = self._client.get_action(parsed_obs)
        parsed_action = self.parse_action(action)
        return parsed_action

    def get_action_from_queue(self) -> np.ndarray:
        action = self._client.get_action_from_queue()
        return self.parse_action(action)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LeRobotPipelineAdapter(BasePipelineAdapter):
    def __init__(self, client: Any = None):
        super().__init__(client)
        self._preprocessor = PolicyProcessorPipeline([LiberoProcessorStep()])
        self._postprocessor = PolicyProcessorPipeline([])

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        parsed_obs = map_structure(lambda x: x[None] if isinstance(x, np.ndarray) else x, obs)
        parsed_obs = preprocess_observation(parsed_obs)
        parsed_obs = self._preprocessor(parsed_obs)
        parsed_obs = map_structure(
            lambda x: (x.numpy()[0] if isinstance(x, torch.Tensor) else x), parsed_obs
        )
        parsed_obs["task"] = obs.get("task_description", "")
        return parsed_obs

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        action_transition = {ACTION: torch.from_numpy(action[None])}
        action_transition = self._postprocessor(action_transition)
        action = action_transition[ACTION].cpu().numpy()[0]
        return action

class Evo1PipelineAdapter(BasePipelineAdapter):
    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        front_img = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1])

        return {
            "image": [front_img, wrist_img, np.zeros_like(front_img)],
            "state": np.concatenate((
                obs["robot_state"]["eef"]["pos"],
                self.quat2axisangle(obs["robot_state"]["eef"]["quat"]),
                obs["robot_state"]["gripper"]["qpos"],
            )),
            "prompt": obs["task_description"],
            "image_mask": [1, 1, 0],
            "action_mask": [1] * 7 + [0] * 17,
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action[:7], dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action

    @staticmethod
    def encode_image_array(img_array: np.ndarray):
        return img_array.astype(np.uint8).tolist()

    @staticmethod
    def quat2axisangle(quat):
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den

class Gr00tPipelineAdapter(BasePipelineAdapter):

    def __init__(self, client: Any = None):
        super().__init__(client)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:

        front = np.ascontiguousarray(obs["pixels"]["image"][::-1, ::-1]).astype(np.uint8)
        wrist = np.ascontiguousarray(obs["pixels"]["image2"][::-1, ::-1]).astype(np.uint8)

        eef_pos = np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32)
        rpy = Evo1PipelineAdapter.quat2axisangle(
            np.array(obs["robot_state"]["eef"]["quat"], dtype=np.float64, copy=True)
        ).astype(np.float32)
        gripper = np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float32)

        def _scalar_state(v: float) -> np.ndarray:

            return np.array([[[v]]], dtype=np.float32)

        return {
            "video.image":       front[None, None],
            "video.wrist_image": wrist[None, None],
            "state.x":       _scalar_state(eef_pos[0]),
            "state.y":       _scalar_state(eef_pos[1]),
            "state.z":       _scalar_state(eef_pos[2]),
            "state.roll":    _scalar_state(rpy[0]),
            "state.pitch":   _scalar_state(rpy[1]),
            "state.yaw":     _scalar_state(rpy[2]),
            "state.gripper": gripper.reshape(1, 1, -1).astype(np.float32),

            "task": (obs.get("task_description", ""),),
            "annotation.human.action.task_description": (obs.get("task_description", ""),),
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:

        action = np.asarray(action[:7], dtype=np.float32).copy()
        action[6] = -1.0 if action[6] > 0.5 else 1.0
        return action

class Gr00tN15PipelineAdapter(Gr00tPipelineAdapter):

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action[:7], dtype=np.float32).copy()


class TurboVlaLiberoAdapter(BasePipelineAdapter):
    """Exact TurboVLA LIBERO camera/state/action preprocessing."""

    STATE_MEAN = np.asarray([
        -0.04651878296410724, 0.034409066171901814, 0.7645525131792095,
        2.9722095290211694, -0.2204697871882314, -0.12557940371042364,
        0.026914252831829258, -0.02719078368876073,
    ], dtype=np.float32)
    STATE_STD = np.asarray([
        0.10494395469120875, 0.15176619455037307, 0.37851671516755075,
        0.3442734256931591, 0.9069468528665473, 0.32539190239881105,
        0.014175903729549695, 0.014058894243853325,
    ], dtype=np.float32)
    ACTION_MIN = np.asarray([
        -0.9375, -0.9375, -0.9375, -0.2582142949104309,
        -0.375, -0.3675000071525574, -1.0,
    ], dtype=np.float32)
    ACTION_MAX = np.asarray([
        0.9375, 0.9375, 0.9375, 0.3557142913341522,
        0.375, 0.375, 1.0,
    ], dtype=np.float32)

    def parse_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        def image_chw(name: str) -> np.ndarray:
            image = np.ascontiguousarray(obs["pixels"][name][::-1, ::-1])
            return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))

        quat = np.asarray(obs["robot_state"]["eef"]["quat"],
                          dtype=np.float64).copy()
        state = np.concatenate((
            np.asarray(obs["robot_state"]["eef"]["pos"], dtype=np.float32),
            Evo1PipelineAdapter.quat2axisangle(quat).astype(np.float32),
            np.asarray(obs["robot_state"]["gripper"]["qpos"], dtype=np.float32),
        )).astype(np.float32)
        state = (state - self.STATE_MEAN) / (self.STATE_STD + 1e-6)
        return {
            "observation.images.image": image_chw("image"),
            "observation.images.image2": image_chw("image2"),
            "observation.state": state.astype(np.float32),
            "task": obs.get("task_description", ""),
        }

    def parse_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action[:7], dtype=np.float32)
        result = np.empty(7, dtype=np.float32)
        result[:6] = (0.5 * (normalized[:6] + 1.0)
                      * (self.ACTION_MAX[:6] - self.ACTION_MIN[:6])
                      + self.ACTION_MIN[:6])
        result[6] = 1.0 if normalized[6] >= 0.0 else -1.0
        return result
