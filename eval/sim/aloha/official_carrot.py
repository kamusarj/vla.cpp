# Copyright (c) 2023 Robotic AI & Learning Lab Berkeley
# Copyright 2026 VinRobotics
#
# Adapted from octo-pytorch under the MIT License. See OCTO_LICENSE.

"""Carrot-in-cup task built on google-deepmind/aloha_sim physics.

This module only defines a task and a 7D-to-14D action adapter. Robot dynamics,
contacts, cameras, actuators, and stepping are provided by upstream
``aloha_sim`` and ``dm_control.composer``.
"""

from __future__ import annotations

from dataclasses import dataclass

from aloha_sim.tasks import dining_place_in_container
from aloha_sim.tasks.base import aloha2_task
from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer.variation import deterministic
import numpy as np


UPSTREAM_ALOHA_SIM_COMMIT = "d02904607cca1bf6dfb72f30b522506ac7ca0f91"
CONTROL_TIMESTEP = 0.02
DATASET_STEPS_PER_CONTROL = 3
EPISODE0_LEFT_BASE_YAW = 1.4949188896694098
EPISODE0_LEFT_BASE_POS = np.asarray(
    [-0.28033594, -0.22687428, 0.09], dtype=np.float64
)

DATASET_GRIPPER_CLOSE = 1.18576717
DATASET_GRIPPER_OPEN = 1.63062167
PLATE_PROP_PARK_Z = 0.020
PLATE_CENTER = (0.118, 0.145)
RELEASED_EE_DISTANCE = 0.15
SETTLED_LINEAR_SPEED = 0.02
DATASET_HOME_QPOS = np.asarray(
    [0.08590293, -0.79613602, 1.00168943, 0.03834952, 0.25770879, 0.03681554],
    dtype=np.float64,
)
DATASET_HOME_CTRL = np.asarray(
    [0.08283500, -0.76392251, 0.97867978, 0.04141748, 0.25464082, 0.04448544],
    dtype=np.float64,
)

_IDENTITY_QUAT = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
_CARROT_QUAT = _IDENTITY_QUAT.copy()


@dataclass(frozen=True)
class AlohaCarrotLayout:
    """Deterministic object placement for one evaluation task."""

    layout_id: str
    carrot_xy: tuple[float, float]
    cup_xy: tuple[float, float]
    description: str
    carrot_yaw: float = 0.0
    initial_left_qpos: tuple[float, float, float, float, float, float] | None = None


# One control layout plus nine deliberately different configurations spanning
# the left arm's reachable half of the table. The non-control layouts move
# objects by roughly 8--25 cm and vary their relative direction and distance.
ALOHA_CARROT_LAYOUTS = {
    layout.layout_id: layout
    for layout in (
        AlohaCarrotLayout(
            "episode0",
            (0.1180, 0.1450),
            (-0.0335, 0.1500),
            "dataset episode-0 placement",
        ),
        AlohaCarrotLayout(
            "layout_01",
            (0.0300, 0.0800),
            (-0.1200, 0.1800),
            "carrot front-right of cup; compact near workspace center",
        ),
        AlohaCarrotLayout(
            "layout_02",
            (0.1200, 0.0500),
            (-0.0800, 0.2000),
            "wide diagonal with carrot toward robot and cup far-left",
        ),
        AlohaCarrotLayout(
            "layout_03",
            (-0.0600, 0.1600),
            (0.0800, 0.0200),
            "reversed diagonal; cup is right and toward the robot",
        ),
        AlohaCarrotLayout(
            "layout_04",
            (0.1000, -0.0200),
            (-0.0800, 0.1300),
            "carrot near the front edge; cup back-left",
        ),
        AlohaCarrotLayout(
            "layout_05",
            (-0.1000, 0.0200),
            (0.0800, 0.1600),
            "carrot front-left; cup back-right",
        ),
        AlohaCarrotLayout(
            "layout_06",
            (0.0600, 0.1800),
            (-0.1200, 0.0500),
            "carrot far-right; cup front-left",
        ),
        AlohaCarrotLayout(
            "layout_07",
            (-0.0800, 0.1000),
            (0.0800, 0.1800),
            "cup to the right of the carrot",
        ),
        AlohaCarrotLayout(
            "layout_08",
            (0.1200, 0.1600),
            (-0.1000, -0.0100),
            "maximum diagonal separation within the reachable workspace",
        ),
        AlohaCarrotLayout(
            "layout_09",
            (-0.0335, 0.1500),
            (0.1180, 0.1450),
            "episode-0 carrot and cup positions swapped",
        ),
    )
}


def get_aloha_carrot_layout(layout_id: str) -> AlohaCarrotLayout:
    """Resolve a layout ID and report all valid IDs on invalid input."""

    try:
        return ALOHA_CARROT_LAYOUTS[str(layout_id)]
    except KeyError as exc:
        choices = ", ".join(ALOHA_CARROT_LAYOUTS)
        raise ValueError(
            f"Unknown ALOHA carrot layout {layout_id!r}; choose one of: {choices}"
        ) from exc


def jitter_aloha_carrot_layout(
    layout: AlohaCarrotLayout,
    *,
    xy_jitter: float,
    seed: int,
) -> AlohaCarrotLayout:
    """Apply a reproducible bounded XY perturbation to active objects."""

    xy_jitter = float(xy_jitter)
    if not np.isfinite(xy_jitter) or xy_jitter < 0.0:
        raise ValueError(f"xy_jitter must be finite and non-negative, got {xy_jitter}")
    if xy_jitter == 0.0:
        return layout
    offsets = np.random.default_rng(int(seed)).uniform(
        -xy_jitter,
        xy_jitter,
        size=(2, 2),
    )
    return AlohaCarrotLayout(
        layout_id=layout.layout_id,
        carrot_xy=tuple((np.asarray(layout.carrot_xy) + offsets[0]).tolist()),
        cup_xy=tuple((np.asarray(layout.cup_xy) + offsets[1]).tolist()),
        description=(
            f"{layout.description}; object XY jitter <= {xy_jitter:.4f} m "
            f"with seed {int(seed)}"
        ),
        carrot_yaw=layout.carrot_yaw,
        initial_left_qpos=layout.initial_left_qpos,
    )


class AlohaCarrotInCupTask(
    dining_place_in_container.DiningPlaceInContainer
):
    """Fixed-reset carrot-in-cup extension of the official dining task.

    The upstream pen and mug free bodies are retained, but their visual and
    collision geoms are replaced with dataset-matched primitive geometry: a
    short toy carrot with a green cap and a low blue handled mug. All contacts
    and robot dynamics remain inside official MuJoCo/composer code.
    """

    def __init__(
        self,
        *,
        layout_id: str = "episode0",
        layout: AlohaCarrotLayout | None = None,
        **kwargs,
    ):
        self._layout = layout or get_aloha_carrot_layout(layout_id)
        super().__init__(task_id="pen", **kwargs)
        self._object_sets["pen"]["instruction"] = (
            "Grasp the carrot from the plate, hold it, place it into the cup."
        )
        self._calibrate_left_robot_base()
        self._style_props()
        self._add_thin_plate_marker()

    @property
    def layout(self) -> AlohaCarrotLayout:
        return self._layout

    def set_layout(self, layout_id: str) -> None:
        """Select the deterministic placement used by the next reset."""

        self._layout = get_aloha_carrot_layout(layout_id)

    def set_custom_layout(self, layout: AlohaCarrotLayout) -> None:
        """Select a generated placement used by the next reset."""

        if not isinstance(layout, AlohaCarrotLayout):
            raise TypeError(f"Expected AlohaCarrotLayout, got {type(layout)!r}")
        self._layout = layout

    def _calibrate_left_robot_base(self) -> None:
        """Align the official left robot base to the episode-0 workcell."""

        base = self.root_entity.mjcf_model.find("body", r"left\base_link")
        if base is None:
            raise RuntimeError("Official ALOHA left/base_link body was not found")
        half_yaw = 0.5 * EPISODE0_LEFT_BASE_YAW
        base.pos = EPISODE0_LEFT_BASE_POS
        base.quat = (
            np.cos(half_yaw),
            0.0,
            0.0,
            np.sin(half_yaw),
        )

    def _style_props(self) -> None:
        # The dataset object is a short, thick toy carrot, not the elongated
        # writing pen supplied by the upstream dining task.
        for geom in self._pen_prop.mjcf_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0
            geom.rgba = (0.0, 0.0, 0.0, 0.0)
        self._add_dataset_carrot_geometry()

        # The upstream plate is substantially thicker than the real plate and
        # hides the carrot at the dataset-aligned grasp height. Hide that mesh;
        # _add_thin_plate_marker creates a non-colliding table-height plate.
        for body in self._plate_prop.mjcf_model.find_all("body"):
            body.gravcomp = 1.0
        for geom in self._plate_prop.mjcf_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0
            geom.rgba = (0.0, 0.0, 0.0, 0.0)

        # Replace the textured red upstream mug with dataset-matched blue mug
        # primitives while retaining its official free body.
        for geom in self._mug_prop.mjcf_model.find_all("geom"):
            geom.contype = 0
            geom.conaffinity = 0
            geom.rgba = (0.0, 0.0, 0.0, 0.0)
        self._add_dataset_cup_geometry()

        # Collision category 2 is reserved for the mug. The table accepts that
        # category while the category-1 robot does not, preventing the arm from
        # knocking over the mug. The category-1 carrot accepts category 2.
        table_geom = self.root_entity.mjcf_model.find("geom", "table")
        if table_geom is None:
            raise RuntimeError("Official ALOHA table geom was not found")
        table_geom.conaffinity = 3

        # Remove unrelated dining props from rendering and collision while
        # retaining the upstream model structure expected by Dining.
        for prop in (
            self._container_prop,
            self._banana_prop,
            self._bowl_prop,
        ):
            for body in prop.mjcf_model.find_all("body"):
                body.gravcomp = 1.0
            for geom in prop.mjcf_model.find_all("geom"):
                geom.contype = 0
                geom.conaffinity = 0
                geom.rgba = (0.0, 0.0, 0.0, 0.0)

    def _add_dataset_carrot_geometry(self) -> None:
        """Create the short orange toy carrot visible in episode 0."""

        body = self._pen_prop.mjcf_model.find_all("body")[0]
        visual = {"contype": 0, "conaffinity": 0, "density": 1}
        body.add(
            "geom",
            name="dataset_carrot_body",
            type="capsule",
            fromto=(-0.016, 0.0, 0.0, 0.012, 0.0, 0.0),
            size=(0.014,),
            rgba=(1.0, 0.31, 0.035, 1.0),
            **visual,
        )
        body.add(
            "geom",
            name="dataset_carrot_nose",
            type="ellipsoid",
            pos=(0.019, 0.0, 0.0),
            size=(0.013, 0.012, 0.011),
            rgba=(1.0, 0.36, 0.045, 1.0),
            **visual,
        )
        body.add(
            "geom",
            name="dataset_carrot_cap",
            type="cylinder",
            pos=(-0.023, 0.0, 0.0),
            quat=(0.70710678, 0.0, 0.70710678, 0.0),
            size=(0.014, 0.006),
            rgba=(0.08, 0.50, 0.24, 1.0),
            **visual,
        )
        # A flat collision hull prevents the round visual mesh from rolling
        # off the plate during reset, matching the stable toy in the dataset.
        body.add(
            "geom",
            name="dataset_carrot_collision",
            type="box",
            size=(0.030, 0.013, 0.013),
            rgba=(0.0, 0.0, 0.0, 0.0),
            contype=1,
            conaffinity=3,
            density=650,
            friction=(1.2, 0.01, 0.001),
        )

    def _add_dataset_cup_geometry(self) -> None:
        """Create a low blue handled mug matching the demonstration."""

        body = self._mug_prop.mjcf_model.find_all("body")[0]
        color = (0.08, 0.60, 0.93, 1.0)
        collision = {
            "contype": 2,
            "conaffinity": 0,
            "density": 1400,
            "friction": (1.3, 0.01, 0.001),
        }
        body.add(
            "geom",
            name="dataset_cup_base",
            type="cylinder",
            pos=(0.0, 0.0, 0.003),
            size=(0.043, 0.003),
            rgba=color,
            **collision,
        )
        wall_radius = 0.039
        wall_half_length = 0.011
        for index, theta in enumerate(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)):
            half_yaw = 0.5 * (theta + 0.5 * np.pi)
            body.add(
                "geom",
                name=f"dataset_cup_wall_{index:02d}",
                type="box",
                pos=(
                    wall_radius * np.cos(theta),
                    wall_radius * np.sin(theta),
                    0.040,
                ),
                quat=(np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)),
                size=(wall_half_length, 0.0035, 0.037),
                rgba=color,
                **collision,
            )

        # Rectangular handle, oriented toward negative x as in episode 0.
        body.add(
            "geom",
            name="dataset_cup_handle_top",
            type="box",
            pos=(-0.060, 0.0, 0.061),
            size=(0.020, 0.009, 0.005),
            rgba=color,
            **collision,
        )
        body.add(
            "geom",
            name="dataset_cup_handle_bottom",
            type="box",
            pos=(-0.060, 0.0, 0.021),
            size=(0.020, 0.009, 0.005),
            rgba=color,
            **collision,
        )
        body.add(
            "geom",
            name="dataset_cup_handle_outer",
            type="box",
            pos=(-0.080, 0.0, 0.041),
            size=(0.005, 0.009, 0.024),
            rgba=color,
            **collision,
        )

    def _add_thin_plate_marker(self) -> None:
        """Add a thin visual-only plate below the physical carrot."""

        worldbody = self.root_entity.mjcf_model.worldbody
        x, y = self._layout.carrot_xy
        self._plate_outer_geom = worldbody.add(
            "geom",
            name="episode0_plate_outer",
            type="cylinder",
            pos=(x, y, 0.028),
            size=(0.060, 0.002),
            rgba=(0.25, 0.62, 0.38, 1.0),
            contype=0,
            conaffinity=0,
        )
        self._plate_surface_geom = worldbody.add(
            "geom",
            name="episode0_plate_surface",
            type="cylinder",
            pos=(x, y, 0.0305),
            size=(0.052, 0.0005),
            rgba=(0.50, 0.78, 0.54, 1.0),
            contype=0,
            conaffinity=0,
        )

    def get_reward(self, physics) -> float:
        """Carrot-specific physical containment reward.

        Upstream's pen reward requires a long pen to overlap two separated
        height boxes. The real carrot is shorter and ends fully inside its cup,
        so this extension checks the same physical bodies for containment while
        the gripper is releasing.
        """

        metrics = self.release_metrics(physics)
        return float(
            metrics["inside_cup"]
            and metrics["physically_supported_in_cup"]
            and metrics["gripper_command_open"]
            and metrics["ee_distance"] >= RELEASED_EE_DISTANCE
            and metrics["carrot_linear_speed"] <= SETTLED_LINEAR_SPEED
        )

    def release_metrics(self, physics) -> dict[str, float | bool]:
        """Return physical placement and release diagnostics."""

        carrot_body = self._pen_prop.mjcf_model.find_all("body")[0]
        cup_body = self._mug_prop.mjcf_model.find_all("body")[0]
        left_gripper_site = next(
            site
            for site in self.root_entity.mjcf_model.find_all("site")
            if site.full_identifier == r"left\gripper"
        )
        carrot_pos = np.asarray(physics.bind(carrot_body).xpos)
        cup_pos = np.asarray(physics.bind(cup_body).xpos)
        ee_pos = np.asarray(physics.bind(left_gripper_site).xpos)
        xy_distance = float(np.linalg.norm(carrot_pos[:2] - cup_pos[:2]))
        inside_height = (cup_pos[2] - 0.005) <= carrot_pos[2] <= (
            cup_pos[2] + 0.13
        )
        gripper_command = aloha2_task.AlohaTask.convert_gripper(
            physics.data.ctrl[6], "sim_ctrl", "follower"
        )
        carrot_geom_ids = {
            int(physics.bind(geom).element_id)
            for geom in self._pen_prop.mjcf_model.find_all("geom")
        }
        cup_geom_ids = {
            int(physics.bind(geom).element_id)
            for geom in self._mug_prop.mjcf_model.find_all("geom")
        }
        carrot_cup_contact = any(
            (
                int(contact.geom1) in carrot_geom_ids
                and int(contact.geom2) in cup_geom_ids
            )
            or (
                int(contact.geom2) in carrot_geom_ids
                and int(contact.geom1) in cup_geom_ids
            )
            for contact in physics.data.contact[: physics.data.ncon]
        )
        return {
            "inside_cup": xy_distance <= 0.04 and inside_height,
            "carrot_cup_contact": carrot_cup_contact,
            # Convex collision decomposition can leave a deeply settled object
            # a fraction above the cup's contact pieces. Accept either direct
            # cup contact or a center below the lower 4 cm of the cup volume.
            "physically_supported_in_cup": bool(
                carrot_cup_contact or carrot_pos[2] <= cup_pos[2] + 0.04
            ),
            "gripper_command_open": bool(gripper_command >= 0.75),
            "ee_distance": float(np.linalg.norm(carrot_pos - ee_pos)),
            "carrot_linear_speed": float(
                np.linalg.norm(np.asarray(physics.bind(carrot_body).cvel)[3:])
            ),
        }

    def _sample_props(self, random_state):
        del random_state
        carrot_x, carrot_y = self._layout.carrot_xy
        cup_x, cup_y = self._layout.cup_xy
        return {
            "plate": np.asarray(
                [carrot_x, carrot_y, PLATE_PROP_PARK_Z],
                dtype=np.float64,
            ),
            "bowl": np.asarray([1.5, 1.5, 0.060]),
            "container": np.asarray([1.7, 1.5, 0.060]),
            "mug": np.asarray([cup_x, cup_y, 0.060], dtype=np.float64),
            "pen": np.asarray([carrot_x, carrot_y, 0.044], dtype=np.float64),
            "banana": np.asarray([1.9, 1.5, 0.060]),
        }

    def initialize_episode(self, physics, random_state) -> None:
        # Use the official Aloha initialization, then place the active objects
        # according to the selected deterministic evaluation layout.
        aloha2_task.AlohaTask.initialize_episode(self, physics, random_state)

        positions = self._sample_props(random_state)
        physics.bind(self._plate_outer_geom).pos[:2] = positions["plate"][:2]
        physics.bind(self._plate_surface_geom).pos[:2] = positions["plate"][:2]
        props = [
            self._plate_prop,
            self._bowl_prop,
            self._container_prop,
            self._mug_prop,
            self._pen_prop,
            self._banana_prop,
        ]
        carrot_quat = np.asarray(
            [
                np.cos(self._layout.carrot_yaw / 2.0),
                0.0,
                0.0,
                np.sin(self._layout.carrot_yaw / 2.0),
            ],
            dtype=np.float64,
        )
        quaternions = [
            _IDENTITY_QUAT,
            _IDENTITY_QUAT,
            _IDENTITY_QUAT,
            _IDENTITY_QUAT,
            carrot_quat,
            _IDENTITY_QUAT,
        ]
        initializers.PropPlacer(
            props=props,
            position=deterministic.Sequence(
                [
                    positions["plate"],
                    positions["bowl"],
                    positions["container"],
                    positions["mug"],
                    positions["pen"],
                    positions["banana"],
                ]
            ),
            quaternion=deterministic.Sequence(quaternions),
            ignore_collisions=True,
            settle_physics=False,
        )(physics, random_state)
        # Write the two active free bodies directly to stable table-aligned
        # poses instead of dropping round props through a long settling phase.
        root_joints = self.root_entity.mjcf_model.find_all("joint")
        carrot_joint = next(
            joint
            for joint in root_joints
            if joint.full_identifier == "writing_pens/"
        )
        cup_joint = next(
            joint for joint in root_joints if joint.full_identifier == "025_mug/"
        )
        physics.bind(carrot_joint).qpos = np.concatenate(
            [positions["pen"], carrot_quat]
        )
        physics.bind(carrot_joint).qvel = np.zeros(6)
        cup_pose = positions["mug"].copy()
        cup_pose[2] = 0.030
        physics.bind(cup_joint).qpos = np.concatenate([cup_pose, _IDENTITY_QUAT])
        physics.bind(cup_joint).qvel = np.zeros(6)
        arm_joints = physics.bind(self._joints)
        initial_left_qpos = np.asarray(
            self._layout.initial_left_qpos
            if self._layout.initial_left_qpos is not None
            else DATASET_HOME_QPOS,
            dtype=np.float64,
        )
        if initial_left_qpos.shape != (6,):
            raise ValueError(
                "initial_left_qpos must contain six left-arm joints, got "
                f"shape={initial_left_qpos.shape}"
            )
        arm_joints.qpos[:6] = initial_left_qpos
        arm_joints.qpos[6:8] = aloha2_task.SIM_GRIPPER_QPOS_OPEN
        physics.data.ctrl[:6] = (
            initial_left_qpos
            if self._layout.initial_left_qpos is not None
            else DATASET_HOME_CTRL
        )
        physics.data.ctrl[6] = aloha2_task.SIM_GRIPPER_CTRL_OPEN
        physics.forward()


def create_official_carrot_env(
    *,
    seed: int = 42,
    max_dataset_steps: int = 160,
    camera_resolution: tuple[int, int] = (480, 640),
    layout_id: str = "episode0",
    layout: AlohaCarrotLayout | None = None,
):
    """Create the official composer environment for a deterministic layout."""

    task = AlohaCarrotInCupTask(
        layout_id=layout_id,
        layout=layout,
        control_timestep=CONTROL_TIMESTEP,
        cameras=("overhead_cam", "wrist_cam_left"),
        camera_resolution=camera_resolution,
        joints_observation_delay_secs=0.0,
        image_observation_delay_secs=0.0,
        terminate_episode=True,
    )
    return composer.Environment(
        task,
        time_limit=max_dataset_steps
        * DATASET_STEPS_PER_CONTROL
        * CONTROL_TIMESTEP,
        strip_singleton_obs_buffer_dim=True,
        recompile_mjcf_every_episode=False,
        random_state=np.random.RandomState(seed),
    )


def dataset_gripper_to_official(value: float | np.ndarray) -> np.ndarray:
    fraction = (
        np.asarray(value, dtype=np.float64) - DATASET_GRIPPER_CLOSE
    ) / (DATASET_GRIPPER_OPEN - DATASET_GRIPPER_CLOSE)
    fraction = np.clip(fraction, 0.0, 1.0)
    return (
        aloha2_task.FOLLOWER_GRIPPER_CLOSE
        + fraction
        * (
            aloha2_task.FOLLOWER_GRIPPER_OPEN
            - aloha2_task.FOLLOWER_GRIPPER_CLOSE
        )
    )


def official_gripper_to_dataset(value: float | np.ndarray) -> np.ndarray:
    fraction = (
        np.asarray(value, dtype=np.float64)
        - aloha2_task.FOLLOWER_GRIPPER_CLOSE
    ) / (
        aloha2_task.FOLLOWER_GRIPPER_OPEN
        - aloha2_task.FOLLOWER_GRIPPER_CLOSE
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    return DATASET_GRIPPER_CLOSE + fraction * (
        DATASET_GRIPPER_OPEN - DATASET_GRIPPER_CLOSE
    )


def left_dataset_action_to_bimanual(action: np.ndarray) -> np.ndarray:
    """Map the dataset 7D left action to official 14D bimanual controls."""

    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,):
        raise ValueError(f"Expected dataset action shape (7,), got {action.shape}")
    official = np.concatenate(
        [aloha2_task.HOME_CTRL.copy(), aloha2_task.HOME_CTRL.copy()]
    ).astype(np.float32)
    official[:6] = action[:6]
    official[6] = float(dataset_gripper_to_official(action[6]))
    return official


@dataclass(frozen=True)
class OfficialRolloutState:
    step_index: int
    success: bool
    object_phase: str
    ee_pos: np.ndarray
    left_qpos: np.ndarray
    gripper: float
    carrot_pos: np.ndarray
    cup_pos: np.ndarray
