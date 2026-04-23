"""
Single-ramp data collection environment.

Strips down the elevation terrain to a minimal controlled setup:
- Flat ground
- One inclined ramp (a tilted cuboid) at a known location
- Robot spawns behind the ramp, facing it, so every episode is a jump attempt

The goal is to produce dense "jump-phase" data: approach + impact + airborne + landing.
After collecting on the single-ramp env, you can mix this dataset with the full-terrain
dataset for training. The ramp geometry lives entirely in Python here — no USD file
needed.

RAMP GEOMETRY (coordinate frame: origin at world center, +X is forward)
  The ramp is a box 2m long (X) x 1m wide (Y) x 0.3m tall (Z), rotated ~15 deg
  around the Y axis so the +X edge is raised and the -X edge touches the ground.
  Centered at X = 0. The robot spawns at X = -4 m (behind the ramp, facing +X).

To tune the ramp, change RAMP_LENGTH / RAMP_WIDTH / RAMP_HEIGHT / RAMP_PITCH_DEG
below. No other code changes required.
"""

import math
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.sensors import RayCasterCfg, patterns
import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    TerminationTermCfg as DoneTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    SceneEntityCfg,
)

from wheeledlab.envs.mdp.observations import root_euler_xyz
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg


# --------------------------------------------------------------------------
# Ramp parameters — change these to try different ramp shapes
# --------------------------------------------------------------------------
RAMP_LENGTH = 2.0       # meters along robot's approach direction (X)
RAMP_WIDTH = 1.5        # meters perpendicular to approach (Y)
RAMP_HEIGHT = 0.30      # meters — how thick the box is
RAMP_PITCH_DEG = 15.0   # tilt angle (degrees around Y axis)
RAMP_CENTER_X = 0.0     # X position of ramp center
RAMP_CENTER_Y = 0.0     # Y position of ramp center

# Spawn zone — behind the ramp, facing it
SPAWN_X_RANGE = (-4.5, -3.5)   # 3.5-4.5 m behind ramp center
SPAWN_Y_RANGE = (-0.3, 0.3)    # small lateral noise so runs aren't identical
SPAWN_YAW_RANGE = (-0.15, 0.15) # ~ ±9 degrees of heading noise, facing +X
SPAWN_FORWARD_VEL = (2.0, 3.0) # world-frame +X velocity — robot points +X so this is forward


# --------------------------------------------------------------------------
# OBSERVATIONS (same as before, with sanitized height map)
# --------------------------------------------------------------------------

def world_height_map(env, sensor_cfg: SceneEntityCfg, offset: int, plane_init_value: int):
    """Corrected height map with nan/inf sanitization."""
    height_scan = -mdp.height_scan(env, sensor_cfg, offset)
    world_pos_z = mdp.root_pos_w(env)[..., 2] - plane_init_value
    corr_height_scan = height_scan + world_pos_z.unsqueeze(-1)
    corr_height_scan = torch.nan_to_num(corr_height_scan, nan=0.0, posinf=2.0, neginf=0.0)
    corr_height_scan = torch.clamp(corr_height_scan, min=-1.0, max=2.0)
    return corr_height_scan


@configclass
class DataCollectionObsCfg:
    @configclass
    class FullStateObs(ObsGroup):
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        world_euler_xyz = ObsTerm(func=root_euler_xyz)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        last_action = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        elevation_map = ObsTerm(
            func=world_height_map,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.084,
                "plane_init_value": 0.19,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: FullStateObs = FullStateObs()


# --------------------------------------------------------------------------
# SCENE — ground plane + one ramp, no USD terrain
# --------------------------------------------------------------------------

def _quat_from_y_rotation(angle_rad):
    """Quaternion (w, x, y, z) for rotation around Y axis only."""
    c = math.cos(angle_rad / 2.0)
    s = math.sin(angle_rad / 2.0)
    return (c, 0.0, s, 0.0)


# The ramp is tilted so its top surface is inclined. Because we're rotating a
# box around Y at its center, the center's Z needs to be RAMP_HEIGHT/2 so the
# lower edge is right at ground level (z=0). The inclined face will face +Z.
_RAMP_PITCH_RAD = math.radians(RAMP_PITCH_DEG)
_RAMP_QUAT = _quat_from_y_rotation(_RAMP_PITCH_RAD)
_RAMP_CENTER_Z = RAMP_HEIGHT / 2.0   # center height so box sits on ground before tilt


@configclass
class SingleRampSceneCfg(InteractiveSceneCfg):
    """Scene config: just a ground plane, a ramp, lighting, and the robot."""

    # Flat ground — large so robots don't drive off the edge even with bad spawns
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(200.0, 200.0),
            color=(0.3, 0.3, 0.3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
    )

    # The ramp itself — a tilted static cuboid
    ramp = AssetBaseCfg(
        prim_path="/World/ramp",
        spawn=sim_utils.CuboidCfg(
            size=(RAMP_LENGTH, RAMP_WIDTH, RAMP_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(RAMP_CENTER_X, RAMP_CENTER_Y, _RAMP_CENTER_Z),
            rot=_RAMP_QUAT,
        ),
    )

    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # The robot
    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Height scanner — observes geometry of both ground AND ramp
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(size=[2.5, 2.5], resolution=0.1),
        debug_vis=False,
        # Raycast against both the ground and the ramp so the elevation map
        # captures the obstacle geometry
        mesh_prim_paths=["/World/ground", "/World/ramp"],
    )

    def __post_init__(self):
        super().__post_init__()
        # Robot spawns on flat ground, not tilted terrain, so z=0.19 is fine
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.19))


# --------------------------------------------------------------------------
# TERMINATIONS
# --------------------------------------------------------------------------

def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty


def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0


@configclass
class SingleRampTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.05},  # lower threshold since we're on flat ground
    )
    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 60.0},
    )


# --------------------------------------------------------------------------
# EVENTS — simple spawn: always behind the ramp, facing it, with forward velocity
# --------------------------------------------------------------------------

@configclass
class SingleRampEventsCfg:

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (1.5, 2.5),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 10,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_.*link"),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.0, 0.8),
            "operation": "add",
        },
    )

    # Spawn robot behind the ramp, facing +X toward it, with forward velocity.
    # Because yaw is near zero, world-frame +X velocity == body-frame forward.
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": SPAWN_X_RANGE,
                "y": SPAWN_Y_RANGE,
                "yaw": SPAWN_YAW_RANGE,
            },
            "velocity_range": {
                "x": SPAWN_FORWARD_VEL,
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
        },
    )


# --------------------------------------------------------------------------
# ACTION CONFIG (unchanged — actions come from collection script)
# --------------------------------------------------------------------------

@configclass
class RandomActionCfg(Mushr4WDActionCfg):
    pass


# --------------------------------------------------------------------------
# MAIN ENV CONFIG
# --------------------------------------------------------------------------

@configclass
class SingleRampDataCollectionEnvCfg(ManagerBasedRLEnvCfg):
    """Data collection environment: one ramp, focused jump attempts."""

    seed: int = 42
    num_envs: int = 512
    env_spacing: float = 15.0  # Needs spacing now — each env has its own ramp

    observations: DataCollectionObsCfg = DataCollectionObsCfg()
    actions: RandomActionCfg = RandomActionCfg()
    events: SingleRampEventsCfg = SingleRampEventsCfg()
    terminations: SingleRampTerminationsCfg = SingleRampTerminationsCfg()
    rewards = None

    def __post_init__(self):
        super().__post_init__()

        self.viewer.eye = [-5.0, -5.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        self.sim.dt = 0.01
        self.decimation = 10
        self.sim.render_interval = self.decimation

        # Short episodes — robot should hit ramp in ~1s, be done within 4s
        self.episode_length_s = 4

        # Same action scaling as before
        self.actions.throttle_steer.scale = (3.0, 0.488)

        self.scene = SingleRampSceneCfg(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
        )


@configclass
class SingleRampDataCollectionPlayEnvCfg(SingleRampDataCollectionEnvCfg):
    """For visualization — fewer envs, same duration."""
    num_envs: int = 4
    episode_length_s: float = 6.0   # a bit longer so you can watch it land