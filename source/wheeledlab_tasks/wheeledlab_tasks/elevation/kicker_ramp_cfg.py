"""
Kicker-ramp data collection environment.

Why this exists
---------------
The earlier full-terrain / single-ramp collectors had three problems that made the
data unusable for a *reachable* dynamics model:

  1. Steering was scaled to 0.0, so the model never saw steering affect the state.
  2. Robots were spawned at ~12 m/s to force them airborne — but the MuSHR tops out
     at 3 m/s (hardware limit), so every jump in that data was unreproducible by any
     legal action sequence. A planner/world model could never trigger it.
  3. The single-ramp scene used ONE global ramp prim while robots are per-env, so
     only the env at the world origin actually had a ramp in front of it.

This file fixes all three by changing the *ramp*, not the *speed*. The robot stays at
its real 3 m/s cap; the ramp is a short, steep "kicker" (~25-45 deg) that converts that
modest speed into a real jump:

    vertical launch speed ~ v * sin(angle)   ->   air height ~ (v sin angle)^2 / 2g

At 3 m/s a 40 deg kicker gives ~10 cm of air and ~0.3 s airborne — a genuine,
*reproducible* jump.

How the per-env ramps work
--------------------------
We use IsaacLab's TerrainGenerator (the same machinery rough-terrain locomotion uses
for raycast height scanning). It tiles a grid of sub-terrains into ONE mesh and gives
one spawn origin per cell, so the RayCaster height scanner "just works". We set
num_cols = num_envs and num_rows = 1 so every env gets its own distinct cell — i.e.
its own kicker — with no two robots sharing an origin (which would collide). Each
cell's ramp angle is randomized via the per-cell `difficulty` parameter.

Each cell is laid out along +X:

    | run-up (flat) | kicker (rises +X) | landing (flat) |
    ^spawn here, facing +X, v in +X

To tune the jumps, change KICKER_* below.
"""

import numpy as np
import trimesh
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
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
# Kicker geometry — tune these to change the jump
# --------------------------------------------------------------------------
CELL_SIZE_X = 9.0          # m along the approach direction (short run-up + ramp + landing)
CELL_SIZE_Y = 4.0          # m across
# Sweet spot for a 3 m/s robot is a SHORT, fairly steep kicker: too shallow and it just
# drives over; too steep (or too long) and it bleeds all its speed climbing and barely
# hops. A run ~= the robot's 0.33 m wheelbase is the shortest that's still physically
# climbable. Randomizing the angle across this range gives the model varied ramp shapes.
# Tuned EMPIRICALLY (the point-mass model misled us -- see below). What the data showed
# across runs:
#   - Air comes from a STEEP angle (42 deg gave 0.35 m air; 36 deg only 0.20 m).
#   - The run must be >= the 0.33 m wheelbase, or the robot bridges the ramp and never
#     tilts to its full angle -- a 0.28 m run actually gave LESS air, not more.
#   - Steep angles high-center / over-rotate, but rollover (80 deg, catches flips) plus
#     stuck (catches robots beached AT the ramp angle, ~40 deg, below the rollover bar)
#     TRUNCATE those crashes -> they reset and retry instead of polluting the data.
# So: long-enough run + steep angle for air, and lean on terminations to keep it clean.
KICKER_ANGLE_MIN_DEG = 30.0  # difficulty 0
KICKER_ANGLE_MAX_DEG = 42.0  # difficulty 1 -- steep, for real air; crashes are truncated
KICKER_RUN = 0.35          # m horizontal length (>= 0.33 m wheelbase so the robot engages)
# Wide ramp: forgiving of lateral drift and approach angle, so the robot reliably
# launches even with live steering. Combined with the short run-up below, this keeps
# almost every episode a clean jump while still recording steering->yaw dynamics.
KICKER_WIDTH = 3.0         # m ramp width (across Y); cell is 4 m wide -> 0.5 m margin
SPAWN_RUNUP = 1.0          # m of flat ground between spawn and the ramp foot (short:
                           # ~0.33 s at 3 m/s, too little time to veer off the ramp)

SPAWN_FORWARD_VEL = (2.7, 3.0)   # m/s — REACHABLE (<= 3 m/s cap), biased high so the
                                 # robot reliably carries enough speed to clear the ramp


# --------------------------------------------------------------------------
# Custom sub-terrain: flat ground + one steep kicker, robot spawns before it
# --------------------------------------------------------------------------
def kicker_terrain(difficulty: float, cfg: "KickerTerrainCfg"):
    """Build one cell: a flat ground plane with a single steep kicker ramp.

    The robot origin is returned on the flat run-up before the ramp, so driving
    +X takes it up the kicker and off the top edge.

    Args:
        difficulty: 0..1, linearly interpolates the kicker angle.
        cfg: the sub-terrain config (provides ``size``).

    Returns:
        (list[trimesh.Trimesh], origin) as required by the terrain generator.
    """
    size_x, size_y = cfg.size
    y_center = size_y / 2.0

    # Flat ground spanning the whole cell, from (0,0) to (size_x, size_y) at z=0.
    ground = make_plane((size_x, size_y), 0.0, center_zero=False)

    # Ramp angle from difficulty.
    angle_deg = cfg.angle_min_deg + difficulty * (cfg.angle_max_deg - cfg.angle_min_deg)
    angle = np.radians(angle_deg)

    run = cfg.run
    height = run * np.tan(angle)          # rise of the kicker
    x_start = cfg.runup                   # ramp foot
    x_end = x_start + run                 # launch edge (highest point)
    y0 = y_center - cfg.width / 2.0
    y1 = y_center + cfg.width / 2.0

    # Right-triangular prism: incline rises from (x_start, 0) to (x_end, height),
    # right angle at the back-bottom (x_end, 0). Robot launches off (x_end, height).
    verts = np.array([
        [x_start, y0, 0.0],   # 0 front-bottom (y0)
        [x_end,   y0, 0.0],   # 1 back-bottom  (y0)
        [x_end,   y0, height],# 2 launch edge  (y0)
        [x_start, y1, 0.0],   # 3 front-bottom (y1)
        [x_end,   y1, 0.0],   # 4 back-bottom  (y1)
        [x_end,   y1, height],# 5 launch edge  (y1)
    ])
    faces = np.array([
        [0, 1, 2],            # side y0
        [3, 5, 4],            # side y1
        [0, 2, 5], [0, 5, 3], # incline (drive-up) face
        [1, 4, 5], [1, 5, 2], # back vertical face
        [0, 3, 4], [0, 4, 1], # bottom face
    ])
    kicker = trimesh.Trimesh(vertices=verts, faces=faces)

    # Spawn origin: on the flat run-up, centered in Y, facing +X.
    origin = np.array([cfg.runup - SPAWN_RUNUP if cfg.runup > SPAWN_RUNUP else 1.0,
                       y_center, 0.0])
    return [ground, kicker], origin


@configclass
class KickerTerrainCfg(SubTerrainBaseCfg):
    """Sub-terrain config for a single steep kicker on flat ground."""

    function = kicker_terrain

    angle_min_deg: float = KICKER_ANGLE_MIN_DEG
    angle_max_deg: float = KICKER_ANGLE_MAX_DEG
    run: float = KICKER_RUN
    width: float = KICKER_WIDTH
    runup: float = 2.0   # ramp-foot X within the cell; spawn sits SPAWN_RUNUP behind it


# --------------------------------------------------------------------------
# OBSERVATIONS — sanitized height map (no inf/NaN when airborne)
# --------------------------------------------------------------------------
def world_height_map(env, sensor_cfg: SceneEntityCfg, offset: int, plane_init_value: int):
    """Corrected height map with nan/inf sanitization for airborne samples."""
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
# SCENE — generated terrain of per-env kickers + robot + height scanner
# --------------------------------------------------------------------------
@configclass
class KickerSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=42,
            size=(CELL_SIZE_X, CELL_SIZE_Y),
            num_rows=1,            # set to num_envs in __post_init__ via num_cols
            num_cols=1,            # overwritten to num_envs so each env gets its own cell
            curriculum=False,      # random kicker angle per cell (not a difficulty ramp)
            difficulty_range=(0.0, 1.0),
            border_width=0.0,
            sub_terrains={"kicker": KickerTerrainCfg(proportion=1.0)},
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(size=[2.5, 2.5], resolution=0.1),
        debug_vis=False,
        mesh_prim_paths=["/World/terrain/terrain"],   # generated mesh lives at {prim_path}/terrain
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.19))


# --------------------------------------------------------------------------
# TERMINATIONS — relaxed so airborne + landing transitions survive
# --------------------------------------------------------------------------
def upright_penalty(env, thresh_deg):
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    return torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)


def upright_bool(env, thresh_deg):
    return upright_penalty(env, thresh_deg) > 0.0


def forward_vel(env):
    """Body-frame forward velocity, clamped (only used for the stuck check)."""
    return torch.clamp(mdp.base_lin_vel(env)[..., 0], max=1.2)


def stuck(env, min_vel, wheel_spin_thr):
    """Robot beached: barely translating forward while its wheels keep spinning.

    Catches a robot high-centered on the ramp apex (pitched ~ the ramp angle, which is
    BELOW the 80 deg rollover bar, so rollover misses it). Safe in mid-air: an airborne
    robot still has forward body-vel ~1-2 m/s, well above min_vel, so this never fires
    during a real jump -- only when the robot is genuinely stuck and spinning its wheels.
    """
    not_moving = forward_vel(env) < min_vel
    joint_vels = mdp.joint_vel(env, asset_cfg=SceneEntityCfg("robot", joint_names=".*throttle"))
    spinning = torch.sum(joint_vels, dim=-1) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning)


@configclass
class KickerTerminationsCfg:
    """End on timeout, falling through the world, or a genuine flip.

    rollover is at a HIGH 80 deg (not the old 60). A clean jump off a <=36 deg ramp
    never tilts past ~50 deg in the air or on landing, so 80 deg preserves the entire
    takeoff/airborne/landing arc and only fires on a true flip/high-center. Crucially,
    terminating a crash RESETS the robot to retry instead of leaving it lying tilted-and-
    stalled for the rest of the episode -- that crash tail was ~25% of the old data and
    is exactly what we want to cut. stuck stays OFF (wheel-spin at takeoff is normal).
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_world = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -1.0},   # only if it falls through the floor
    )
    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 80.0},
    )
    # Truncate robots beached on the ramp apex (steep angles cause this). Resets them to
    # retry instead of leaving ~14 stuck-and-tilted steps in the data. min_vel small so it
    # only fires when truly not translating; never fires mid-jump (forward vel stays high).
    stuck = DoneTerm(
        func=stuck,
        params={"min_vel": 0.05, "wheel_spin_thr": 20.0},
    )


# --------------------------------------------------------------------------
# EVENTS — domain randomization + spawn before the ramp at REACHABLE speed
# --------------------------------------------------------------------------
@configclass
class KickerEventsCfg:

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

    # Spawn at the cell origin (flat run-up), facing +X, with a REACHABLE forward
    # speed. Small lateral / heading / speed noise so takeoff conditions vary across
    # episodes instead of being identical.
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-0.12, 0.12),     # ~ +/- 7 deg heading noise, still hits the ramp
            },
            "velocity_range": {
                "x": SPAWN_FORWARD_VEL,   # 2.5-3.0 m/s, reachable
                "y": (-0.1, 0.1),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
        },
    )


# --------------------------------------------------------------------------
# ACTIONS — steering LIVE (sampling is biased in the collection script)
# --------------------------------------------------------------------------
@configclass
class RandomActionCfg(Mushr4WDActionCfg):
    pass


# --------------------------------------------------------------------------
# MAIN ENV CONFIG
# --------------------------------------------------------------------------
@configclass
class KickerRampDataCollectionEnvCfg(ManagerBasedRLEnvCfg):
    """Data collection env: one randomized steep kicker per env, reachable jumps."""

    seed: int = 42
    num_envs: int = 512
    env_spacing: float = 0.0   # ignored for generator terrains (origins come from cells)

    observations: DataCollectionObsCfg = DataCollectionObsCfg()
    actions: RandomActionCfg = RandomActionCfg()
    events: KickerEventsCfg = KickerEventsCfg()
    terminations: KickerTerminationsCfg = KickerTerminationsCfg()
    rewards = None

    def __post_init__(self):
        super().__post_init__()

        self.viewer.eye = [-4.0, -4.0, 3.0]
        self.viewer.lookat = [4.0, 0.0, 0.0]

        self.sim.dt = 0.01
        self.decimation = 10
        self.sim.render_interval = self.decimation

        # Tight episodes so most samples are the jump, not flat coasting: spawn ~1 m
        # before the ramp at ~3 m/s -> hits in ~0.33 s, airborne ~0.3 s, then ~1 s of
        # landing + roll-out. 2 s captures the whole jump event with a little margin
        # (and >= the 1 s / 10-step training chunk used downstream).
        self.episode_length_s = 2.0

        # Steering is LIVE again (this was the worst bug in the old config).
        self.actions.throttle_steer.scale = (3.0, 0.488)

        self.scene = KickerSceneCfg(
            num_envs=self.num_envs,
            env_spacing=self.env_spacing,
        )
        # One distinct cell (kicker) per env so no two robots share an origin.
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = self.num_envs


@configclass
class KickerRampDataCollectionPlayEnvCfg(KickerRampDataCollectionEnvCfg):
    """For visualization — a handful of envs, slightly longer to watch the landing."""
    num_envs: int = 8
    episode_length_s: float = 3.0
