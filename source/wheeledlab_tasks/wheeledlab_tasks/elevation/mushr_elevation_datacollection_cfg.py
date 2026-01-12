"""
Configuration for MuSHR elevation data collection.
Randomly controls robots through terrain to collect state-action trajectories
for training dynamics models.
"""

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
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

from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab.envs.mdp.observations import root_euler_xyz
from wheeledlab_assets.mushr import MUSHR_SUS_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg

# ##########################
# ###### OBSERVATIONS ######
# ##########################

def world_height_map(env, sensor_cfg: SceneEntityCfg, offset: int, plane_init_value: int):
    """Get corrected height map relative to world frame."""
    height_scan = -mdp.height_scan(env, sensor_cfg, offset)
    world_pos_z = mdp.root_pos_w(env)[..., 2] - plane_init_value
    corr_height_scan = height_scan + world_pos_z.unsqueeze(-1)
    return corr_height_scan


@configclass
class DataCollectionObsCfg:
    """Observation specification for data collection - includes full state information."""
    
    @configclass
    class FullStateObs(ObsGroup):
        """Full state observations for dynamics model training."""
        
        # Position and orientation
        root_pos_w = ObsTerm(func=mdp.root_pos_w)
        root_quat_w = ObsTerm(func=mdp.root_quat_w)
        world_euler_xyz = ObsTerm(func=root_euler_xyz)
        
        # Velocities
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        
        # Actions (controls)
        last_action = ObsTerm(func=mdp.last_action)
        
        # Joint states (wheel velocities, suspension positions)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        # Terrain information
        elevation_map = ObsTerm(
            func=world_height_map,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.084,
                "plane_init_value": 0.19
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: FullStateObs = FullStateObs()


##########################
### Scene Setup Config ###
##########################

@configclass
class DataCollectionTerrainImporterCfg(TerrainImporterCfg):
    """Terrain configuration for data collection."""
    
    height = 0.25
    prim_path = "/World/elevation"
    terrain_type = "usd"
    usd_path = f"{WHEELEDLAB_ASSETS_DATA_DIR}/Terrains/huge_compact.usd"
    collision_group = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    )
    debug_vis = False


@configclass
class DataCollectionSceneCfg(InteractiveSceneCfg):
    """Scene configuration for data collection."""
    
    terrain = DataCollectionTerrainImporterCfg()
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    ground = AssetBaseCfg(
        prim_path="/World/base",
        spawn=sim_utils.GroundPlaneCfg(
            size=(1600.0, 1200.0),
            color=(3, 3, 3),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.5,
                restitution=0.0
            )
        ),
    )

    robot: ArticulationCfg = MUSHR_SUS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mushr_nano/base_link",
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 20.0),
        ),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(size=[2.5, 2.5], resolution=0.1),
        debug_vis=False,
        mesh_prim_paths=["/World/elevation/terrain"],
    )

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, self.terrain.height)
        )


##########################
###### TERMINATION #######
##########################

def forward_vel(env):
    """Get forward velocity."""
    lin_vel = mdp.base_lin_vel(env)
    return torch.clamp(lin_vel[..., 0], max=1.2)


def forward_wheel_spin(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Get total wheel spin velocity."""
    asset = env.scene[asset_cfg.name]
    throttle_joints = asset.find_joints(".*_throttle")[0]
    throttle_joint_vel = mdp.joint_vel(env)[..., throttle_joints]
    sum_vels = torch.sum(throttle_joint_vel, dim=-1)
    return torch.clamp(sum_vels, max=200)


def upright_penalty(env, thresh_deg):
    """Check if robot is upright."""
    rot_mat = math_utils.matrix_from_quat(mdp.root_quat_w(env))
    up_dot = rot_mat[:, 2, 2]
    up_dot = torch.rad2deg(torch.arccos(up_dot))
    penalty = torch.where(up_dot > thresh_deg, up_dot - thresh_deg, 0.)
    return penalty


def upright_bool(env, thresh_deg):
    """Boolean check for rollover."""
    return upright_penalty(env, thresh_deg) > 0.0


def stuck(env, min_vel, wheel_spin_thr):
    """Check if robot is stuck."""
    not_moving = forward_vel(env) < min_vel
    throttle_joints_asset = SceneEntityCfg("robot", joint_names=".*throttle")
    joint_vels = mdp.joint_vel(env, asset_cfg=throttle_joints_asset)
    spinning_wheels = torch.sum(joint_vels, dim=-1) > wheel_spin_thr
    return torch.logical_and(not_moving, spinning_wheels)


@configclass
class DataCollectionTerminationsCfg:
    """Termination terms for data collection - same as training but no goal."""
    
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # (2) Robot fell below terrain
    cart_out_of_bounds = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15},
    )
    
    # (3) Robot stuck
    stuck = DoneTerm(
        func=stuck,
        params={
            "min_vel": 0.02,
            "wheel_spin_thr": 5.,
        },
    )
    
    # (4) Robot rolled over
    rollover = DoneTerm(
        func=upright_bool,
        params={"thresh_deg": 60.},
    )


#####################
###### EVENTS #######
#####################

@configclass
class DataCollectionEventsCfg:
    """Configuration for data collection events."""

    # Startup events - randomize physical properties for diversity
    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (1.5, 2.5),  # More variation for data diversity
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
            "mass_distribution_params": (0.0, 0.8),  # More variation
            "operation": "add",
        },
    )

    # Reset event - randomize spawn location
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-19., 19.), 
                "y": (-19., 19.), 
                "yaw": (-3.14, 3.14)
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        }
    )


###########################
###### ACTION CONFIG ######
###########################

@configclass
class RandomActionCfg(Mushr4WDActionCfg):
    """
    Random action configuration for data collection.
    Actions will be randomly sampled instead of from a policy.
    """
    pass


##########################
##### MAIN ENV CONFIG ####
##########################

@configclass
class MushrElevationDataCollectionEnvCfg(ManagerBasedRLEnvCfg):
    """
    Configuration for MuSHR elevation data collection environment.
    
    This environment is designed for collecting state-action-next_state trajectories
    for training dynamics models. Robots execute random control commands and their
    full state trajectories are recorded.
    
    Note: Uses ManagerBasedRLEnvCfg for proper termination support.
    """

    seed: int = 42
    num_envs: int = 512
    env_spacing: float = 0.

    # Observations include full state information
    observations: DataCollectionObsCfg = DataCollectionObsCfg()
    
    # Actions (will be randomly sampled)
    actions: RandomActionCfg = RandomActionCfg()

    # MDP settings
    events: DataCollectionEventsCfg = DataCollectionEventsCfg()
    terminations: DataCollectionTerminationsCfg = DataCollectionTerminationsCfg()
    
    # Rewards not needed but required for RL environment
    rewards = None

    def __post_init__(self):
        super().__post_init__()
        
        # Viewer settings
        self.viewer.eye = [20., -20.0, 20.0]
        self.viewer.lookat = [0.0, 0.0, 0.]

        # Simulation settings
        self.sim.dt = 0.01  # 100 Hz physics
        self.decimation = 10  # 10 Hz control
        self.sim.render_interval = self.decimation
        
        # Episode length for data collection (longer episodes = more data per trajectory)
        self.episode_length_s = 10  # 10 seconds per episode

        # Action scaling: (throttle_scale, steering_scale)
        self.actions.throttle_steer.scale = (3.0, 0.488)

        # Create scene
        self.scene = DataCollectionSceneCfg(
            num_envs=self.num_envs, 
            env_spacing=self.env_spacing,
        )


@configclass
class MushrElevationDataCollectionPlayEnvCfg(MushrElevationDataCollectionEnvCfg):
    """
    Play/visualization config for data collection - fewer environments for easier viewing.
    """
    num_envs: int = 16
    episode_length_s: float = 10.0  # Longer episodes for observation
