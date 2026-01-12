# MuSHR Elevation Data Collection

This guide explains how to collect dynamics data from the MuSHR robot navigating elevation terrain. The collected data can be used to train forward dynamics models for model-based control or world models.

## Overview

The data collection system runs 512 parallel robot simulations with random control inputs, recording full state trajectories including:
- Robot pose and velocities
- Joint states (wheels, suspension)
- Elevation maps (25×25 heightmap grid)
- Actions (throttle, steering)
- Termination flags (rollover, stuck, out-of-bounds, timeout)

## Files

- **Configuration**: `wheeledlab_tasks/elevation/mushr_elevation_datacollection_cfg.py`
- **Collection Script**: `scripts/collect_dynamics_data.py`
- **This Guide**: `docs/DATA_COLLECTION.md`

## Quick Start

### Basic Usage

```bash
cd /path/to/WheeledLab/source/wheeledlab_tasks

# Collect 10 episodes per environment (512 parallel environments)
python scripts/collect_dynamics_data.py \
    --num_envs 512 \
    --num_episodes 10 \
    --episode_length 10.0 \
    --output_dir ./data/dynamics
```

### With Visualization (Headless Mode Disabled)

```bash
# Fewer environments for easier viewing
python scripts/collect_dynamics_data.py \
    --num_envs 16 \
    --num_episodes 5 \
    --episode_length 10.0 \
    --output_dir ./data/dynamics \
    --enable_cameras
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_envs` | int | 512 | Number of parallel environments |
| `--num_episodes` | int | 10 | Number of episodes to collect **per environment** |
| `--episode_length` | float | 10.0 | Maximum episode length in seconds |
| `--output_dir` | str | `./data/dynamics` | Directory to save collected data |

### Additional IsaacLab Arguments

The script also accepts standard IsaacLab arguments:

```bash
# Run headless (no GUI) - recommended for data collection
python scripts/collect_dynamics_data.py --headless

# Enable GPU rendering
python scripts/collect_dynamics_data.py --enable_cameras

# Set specific GPU
python scripts/collect_dynamics_data.py --device cuda:0
```

## Output Data Format

### Directory Structure

```
./data/dynamics/
└── run_20260112_143052/          # Timestamped run directory
    ├── dynamics_data_0000.h5      # First batch (episodes 0-9)
    ├── dynamics_data_0001.h5      # Second batch (episodes 10-19)
    └── ...
```

### HDF5 File Contents

Each HDF5 file contains:

#### Datasets

| Dataset | Shape | Type | Description |
|---------|-------|------|-------------|
| `states` | `(N_samples, state_dim)` | float32 | Current state observations |
| `actions` | `(N_samples, 2)` | float32 | Actions taken [throttle, steering] |
| `next_states` | `(N_samples, state_dim)` | float32 | Next state observations |
| `episode_ids` | `(N_samples,)` | int64 | Episode number (0 to num_episodes-1) |
| `env_ids` | `(N_samples,)` | int64 | Environment ID (0 to num_envs-1) |
| `timesteps` | `(N_samples,)` | int64 | Timestep within episode (0 to ~100) |
| `terminated` | `(N_samples,)` | bool | True if episode terminated (rollover/stuck/OOB) |
| `truncated` | `(N_samples,)` | bool | True if episode truncated (timeout) |

#### Metadata Attributes

- `num_samples`: Total number of transitions
- `state_dim`: State vector dimension (typically ~650)
- `action_dim`: Action vector dimension (2)
- `elevation_map_size`: Number of elevation map points (625)
- `elevation_map_grid_size`: Grid resolution (25×25)
- `elevation_map_location`: Where in state vector (`"last_625_values"`)
- `action_0_name`, `action_1_name`: Action names (`"throttle"`, `"steering"`)
- `action_0_scale`, `action_1_scale`: Scaling factors (3.0 m/s, 0.488 rad)
- `episode_length_s`: Episode length (10.0 seconds)
- `sim_dt`: Physics timestep (0.01 seconds = 100 Hz)
- `decimation`: Control decimation factor (10)
- `control_dt`: Control timestep (0.1 seconds = 10 Hz)

### State Vector Structure

The state vector is a concatenation of the following (in order):

```python
# Total dimension: ~650 (depends on robot joint configuration)

# Pose (10 values)
root_pos_w        # [3] (x, y, z) position in world frame
root_quat_w       # [4] (w, x, y, z) quaternion orientation
world_euler_xyz   # [3] (roll, pitch, yaw) in radians

# Velocities (12 values)
base_lin_vel      # [3] (vx, vy, vz) linear velocity in body frame
base_ang_vel      # [3] (ωx, ωy, ωz) angular velocity in body frame
root_lin_vel_w    # [3] (vx, vy, vz) linear velocity in world frame
root_ang_vel_w    # [3] (ωx, ωy, ωz) angular velocity in world frame

# Control (2 values)
last_action       # [2] (throttle, steering) from previous timestep

# Joint States (varies, typically ~8-12 values)
joint_pos         # [N] joint positions (wheels, suspension)
joint_vel         # [N] joint velocities

# Terrain (625 values)
elevation_map     # [625] 25×25 heightmap grid (2.5m × 2.5m, 0.1m resolution)
                  # ALWAYS the last 625 values in the state vector
```

### Action Vector Structure

```python
actions[0]  # throttle: -1.0 to +1.0 (scaled to ±3.0 m/s target velocity)
actions[1]  # steering: -1.0 to +1.0 (scaled to ±0.488 rad ≈ ±28°)
```

## Loading Data in Python

### Example: Load and Parse Data

```python
import h5py
import numpy as np

# Load data
with h5py.File('data/dynamics/run_20260112_143052/dynamics_data_0000.h5', 'r') as f:
    states = f['states'][:]
    actions = f['actions'][:]
    next_states = f['next_states'][:]
    episode_ids = f['episode_ids'][:]
    env_ids = f['env_ids'][:]
    timesteps = f['timesteps'][:]
    terminated = f['terminated'][:]
    truncated = f['truncated'][:]
    
    # Load metadata
    state_dim = f.attrs['state_dim']
    action_dim = f.attrs['action_dim']
    elevation_map_size = f.attrs['elevation_map_size']
    
print(f"Loaded {len(states)} transitions")
print(f"State dimension: {state_dim}")
print(f"Action dimension: {action_dim}")
```

### Example: Extract Components from State

```python
# Extract elevation map (always last 625 values)
elevation_maps = states[:, -625:]  # Shape: (N_samples, 625)
elevation_maps = elevation_maps.reshape(-1, 25, 25)  # Shape: (N_samples, 25, 25)

# Extract core state (everything except elevation map)
core_states = states[:, :-625]  # Shape: (N_samples, state_dim - 625)

# Parse core state components
idx = 0
root_pos_w = core_states[:, idx:idx+3]; idx += 3
root_quat_w = core_states[:, idx:idx+4]; idx += 4
world_euler_xyz = core_states[:, idx:idx+3]; idx += 3
base_lin_vel = core_states[:, idx:idx+3]; idx += 3
base_ang_vel = core_states[:, idx:idx+3]; idx += 3
root_lin_vel_w = core_states[:, idx:idx+3]; idx += 3
root_ang_vel_w = core_states[:, idx:idx+3]; idx += 3
last_action = core_states[:, idx:idx+2]; idx += 2
# Remaining values are joint_pos and joint_vel
joint_states = core_states[:, idx:]
```

### Example: Filter Complete Episodes

```python
# Filter out incomplete episodes (only keep transitions where episode finished normally)
complete_mask = truncated  # Timeout = complete episode
complete_states = states[complete_mask]
complete_actions = actions[complete_mask]
complete_next_states = next_states[complete_mask]

print(f"Complete transitions: {complete_mask.sum()} / {len(complete_mask)}")
```

### Example: Group by Episode

```python
# Get unique episodes
unique_episodes = np.unique(episode_ids)

for ep_id in unique_episodes[:5]:  # First 5 episodes
    # Get all transitions for this episode
    ep_mask = (episode_ids == ep_id)
    ep_states = states[ep_mask]
    ep_actions = actions[ep_mask]
    ep_timesteps = timesteps[ep_mask]
    
    print(f"Episode {ep_id}: {len(ep_states)} timesteps")
```

## Configuration Details

### Environment Configuration (`mushr_elevation_datacollection_cfg.py`)

#### Key Parameters

```python
class MushrElevationDataCollectionEnvCfg(ManagerBasedRLEnvCfg):
    seed: int = 42                  # Random seed
    num_envs: int = 512             # Number of parallel environments
    env_spacing: float = 0.0        # Spacing between environments (0 = overlapping)
    
    episode_length_s: float = 10    # Maximum episode length (seconds)
    
    sim.dt: float = 0.01            # Physics timestep (100 Hz)
    decimation: int = 10            # Control decimation (10 Hz control)
    
    actions.throttle_steer.scale = (3.0, 0.488)  # Action scaling
```

#### Randomization (Domain Randomization)

The configuration includes domain randomization for better generalization:

```python
# Startup randomization (applied once per environment)
change_wheel_friction:
    static_friction_range: (1.5, 2.5)
    dynamic_friction_range: (0.8, 1.2)
    restitution_range: (0.0, 0.1)

add_base_mass:
    mass_distribution_params: (0.0, 0.8)  # Add 0-0.8 kg to base

# Reset randomization (applied each episode)
reset_robot_position:
    pose_range:
        x: (-19, 19)       # meters
        y: (-19, 19)       # meters
        yaw: (-π, π)       # radians
    velocity_range:
        x: (-0.2, 0.2)     # m/s
        y: (-0.2, 0.2)     # m/s
        yaw: (-0.5, 0.5)   # rad/s
```

#### Termination Conditions

Episodes terminate early if any of these occur:

1. **Rollover**: Robot tilts more than 60° from upright
2. **Stuck**: Wheels spinning but robot not moving (velocity < 0.02 m/s)
3. **Out of Bounds**: Robot falls below terrain (z < 0.15 m)
4. **Timeout**: Episode reaches 10 seconds (normal completion)

### Modifying the Configuration

To customize data collection, edit `mushr_elevation_datacollection_cfg.py`:

```python
# Example: Collect longer episodes
self.episode_length_s = 20  # 20 seconds instead of 10

# Example: Different action scaling
self.actions.throttle_steer.scale = (2.0, 0.4)  # Slower, less steering

# Example: More diverse physics
change_wheel_friction = EventTerm(
    params={
        "static_friction_range": (1.0, 3.0),  # Even wider range
        ...
    }
)
```

## Data Collection Behavior

### How It Works

1. **Initialization**: Creates 512 parallel environments with randomized physics properties
2. **Episode Loop**: For each environment, runs episodes until reaching `num_episodes`
3. **Random Actions**: At each 10Hz control step, samples random actions from uniform[-1, 1]
4. **Auto-Reset**: When an environment terminates, IsaacLab automatically resets it
5. **Progress Tracking**: Tracks episode count per environment independently
6. **Periodic Saving**: Saves data to HDF5 every 10 episodes (minimum across all envs)
7. **Completion**: Stops when slowest environment reaches `num_episodes`

### Expected Runtime

With default settings (512 envs, 10 episodes, 10 seconds each):
- **Total simulation time**: ~10 seconds × 10 episodes = 100 seconds
- **Wall-clock time**: ~2-5 minutes (depends on GPU)
- **Data generated**: ~50,000-100,000 transitions per batch file
- **Files created**: 1 file (saves at episode 10)

With more episodes (100 episodes):
- **Total simulation time**: ~1000 seconds
- **Wall-clock time**: ~15-30 minutes
- **Data generated**: ~500,000-1,000,000 transitions
- **Files created**: 10 files (saves every 10 episodes)

### Memory Usage

The script buffers data in memory before saving:
- **Memory per sample**: ~2.6 KB (state + action + metadata)
- **Memory per 10 episodes**: ~1-2 GB (512 envs × 100 steps × 10 episodes)
- **Saves automatically**: Every 10 episodes to prevent memory overflow

## Troubleshooting

### Issue: Episodes not reaching 10 seconds

**Symptom**: Episodes terminate early (e.g., at 2-3 seconds)

**Cause**: Robots are rolling over, getting stuck, or going out of bounds

**Solutions**:
- Check termination flags: `terminated` dataset shows why episodes ended
- Disable early termination for pure data collection:
  ```python
  # In mushr_elevation_datacollection_cfg.py
  class DataCollectionTerminationsCfg:
      time_out = DoneTerm(func=mdp.time_out, time_out=True)
      # Comment out other termination conditions
      # rollover = ...
      # stuck = ...
      # cart_out_of_bounds = ...
  ```

### Issue: Simulation is slow

**Solutions**:
- Run headless: `--headless`
- Reduce environments: `--num_envs 256`
- Check GPU utilization: `nvidia-smi`
- Ensure using GPU: `--device cuda:0`

### Issue: Out of memory

**Solutions**:
- Reduce `num_envs`: Try 256 instead of 512
- Data is saved periodically (every 10 episodes) to prevent memory issues
- If still problematic, reduce saving frequency in the script

### Issue: Data files are too large

**Solutions**:
- Data is already compressed with gzip
- Split into smaller batches (already done automatically every 10 episodes)
- Filter data after collection to keep only relevant transitions

## Tips for Model Training

### Data Preprocessing

```python
# Normalize states and actions
state_mean = states.mean(axis=0)
state_std = states.std(axis=0)
normalized_states = (states - state_mean) / (state_std + 1e-8)

# Split into train/val
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    np.arange(len(states)), 
    test_size=0.2, 
    random_state=42
)
```

### Data Augmentation

- **Time reversal**: Flip sign of velocities and actions
- **Mirror symmetry**: Flip left/right (y-axis, yaw, steering)
- **Elevation map rotation**: Rotate heightmap and adjust orientation

### Dataset Recommendations

- **Minimum**: 100k transitions (~20 episodes × 512 envs)
- **Recommended**: 500k-1M transitions for robust dynamics models
- **Large-scale**: 5M+ transitions for world models

## Next Steps

After collecting data:
1. **Visualize**: Plot trajectories, elevation maps, action distributions
2. **Validate**: Check for NaNs, extreme values, or anomalies
3. **Train**: Use data to train forward dynamics model (s, a → s')
4. **Evaluate**: Test model on held-out validation set

## Support

For issues or questions:
- Check IsaacLab documentation: [isaac-sim.github.io/IsaacLab](https://isaac-sim.github.io/IsaacLab)
- Review code comments in `collect_dynamics_data.py`
- Check termination conditions in `mushr_elevation_datacollection_cfg.py`
