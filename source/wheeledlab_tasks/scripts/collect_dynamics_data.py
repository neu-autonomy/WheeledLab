#!/usr/bin/env python3
"""
Script for collecting dynamics data from MuSHR elevation environment.

This script runs robots with random control inputs and collects state-action-next_state
trajectories for training dynamics models.

Changes from original:
- Action persistence: actions are held for N steps instead of resampled every step.
  This produces smoother, more "curved" trajectories and lets the robot build up
  enough momentum to actually clear ramps.
- Asymmetric sampling: throttle uses Beta(0.3, 0.3) so values cluster near -1/+1
  (aggressive driving), steering uses Beta(2, 2) so values cluster near 0
  (mostly gentle turns, occasional sharp ones). This gives the distribution of
  "commit to high throttle while turning moderately" that you need for jumps.
- Per-env resample timers: each env picks its own random persistence length so
  envs aren't all switching actions at the same global tick.

Usage:
    python collect_dynamics_data.py --num_envs 512 --num_episodes 1000 --output_dir ./data/dynamics
"""

import argparse
import os
import sys
import torch
import numpy as np
from datetime import datetime
import h5py
from pathlib import Path

# Add parent directory to path to allow imports of wheeledlab_tasks package
sys.path.insert(0, str(Path(__file__).parent.parent))

# IsaacLab imports
from isaaclab.app import AppLauncher


def sample_actions(num_envs, action_dim, device):
    """Sample a batch of actions with asymmetric distributions.

    - throttle (dim 0): Beta(0.3, 0.3) -> U-shaped, biases to full-forward/full-reverse
    - steering (dim 1+): Beta(2, 2)    -> bell-shaped around 0, occasional sharp turns

    Both are rescaled from [0, 1] to [-1, 1].
    """
    # Throttle: aggressive, commits to extremes
    throttle_dist = torch.distributions.Beta(0.3, 0.3)
    throttle = throttle_dist.sample((num_envs, 1)).to(device) * 2.0 - 1.0

    # Steering: mostly mild, occasionally sharp
    steering_dist = torch.distributions.Beta(2.0, 2.0)
    steering = steering_dist.sample((num_envs, action_dim - 1)).to(device) * 2.0 - 1.0

    return torch.cat([throttle, steering], dim=-1)


def main():
    """Main data collection loop."""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Collect dynamics data from MuSHR environment")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect per environment")
    parser.add_argument("--output_dir", type=str, default="./data/dynamics", help="Output directory for data")
    parser.add_argument("--episode_length", type=float, default=10.0, help="Episode length in seconds")
    # NEW: action persistence controls
    parser.add_argument("--min_hold_steps", type=int, default=10,
                        help="Minimum control steps to hold an action (10 steps = 1.0s at 10Hz)")
    parser.add_argument("--max_hold_steps", type=int, default=25,
                        help="Maximum control steps to hold an action (25 steps = 2.5s at 10Hz)")

    # Append AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # Launch the simulation
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import after launching (required for IsaacLab)
    from wheeledlab_tasks.elevation.mushr_elevation_datacollection_cfg import (
        MushrElevationDataCollectionEnvCfg
    )
    from isaaclab.envs import ManagerBasedRLEnv

    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"MuSHR Dynamics Data Collection")
    print(f"{'='*80}")
    print(f"Number of environments: {args.num_envs}")
    print(f"Episodes per environment: {args.num_episodes}")
    print(f"Episode length: {args.episode_length}s")
    print(f"Action hold steps: [{args.min_hold_steps}, {args.max_hold_steps}] "
          f"({args.min_hold_steps*0.1:.1f}s - {args.max_hold_steps*0.1:.1f}s)")
    print(f"Output directory: {run_dir}")
    print(f"{'='*80}\n")

    # Create environment configuration
    env_cfg = MushrElevationDataCollectionEnvCfg()
    env_cfg.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length
    # Update scene config to match num_envs (since __post_init__ was already called with defaults)
    env_cfg.scene.num_envs = args.num_envs

    # Create environment (using ManagerBasedRLEnv for proper termination support)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Get observation dimensions
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    action_dim = env.action_manager.total_action_dim
    print(f"Total observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Note: Observations include full state + elevation map (last 625 values)")

    # Data storage buffers
    episode_data = {
        'states': [],
        'actions': [],
        'next_states': [],
        'episode_ids': [],
        'env_ids': [],
        'timesteps': [],
        'terminated': [],
        'truncated': [],
    }
    # Tracking
    file_counter = 0
    episodes_completed = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    current_episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)

    # ---- NEW: action persistence state ----
    # Each env holds its current action and a per-env counter for how many
    # more steps it should keep holding it before resampling.
    current_actions = sample_actions(args.num_envs, action_dim, env.device)
    hold_remaining = torch.randint(
        args.min_hold_steps, args.max_hold_steps + 1,
        (args.num_envs,), device=env.device
    )
    # ----------------------------------------

    # Reset environment once at the start
    obs, _ = env.reset()

    print("Starting data collection...")
    print(f"Collecting {args.num_episodes} episodes across {args.num_envs} parallel environments...")
    print(f"Each episode should last {args.episode_length} seconds "
          f"({int(args.episode_length / (env.cfg.sim.dt * env.cfg.decimation))} control steps)\n")

    try:
        while episodes_completed.min().item() < args.num_episodes:

            # ---- Action persistence: only resample for envs whose hold expired ----
            needs_new_action = hold_remaining <= 0
            if needs_new_action.any():
                n = int(needs_new_action.sum().item())
                fresh = sample_actions(n, action_dim, env.device)
                current_actions[needs_new_action] = fresh
                # Pick new hold durations for those envs
                new_holds = torch.randint(
                    args.min_hold_steps, args.max_hold_steps + 1,
                    (n,), device=env.device
                )
                hold_remaining[needs_new_action] = new_holds

            actions = current_actions
            hold_remaining -= 1
            # -----------------------------------------------------------------------

            # Get current state
            current_state = obs["policy"].clone()

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            # Get next state
            next_state = obs["policy"].clone()

            # Store transition
            episode_data['states'].append(current_state.cpu())
            episode_data['actions'].append(actions.cpu().clone())
            episode_data['next_states'].append(next_state.cpu())
            episode_data['episode_ids'].append(episodes_completed.cpu().clone())
            episode_data['env_ids'].append(torch.arange(args.num_envs, dtype=torch.long))
            episode_data['timesteps'].append(current_episode_steps.cpu().clone())
            episode_data['terminated'].append(terminated.cpu())
            episode_data['truncated'].append(truncated.cpu())

            current_episode_steps += 1

            # Check which environments finished this step
            done = terminated | truncated
            if done.any():
                num_finished = done.sum().item()
                episodes_completed[done] += 1
                current_episode_steps[done] = 0

                # ---- NEW: force resample on reset so new episodes don't
                # inherit a stale action from the previous episode ----
                hold_remaining[done] = 0
                # -----------------------------------------------------

                min_eps = episodes_completed.min().item()
                max_eps = episodes_completed.max().item()
                print(f"Progress: {num_finished} envs finished | Episode counts - "
                      f"Min: {min_eps}, Max: {max_eps}/{args.num_episodes}")

                # Save data periodically
                if min_eps > 0 and min_eps % 10 == 0 and min_eps > file_counter * 10:
                    print(f"\nSaving batch {file_counter}...")
                    save_data(episode_data, run_dir, file_counter, env_cfg)
                    file_counter += 1
                    for key in episode_data:
                        episode_data[key] = []

    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user.")

    finally:
        if len(episode_data['states']) > 0:
            print("\nSaving final data...")
            save_data(episode_data, run_dir, file_counter, env_cfg)

        env.close()

        print(f"\nData collection complete!")
        print(f"Data saved to: {run_dir}")
        print(f"Total files created: {file_counter + 1}")


def save_data(episode_data, output_dir, file_counter, env_cfg):
    """Save collected data to HDF5 file. (unchanged from original)"""

    output_file = output_dir / f"dynamics_data_{file_counter:04d}.h5"
    print(f"Saving data to {output_file}...")

    states = torch.stack(episode_data['states'], dim=0)
    actions = torch.stack(episode_data['actions'], dim=0)
    next_states = torch.stack(episode_data['next_states'], dim=0)
    episode_ids = torch.stack(episode_data['episode_ids'], dim=0)
    env_ids = torch.stack(episode_data['env_ids'], dim=0)
    timesteps = torch.stack(episode_data['timesteps'], dim=0)
    terminated = torch.stack(episode_data['terminated'], dim=0)
    truncated = torch.stack(episode_data['truncated'], dim=0)

    T, N = states.shape[0], states.shape[1]
    states = states.reshape(T * N, -1).numpy()
    actions = actions.reshape(T * N, -1).numpy()
    next_states = next_states.reshape(T * N, -1).numpy()
    episode_ids = episode_ids.reshape(T * N).numpy()
    env_ids = env_ids.reshape(T * N).numpy()
    timesteps = timesteps.reshape(T * N).numpy()
    terminated = terminated.reshape(T * N).numpy()
    truncated = truncated.reshape(T * N).numpy()

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('states', data=states, compression='gzip')
        f.create_dataset('actions', data=actions, compression='gzip')
        f.create_dataset('next_states', data=next_states, compression='gzip')
        f.create_dataset('episode_ids', data=episode_ids, compression='gzip')
        f.create_dataset('env_ids', data=env_ids, compression='gzip')
        f.create_dataset('timesteps', data=timesteps, compression='gzip')
        f.create_dataset('terminated', data=terminated, compression='gzip')
        f.create_dataset('truncated', data=truncated, compression='gzip')

        f.attrs['num_samples'] = len(states)
        f.attrs['state_dim'] = states.shape[1]
        f.attrs['action_dim'] = actions.shape[1]
        f.attrs['elevation_map_size'] = 625
        f.attrs['elevation_map_grid_size'] = 25
        f.attrs['elevation_map_location'] = 'last_625_values'
        f.attrs['action_0_name'] = 'throttle'
        f.attrs['action_1_name'] = 'steering'
        f.attrs['action_0_scale'] = 3.0
        f.attrs['action_1_scale'] = 0.488
        f.attrs['action_frequency'] = 'persistent_hold'  # CHANGED
        f.attrs['num_envs'] = N
        f.attrs['num_timesteps'] = T
        f.attrs['episode_length_s'] = env_cfg.episode_length_s
        f.attrs['sim_dt'] = env_cfg.sim.dt
        f.attrs['decimation'] = env_cfg.decimation
        f.attrs['control_dt'] = env_cfg.sim.dt * env_cfg.decimation

    print(f"Saved {len(states)} samples to {output_file}")
    print(f"  - State dim: {states.shape[1]} (includes 625-value elevation map at end)")
    print(f"  - Action dim: {actions.shape[1]} (throttle, steering)")


if __name__ == "__main__":
    main()