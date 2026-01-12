#!/usr/bin/env python3
"""
Script for collecting dynamics data from MuSHR elevation environment.

This script runs robots with random control inputs and collects state-action-next_state
trajectories for training dynamics models.

Usage:
    python collect_dynamics_data.py --num_envs 512 --num_episodes 1000 --output_dir ./data/dynamics
"""

import argparse
import os
import torch
import numpy as np
from datetime import datetime
import h5py
from pathlib import Path

# IsaacLab imports
from isaaclab.app import AppLauncher


def main():
    """Main data collection loop."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Collect dynamics data from MuSHR environment")
    parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel environments")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect per environment")
    parser.add_argument("--output_dir", type=str, default="./data/dynamics", help="Output directory for data")
    parser.add_argument("--episode_length", type=float, default=10.0, help="Episode length in seconds")
    
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
    print(f"Output directory: {run_dir}")
    print(f"{'='*80}\n")
    
    # Create environment configuration
    env_cfg = MushrElevationDataCollectionEnvCfg()
    env_cfg.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length
    
    # Create environment (using ManagerBasedRLEnv for proper termination support)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Get observation dimensions
    obs_dim = env.observation_manager.group_obs_dim["policy"][0]
    print(f"Total observation dimension: {obs_dim}")
    print(f"Note: Observations include full state + elevation map (last 625 values)")
    
    # Data storage buffers
    episode_data = {
        'states': [],        # List of state tensors per timestep (includes elevation map)
        'actions': [],       # List of action tensors per timestep
        'next_states': [],   # List of next state tensors per timestep (includes elevation map)
        'episode_ids': [],   # Simple sequential episode ID for each sample
        'env_ids': [],       # Which environment each sample belongs to
        'timesteps': [],     # Timestep within episode
        'terminated': [],    # Termination flags (rollover, stuck, out-of-bounds)
        'truncated': [],     # Truncation flags (timeout)
    }
    # Tracking
    file_counter = 0
    episodes_completed = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    current_episode_steps = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    
    # Reset environment once at the start
    obs, _ = env.reset()
    
    print("Starting data collection...")
    print(f"Collecting {args.num_episodes} episodes across {args.num_envs} parallel environments...")
    print(f"Each episode should last {args.episode_length} seconds ({int(args.episode_length / (env.cfg.sim.dt * env.cfg.decimation))} control steps)\n")
    
    try:
        # Collect data continuously until we have enough episodes
        while episodes_completed.min().item() < args.num_episodes:
            
            # Sample random actions (normalized to [-1, 1])
            # Actions are sampled at EVERY timestep (10 Hz control frequency)
            # Action vector: [throttle, steering]
            #   - throttle: -1 (reverse) to +1 (forward), scaled by 3.0 m/s
            #   - steering: -1 (left) to +1 (right), scaled by 0.488 rad (~28°)
            actions = torch.rand((args.num_envs, env.action_manager.total_action_dim), 
                               device=env.device) * 2.0 - 1.0
            
            # Get current state (observations contain full state + elevation map)
            current_state = obs["policy"].clone()
            
            # Step environment with the random actions
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            # Get next state
            next_state = obs["policy"].clone()
            
            # Store transition with episode tracking per environment
            episode_data['states'].append(current_state.cpu())
            episode_data['actions'].append(actions.cpu())
            episode_data['next_states'].append(next_state.cpu())
            episode_data['episode_ids'].append(episodes_completed.cpu().clone())  # Each env has its own episode count
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
                
                min_eps = episodes_completed.min().item()
                max_eps = episodes_completed.max().item()
                print(f"Progress: {num_finished} envs finished | Episode counts - Min: {min_eps}, Max: {max_eps}/{args.num_episodes}")
                
                # Save data periodically (every 10 episodes minimum)
                if min_eps > 0 and min_eps % 10 == 0 and min_eps > file_counter * 10:
                    print(f"\nSaving batch {file_counter}...")
                    save_data(episode_data, run_dir, file_counter, env_cfg)
                    file_counter += 1
                    # Clear buffers after saving
                    for key in episode_data:
                        episode_data[key] = []
    
    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user.")
    
    finally:
        # Save remaining data
        if len(episode_data['states']) > 0:
            print("\nSaving final data...")
            save_data(episode_data, run_dir, file_counter, env_cfg)
        
        # Close environment
        env.close()
        
        print(f"\nData collection complete!")
        print(f"Data saved to: {run_dir}")
        print(f"Total files created: {file_counter + 1}")


def save_data(episode_data, output_dir, file_counter, env_cfg):
    """Save collected data to HDF5 file.
    
    State vector structure (concatenated in order):
    - root_pos_w: 3 values (x, y, z position in world frame)
    - root_quat_w: 4 values (w, x, y, z quaternion)
    - world_euler_xyz: 3 values (roll, pitch, yaw)
    - base_lin_vel: 3 values (vx, vy, vz in body frame)
    - base_ang_vel: 3 values (ωx, ωy, ωz in body frame)
    - root_lin_vel_w: 3 values (vx, vy, vz in world frame)
    - root_ang_vel_w: 3 values (ωx, ωy, ωz in world frame)
    - last_action: 2 values (throttle, steering from previous timestep)
    - joint_pos: N values (joint positions - wheels, suspension)
    - joint_vel: N values (joint velocities)
    - elevation_map: 625 values (25×25 heightmap grid, LAST 625 VALUES)
    
    Action vector structure:
    - actions: 2 values per timestep
        [0] throttle: -1.0 to +1.0 (scaled to ±3.0 m/s)
        [1] steering: -1.0 to +1.0 (scaled to ±0.488 rad ≈ ±28°)
    
    """
    
    output_file = output_dir / f"dynamics_data_{file_counter:04d}.h5"
    
    print(f"Saving data to {output_file}...")
    
    # Stack all timesteps
    states = torch.stack(episode_data['states'], dim=0)  # (T, N, state_dim)
    actions = torch.stack(episode_data['actions'], dim=0)  # (T, N, action_dim)
    next_states = torch.stack(episode_data['next_states'], dim=0)  # (T, N, state_dim)
    episode_ids = torch.stack(episode_data['episode_ids'], dim=0)  # (T, N)
    env_ids = torch.stack(episode_data['env_ids'], dim=0)  # (T, N)
    timesteps = torch.stack(episode_data['timesteps'], dim=0)  # (T, N)
    terminated = torch.stack(episode_data['terminated'], dim=0)  # (T, N)
    truncated = torch.stack(episode_data['truncated'], dim=0)  # (T, N)
    
    # Reshape to (samples, feature_dim)
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
        # Data
        f.create_dataset('states', data=states, compression='gzip')
        f.create_dataset('actions', data=actions, compression='gzip')
        f.create_dataset('next_states', data=next_states, compression='gzip')
        f.create_dataset('episode_ids', data=episode_ids, compression='gzip')
        f.create_dataset('env_ids', data=env_ids, compression='gzip')
        f.create_dataset('timesteps', data=timesteps, compression='gzip')
        f.create_dataset('terminated', data=terminated, compression='gzip')
        f.create_dataset('truncated', data=truncated, compression='gzip')
        
        # Metadata
        f.attrs['num_samples'] = len(states)
        f.attrs['state_dim'] = states.shape[1]
        f.attrs['action_dim'] = actions.shape[1]
        f.attrs['elevation_map_size'] = 625
        f.attrs['elevation_map_grid_size'] = 25  # 25×25 grid
        f.attrs['elevation_map_location'] = 'last_625_values'  # Position in state vector
        f.attrs['action_0_name'] = 'throttle'
        f.attrs['action_1_name'] = 'steering'
        f.attrs['action_0_scale'] = 3.0  # m/s
        f.attrs['action_1_scale'] = 0.488  # rad
        f.attrs['action_frequency'] = 'per_timestep'  # New action every timestep
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
