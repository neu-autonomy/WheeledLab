import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import argparse


def parse_state_vector(states):
    """Parse state vector into individual components.
    
    State vector structure:
    - root_pos_w: [3] (x, y, z)
    - root_quat_w: [4] (w, x, y, z)
    - world_euler_xyz: [3] (roll, pitch, yaw)
    - base_lin_vel: [3] (vx, vy, vz)
    - base_ang_vel: [3] (ωx, ωy, ωz)
    - root_lin_vel_w: [3] (vx, vy, vz)
    - root_ang_vel_w: [3] (ωx, ωy, ωz)
    - last_action: [2] (throttle, steering)
    - joint_pos: [N] joint positions
    - joint_vel: [N] joint velocities
    - elevation_map: [676] heightmap (LAST 676 values = 26x26)
    """
    # Extract elevation map (always last 676 values = 26x26 grid).
    # NOT 625/25x25 -- GridPatternCfg(size=2.5, res=0.1) -> 26 points/axis. Verified
    # against the data; the 625 slice misaligns every grid row.
    elevation_maps = states[:, -676:].reshape(-1, 26, 26)

    # Extract core state (everything except elevation map)
    core_states = states[:, :-676]
    
    # Parse components
    idx = 0
    root_pos_w = core_states[:, idx:idx+3]; idx += 3      # XYZ position
    root_quat_w = core_states[:, idx:idx+4]; idx += 4     # Quaternion
    world_euler_xyz = core_states[:, idx:idx+3]; idx += 3 # Roll, Pitch, Yaw
    base_lin_vel = core_states[:, idx:idx+3]; idx += 3
    base_ang_vel = core_states[:, idx:idx+3]; idx += 3
    root_lin_vel_w = core_states[:, idx:idx+3]; idx += 3
    root_ang_vel_w = core_states[:, idx:idx+3]; idx += 3
    last_action = core_states[:, idx:idx+2]; idx += 2
    
    return {
        'xyz': root_pos_w,
        'quaternion': root_quat_w,
        'rpy': world_euler_xyz,
        'lin_vel': base_lin_vel,
        'ang_vel': base_ang_vel,
        'actions': last_action,
        'elevation_maps': elevation_maps
    }


def main(args):
    # Load data
    file_path = args.file_path
    # if file_path is None or the extension is not h5:
    if file_path is None or not file_path.endswith('.h5'):
        raise ValueError("Please provide a valid file path using --file_path argument.")

    with h5py.File(file_path, 'r') as f:
    # with h5py.File('data/dynamics/run_20260112_143052/dynamics_data_0000.h5', 'r') as f:
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
    print(f"Elevation map size: {elevation_map_size}")
    print()
    
    # Parse state vector
    parsed_data = parse_state_vector(states)
    xyz = parsed_data['xyz']
    rpy = parsed_data['rpy']
    elevation_maps = parsed_data['elevation_maps']
    
    # Select which episode and environment to visualize
    if args.episode_id is not None:
        ep_id = args.episode_id
    else:
        ep_id = episode_ids[0]  # Default to first episode
    
    if args.env_id is not None:
        env_id = args.env_id
    else:
        env_id = env_ids[0]  # Default to first environment
    
    # Filter data for selected episode and environment
    mask = (episode_ids == ep_id) & (env_ids == env_id)
    
    if mask.sum() == 0:
        print(f"Warning: No data found for episode {ep_id}, env {env_id}")
        print(f"Available episodes: {np.unique(episode_ids)}")
        print(f"Available envs: {np.unique(env_ids)}")
        return
    
    # Extract trajectory for this episode/env
    traj_xyz = xyz[mask]
    traj_rpy = rpy[mask]
    traj_elevation = elevation_maps[mask]
    traj_timesteps = timesteps[mask]
    traj_actions = actions[mask]
    traj_terminated = terminated[mask]
    traj_truncated = truncated[mask]
    
    print(f"Visualizing Episode {ep_id}, Environment {env_id}")
    print(f"Trajectory length: {len(traj_xyz)} timesteps")
    print(f"Duration: {len(traj_xyz) * 0.1:.1f} seconds")
    print(f"Terminated: {traj_terminated.any()}, Truncated: {traj_truncated.any()}")
    print()
    
    # Create visualization
    if args.mode == 'static':
        plot_static(traj_xyz, traj_rpy, traj_elevation, traj_actions, traj_timesteps)
    elif args.mode == 'animation':
        plot_animation(traj_xyz, traj_rpy, traj_elevation, traj_actions, traj_timesteps, args.fps)
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Choose 'static' or 'animation'")


def plot_static(xyz, rpy, elevation_maps, actions, timesteps):
    """Create static plots of the entire trajectory."""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Time axis
    time = timesteps * 0.1  # Convert to seconds (10 Hz control)
    
    # 1. XYZ Position over time (3 subplots)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(time, xyz[:, 0], 'b-', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(time, xyz[:, 1], 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y Position')
    ax2.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(time, xyz[:, 2], 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Z Position (Height)')
    ax3.grid(True, alpha=0.3)
    
    # 2. Roll, Pitch, Yaw over time (3 subplots)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(time, np.rad2deg(rpy[:, 0]), 'b-', linewidth=2)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Roll (deg)')
    ax4.set_title('Roll')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(time, np.rad2deg(rpy[:, 1]), 'g-', linewidth=2)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pitch (deg)')
    ax5.set_title('Pitch')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(time, np.rad2deg(rpy[:, 2]), 'r-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Yaw (deg)')
    ax6.set_title('Yaw')
    ax6.grid(True, alpha=0.3)
    
    # 3. XY Trajectory (top-down view)
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(xyz[:, 0], xyz[:, 1], c=time, cmap='viridis', s=20)
    ax7.plot(xyz[:, 0], xyz[:, 1], 'k-', alpha=0.3, linewidth=1)
    ax7.plot(xyz[0, 0], xyz[0, 1], 'go', markersize=10, label='Start')
    ax7.plot(xyz[-1, 0], xyz[-1, 1], 'ro', markersize=10, label='End')
    ax7.set_xlabel('X Position (m)')
    ax7.set_ylabel('Y Position (m)')
    ax7.set_title('XY Trajectory (Top View)')
    ax7.axis('equal')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    plt.colorbar(scatter, ax=ax7, label='Time (s)')
    
    # 4. Actions over time
    ax8 = plt.subplot(3, 3, 8)
    ax8.plot(time, actions[:, 0], 'b-', linewidth=2, label='Throttle')
    ax8.plot(time, actions[:, 1], 'r-', linewidth=2, label='Steering')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Action Value [-1, 1]')
    ax8.set_title('Actions')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # 5. Elevation map at middle of trajectory
    ax9 = plt.subplot(3, 3, 9)
    mid_idx = len(elevation_maps) // 2
    elevation_map = elevation_maps[mid_idx]
    
    im = ax9.imshow(elevation_map, cmap='terrain', origin='lower', 
                    extent=[-1.25, 1.25, -1.25, 1.25])
    ax9.plot(0, 0, 'r+', markersize=20, markeredgewidth=3, label='Robot')
    ax9.set_xlabel('Y (m)')
    ax9.set_ylabel('X (m)')
    ax9.set_title(f'Elevation Map (t={time[mid_idx]:.1f}s)')
    ax9.legend()
    plt.colorbar(im, ax=ax9, label='Height (m)')
    
    plt.tight_layout()
    plt.show()


def plot_animation(xyz, rpy, elevation_maps, actions, timesteps, fps=10):
    """Create animated visualization of the trajectory with 3D view."""
    
    fig = plt.figure(figsize=(20, 12))
    
    time = timesteps * 0.1
    
    # Create grid layout: 4 rows, 4 columns
    # Top row: 3D trajectory + XYZ positions
    # Second row: Roll, Pitch, Yaw + XY trajectory
    # Third row: Actions + Elevation map
    
    ax_3d = fig.add_subplot(3, 4, 1, projection='3d')  # 3D trajectory
    ax1 = plt.subplot(3, 4, 2)  # X position
    ax2 = plt.subplot(3, 4, 3)  # Y position
    ax3 = plt.subplot(3, 4, 4)  # Z position
    
    ax4 = plt.subplot(3, 4, 5)  # Roll
    ax5 = plt.subplot(3, 4, 6)  # Pitch
    ax6 = plt.subplot(3, 4, 7)  # Yaw
    ax7 = plt.subplot(3, 4, 8)  # XY trajectory
    
    ax8 = plt.subplot(3, 4, 9)  # Actions
    ax9 = plt.subplot(3, 4, 10)  # Elevation map
    ax_info = plt.subplot(3, 4, 11)  # Info text
    ax_info.axis('off')
    
    # ===== 3D Trajectory Setup =====
    line_3d, = ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.6)
    point_3d, = ax_3d.plot([], [], [], 'ro', markersize=8)
    
    # Orientation arrows (RPY visualization)
    # Forward (X), Left (Y), Up (Z) in body frame
    arrow_x = None  # Will be initialized in animate
    arrow_y = None
    arrow_z = None
    
    ax_3d.set_xlim(xyz[:, 0].min() - 1, xyz[:, 0].max() + 1)
    ax_3d.set_ylim(xyz[:, 1].min() - 1, xyz[:, 1].max() + 1)
    ax_3d.set_zlim(xyz[:, 2].min() - 0.5, xyz[:, 2].max() + 0.5)
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Trajectory')
    ax_3d.view_init(elev=20, azim=45)
    
    # ===== Time Series Plots Setup =====
    line1, = ax1.plot([], [], 'b-', linewidth=2)
    point1, = ax1.plot([], [], 'ro', markersize=8)
    ax1.set_xlim(0, time[-1])
    ax1.set_ylim(xyz[:, 0].min() - 0.5, xyz[:, 0].max() + 0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (m)')
    ax1.set_title('X Position')
    ax1.grid(True, alpha=0.3)
    
    line2, = ax2.plot([], [], 'g-', linewidth=2)
    point2, = ax2.plot([], [], 'ro', markersize=8)
    ax2.set_xlim(0, time[-1])
    ax2.set_ylim(xyz[:, 1].min() - 0.5, xyz[:, 1].max() + 0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_title('Y Position')
    ax2.grid(True, alpha=0.3)
    
    line3, = ax3.plot([], [], 'r-', linewidth=2)
    point3, = ax3.plot([], [], 'ro', markersize=8)
    ax3.set_xlim(0, time[-1])
    ax3.set_ylim(xyz[:, 2].min() - 0.1, xyz[:, 2].max() + 0.1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Z Position (Height)')
    ax3.grid(True, alpha=0.3)
    
    line4, = ax4.plot([], [], 'b-', linewidth=2)
    point4, = ax4.plot([], [], 'ro', markersize=8)
    ax4.set_xlim(0, time[-1])
    roll_deg = np.rad2deg(rpy[:, 0])
    ax4.set_ylim(roll_deg.min() - 5, roll_deg.max() + 5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Roll (deg)')
    ax4.set_title('Roll')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    line5, = ax5.plot([], [], 'g-', linewidth=2)
    point5, = ax5.plot([], [], 'ro', markersize=8)
    ax5.set_xlim(0, time[-1])
    pitch_deg = np.rad2deg(rpy[:, 1])
    ax5.set_ylim(pitch_deg.min() - 5, pitch_deg.max() + 5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Pitch (deg)')
    ax5.set_title('Pitch')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    line6, = ax6.plot([], [], 'r-', linewidth=2)
    point6, = ax6.plot([], [], 'ro', markersize=8)
    ax6.set_xlim(0, time[-1])
    yaw_deg = np.rad2deg(rpy[:, 2])
    ax6.set_ylim(yaw_deg.min() - 10, yaw_deg.max() + 10)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Yaw (deg)')
    ax6.set_title('Yaw')
    ax6.grid(True, alpha=0.3)
    
    # XY trajectory
    line7, = ax7.plot([], [], 'b-', linewidth=2)
    point7, = ax7.plot([], [], 'ro', markersize=10)
    ax7.plot(xyz[0, 0], xyz[0, 1], 'go', markersize=10, label='Start')
    ax7.set_xlim(xyz[:, 0].min() - 1, xyz[:, 0].max() + 1)
    ax7.set_ylim(xyz[:, 1].min() - 1, xyz[:, 1].max() + 1)
    ax7.set_xlabel('X Position (m)')
    ax7.set_ylabel('Y Position (m)')
    ax7.set_title('XY Trajectory (Top View)')
    ax7.axis('equal')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Actions
    line8a, = ax8.plot([], [], 'b-', linewidth=2, label='Throttle')
    line8b, = ax8.plot([], [], 'r-', linewidth=2, label='Steering')
    point8a, = ax8.plot([], [], 'bo', markersize=8)
    point8b, = ax8.plot([], [], 'ro', markersize=8)
    ax8.set_xlim(0, time[-1])
    ax8.set_ylim(-1.1, 1.1)
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Action Value [-1, 1]')
    ax8.set_title('Actions')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Elevation map
    im = ax9.imshow(elevation_maps[0], cmap='terrain', origin='lower',
                    extent=[-1.25, 1.25, -1.25, 1.25], animated=True)
    robot_marker, = ax9.plot(0, 0, 'r+', markersize=20, markeredgewidth=3)
    ax9.set_xlabel('Y (m)')
    ax9.set_ylabel('X (m)')
    title_text = ax9.text(0.5, 1.05, '', transform=ax9.transAxes, ha='center')
    plt.colorbar(im, ax=ax9, label='Height (m)')
    
    # Info text
    info_text = ax_info.text(0.1, 0.9, '', transform=ax_info.transAxes, 
                             verticalalignment='top', fontsize=10, family='monospace')
    
    def rotation_matrix_from_euler(roll, pitch, yaw):
        """Convert Euler angles to rotation matrix."""
        # Roll (X), Pitch (Y), Yaw (Z)
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # ZYX convention (yaw-pitch-roll)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R
    
    def init():
        line_3d.set_data([], [])
        line_3d.set_3d_properties([])
        point_3d.set_data([], [])
        point_3d.set_3d_properties([])
        
        line1.set_data([], [])
        point1.set_data([], [])
        line2.set_data([], [])
        point2.set_data([], [])
        line3.set_data([], [])
        point3.set_data([], [])
        line4.set_data([], [])
        point4.set_data([], [])
        line5.set_data([], [])
        point5.set_data([], [])
        line6.set_data([], [])
        point6.set_data([], [])
        line7.set_data([], [])
        point7.set_data([], [])
        line8a.set_data([], [])
        line8b.set_data([], [])
        point8a.set_data([], [])
        point8b.set_data([], [])
        
        return (line_3d, point_3d, line1, point1, line2, point2, line3, point3, 
                line4, point4, line5, point5, line6, point6, line7, point7, 
                line8a, line8b, point8a, point8b, im, title_text, info_text)
    
    def animate(frame):
        nonlocal arrow_x, arrow_y, arrow_z
        
        # Update 3D trajectory
        line_3d.set_data(xyz[:frame+1, 0], xyz[:frame+1, 1])
        line_3d.set_3d_properties(xyz[:frame+1, 2])
        point_3d.set_data([xyz[frame, 0]], [xyz[frame, 1]])
        point_3d.set_3d_properties([xyz[frame, 2]])
        
        # Remove old orientation arrows
        if arrow_x is not None:
            arrow_x.remove()
            arrow_y.remove()
            arrow_z.remove()
        
        # Draw orientation axes at current position
        pos = xyz[frame]
        roll, pitch, yaw = rpy[frame]
        R = rotation_matrix_from_euler(roll, pitch, yaw)
        
        # Axis directions in world frame
        arrow_length = 0.5
        x_dir = R @ np.array([1, 0, 0]) * arrow_length  # Forward (red)
        y_dir = R @ np.array([0, 1, 0]) * arrow_length  # Left (green)
        z_dir = R @ np.array([0, 0, 1]) * arrow_length  # Up (blue)
        
        arrow_x = ax_3d.quiver(pos[0], pos[1], pos[2], 
                               x_dir[0], x_dir[1], x_dir[2], 
                               color='red', arrow_length_ratio=0.3, linewidth=2)
        arrow_y = ax_3d.quiver(pos[0], pos[1], pos[2], 
                               y_dir[0], y_dir[1], y_dir[2], 
                               color='green', arrow_length_ratio=0.3, linewidth=2)
        arrow_z = ax_3d.quiver(pos[0], pos[1], pos[2], 
                               z_dir[0], z_dir[1], z_dir[2], 
                               color='blue', arrow_length_ratio=0.3, linewidth=2)
        
        # Update time series plots
        line1.set_data(time[:frame+1], xyz[:frame+1, 0])
        point1.set_data([time[frame]], [xyz[frame, 0]])
        
        line2.set_data(time[:frame+1], xyz[:frame+1, 1])
        point2.set_data([time[frame]], [xyz[frame, 1]])
        
        line3.set_data(time[:frame+1], xyz[:frame+1, 2])
        point3.set_data([time[frame]], [xyz[frame, 2]])
        
        line4.set_data(time[:frame+1], np.rad2deg(rpy[:frame+1, 0]))
        point4.set_data([time[frame]], [np.rad2deg(rpy[frame, 0])])
        
        line5.set_data(time[:frame+1], np.rad2deg(rpy[:frame+1, 1]))
        point5.set_data([time[frame]], [np.rad2deg(rpy[frame, 1])])
        
        line6.set_data(time[:frame+1], np.rad2deg(rpy[:frame+1, 2]))
        point6.set_data([time[frame]], [np.rad2deg(rpy[frame, 2])])
        
        line7.set_data(xyz[:frame+1, 0], xyz[:frame+1, 1])
        point7.set_data([xyz[frame, 0]], [xyz[frame, 1]])
        
        line8a.set_data(time[:frame+1], actions[:frame+1, 0])
        line8b.set_data(time[:frame+1], actions[:frame+1, 1])
        point8a.set_data([time[frame]], [actions[frame, 0]])
        point8b.set_data([time[frame]], [actions[frame, 1]])
        
        # Update elevation map
        im.set_array(elevation_maps[frame])
        title_text.set_text(f'Elevation Map (t={time[frame]:.1f}s)')
        
        # Update info text
        info_str = f"Time: {time[frame]:.2f} s\n\n"
        info_str += f"Position (XYZ):\n"
        info_str += f"  X: {xyz[frame, 0]:6.2f} m\n"
        info_str += f"  Y: {xyz[frame, 1]:6.2f} m\n"
        info_str += f"  Z: {xyz[frame, 2]:6.2f} m\n\n"
        info_str += f"Orientation (RPY):\n"
        info_str += f"  Roll:  {np.rad2deg(rpy[frame, 0]):6.1f}°\n"
        info_str += f"  Pitch: {np.rad2deg(rpy[frame, 1]):6.1f}°\n"
        info_str += f"  Yaw:   {np.rad2deg(rpy[frame, 2]):6.1f}°\n\n"
        info_str += f"Actions:\n"
        info_str += f"  Throttle: {actions[frame, 0]:5.2f}\n"
        info_str += f"  Steering: {actions[frame, 1]:5.2f}\n\n"
        info_str += f"Frame: {frame+1}/{len(time)}"
        info_text.set_text(info_str)
        
        return (line_3d, point_3d, line1, point1, line2, point2, line3, point3,
                line4, point4, line5, point5, line6, point6, line7, point7,
                line8a, line8b, point8a, point8b, im, title_text, info_text)
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(time),
                        interval=1000/fps, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize H5 File Data")
    parser.add_argument('--file_path', type=str, required=True, 
                       help='Path to the H5 file to visualize')
    parser.add_argument('--episode_id', type=int, default=None,
                       help='Episode ID to visualize (default: first episode)')
    parser.add_argument('--env_id', type=int, default=None,
                       help='Environment ID to visualize (default: first environment)')
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'animation'],
                       help='Visualization mode: static plots or animation')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for animation mode (default: 10)')
    args = parser.parse_args()
    main(args)  