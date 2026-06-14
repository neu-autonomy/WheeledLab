#!/usr/bin/env python3
"""
Sanity-check a collected dynamics dataset.

Answers the questions that told us the old data was broken:
  1. Are there NaNs/infs (esp. in the elevation map when airborne)?
  2. What speeds appear?  (Old data had ~12 m/s from the spawn hack -> unreachable.
     New data should stay <= ~3 m/s, the hardware cap.)
  3. Do real jumps happen?  (Fraction of samples airborne + max air height.)
  4. Does steering actually affect the robot?  (Correlation between the steering
     action and the resulting yaw rate. Old data had steering scaled to 0 -> ~0 corr.)
  5. Action distributions (throttle should be forward-only, steering spread).

Usage:
    python scripts/inspect_dynamics_data.py <path/to/run_dir_or_h5>
    python scripts/inspect_dynamics_data.py            # auto-picks newest run under ./data/dynamics
"""

import sys
import glob
import os

import numpy as np
import h5py

# State layout (see docs/DATA_COLLECTION.md). Everything before the last ELEV_SIZE is core state.
IDX = {
    "root_pos_w": (0, 3),       # x, y, z (world)
    "root_quat_w": (3, 7),
    "world_euler_xyz": (7, 10), # roll, pitch, yaw
    "base_lin_vel": (10, 13),   # vx, vy, vz (body)
    "base_ang_vel": (13, 16),   # wx, wy, wz (body)
    "root_lin_vel_w": (16, 19),
    "root_ang_vel_w": (19, 22),
    "last_action": (22, 24),
}
# Elevation map is the LAST 676 values = 26x26 grid (GridPatternCfg size=2.5, res=0.1 ->
# 26 points/axis incl. endpoints). NOT 625/25x25 -- verified against the data (the joint
# region ends at col 44; 720-44=676) and the grid_pattern source. The h5 'elevation_map_size'
# attr is hardcoded to a wrong 625 in older runs, so do not trust it.
GRID = 26
ELEV_SIZE = GRID * GRID   # 676
GROUND_Z = 0.19   # robot base height resting on flat ground


def _find_inputs(arg):
    if arg and arg.endswith(".h5"):
        return [arg]
    root = arg or "./data/dynamics"
    if arg and os.path.isdir(arg) and not glob.glob(os.path.join(arg, "*.h5")):
        # maybe a parent dir of run_* dirs
        runs = sorted(glob.glob(os.path.join(arg, "run_*")))
        root = runs[-1] if runs else arg
    elif not arg:
        runs = sorted(glob.glob(os.path.join(root, "run_*")))
        if not runs:
            sys.exit(f"No run_* dirs under {root}. Pass a run dir or .h5 explicitly.")
        root = runs[-1]
    files = sorted(glob.glob(os.path.join(root, "*.h5")))
    if not files:
        sys.exit(f"No .h5 files found in {root}")
    print(f"Inspecting {len(files)} file(s) in {root}")
    return files


def _load(files):
    S, A, NS, term, trunc = [], [], [], [], []
    for f in files:
        with h5py.File(f, "r") as h:
            S.append(h["states"][:])
            A.append(h["actions"][:])
            NS.append(h["next_states"][:])
            term.append(h["terminated"][:])
            trunc.append(h["truncated"][:])
    return (np.concatenate(S), np.concatenate(A), np.concatenate(NS),
            np.concatenate(term), np.concatenate(trunc))


def col(states, name):
    a, b = IDX[name]
    return states[:, a:b]


def pct(x, ps=(50, 90, 99, 100)):
    return ", ".join(f"p{p}={np.percentile(x, p):.2f}" for p in ps)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    files = _find_inputs(arg)
    states, actions, next_states, terminated, truncated = _load(files)
    n = len(states)
    print(f"\n{'='*70}\n{n:,} transitions | state_dim={states.shape[1]} | action_dim={actions.shape[1]}")
    print(f"terminated={terminated.mean()*100:.1f}%  truncated(timeout)={truncated.mean()*100:.1f}%")
    print('='*70)

    # 1) NaN / inf
    elev = states[:, -ELEV_SIZE:]
    core = states[:, :-ELEV_SIZE]
    print("\n[1] NaN / inf")
    print(f"    core state:    nan={np.isnan(core).any(1).mean()*100:.2f}%  inf={np.isinf(core).any(1).mean()*100:.2f}%")
    print(f"    elevation map: nan={np.isnan(elev).any(1).mean()*100:.2f}%  inf={np.isinf(elev).any(1).mean()*100:.2f}%")

    # 2) Speed
    vx = col(states, "base_lin_vel")[:, 0]
    speed = np.linalg.norm(col(states, "root_lin_vel_w"), axis=1)
    print("\n[2] Speed (m/s)  -- expect <= ~3 if reachable; ~12 means the old spawn hack")
    print(f"    forward vx:    {pct(np.abs(vx))}")
    print(f"    world speed:   {pct(speed)}")

    # 3) Jumps
    z = col(states, "root_pos_w")[:, 2]
    vz = col(states, "root_lin_vel_w")[:, 2]
    vx = col(states, "base_lin_vel")[:, 0]
    air = z - GROUND_Z
    airborne = air > 0.05
    # pitch is stored in [0, 2pi]; wrap to [-pi, pi].
    pitch = col(states, "world_euler_xyz")[:, 1]
    pitch_w = (pitch + np.pi) % (2 * np.pi) - np.pi
    pitched = np.abs(pitch_w) > np.radians(10)
    moving = np.abs(vx) > 0.3
    # Distinguish USEFUL jump dynamics (on ramp / in air, still moving) from crashed
    # robots lying tilted-and-stalled. Lumping them together (the old "jump-involved")
    # massively overcounts: a single flip with rollover-termination off produces a long
    # tail of pitched-but-stuck samples that are NOT useful jump data.
    active_jump = airborne | (pitched & moving)
    tipped_stuck = pitched & ~moving
    print("\n[3] Jumps")
    print(f"    airborne (base >5cm up):           {airborne.mean()*100:5.2f}%")
    print(f"    actively jumping (air or pitched+moving): {active_jump.mean()*100:5.2f}%  <- the useful jump data")
    print(f"    tipped/stuck (pitched + <0.3 m/s): {tipped_stuck.mean()*100:5.2f}%  <- crashed robots, pollutes data")
    print(f"    air height (m):    max={air.max():.2f}, {pct(air[airborne]) if airborne.any() else 'none'}")
    print(f"    upward vz (m/s):   max={vz.max():.2f}")
    if active_jump.mean() < 0.02:
        print("    !! very little active-jump data -- ramp too shallow or robot not reaching it")
    if tipped_stuck.mean() > 0.10:
        print("    !! high tipped/stuck fraction -- ramp too aggressive and/or rollover termination off.")
        print("       Re-enable rollover (~75deg) and/or cap the ramp angle lower to cut the crash tail.")

    # 4) Steering effect: does the steering action change yaw rate?
    steer = actions[:, 1]
    yaw_rate = col(next_states, "base_ang_vel")[:, 2]  # body yaw rate after the step
    # only look at moving samples (steering does nothing at standstill)
    moving = np.abs(vx) > 0.3
    if moving.sum() > 100:
        c = np.corrcoef(steer[moving], yaw_rate[moving])[0, 1]
    else:
        c = float("nan")
    print("\n[4] Steering effect  -- corr(steer, next yaw-rate) on moving samples")
    print(f"    corr = {c:+.3f}   (|corr|~0 => steering is inert, the old scale=0 bug)")
    if abs(c) < 0.05:
        print("    !! steering appears to have no effect -- check throttle_steer.scale")

    # 5) Action distributions
    thr = actions[:, 0]
    print("\n[5] Actions")
    print(f"    throttle: min={thr.min():.2f} max={thr.max():.2f} mean={thr.mean():.2f} "
          f"(forward-only expected: min>=0)")
    print(f"    steering: min={steer.min():.2f} max={steer.max():.2f} std={steer.std():.2f}")
    print()


if __name__ == "__main__":
    main()
