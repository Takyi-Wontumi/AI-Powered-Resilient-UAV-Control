"""Record simulation rollouts and save them as GIF files.

Supports:
1) Random policy rollout from any Gymnasium env id
2) Checkpoint rollout from examples/train_mission_trajectory.py
"""

import argparse
import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import phoenix_drone_simulation  # noqa: F401 - needed for env registration side-effects
from phoenix_drone_simulation.envs.register_envs import register_all_envs


def _build_mission_ckpt_env_and_policy(args):
    from examples.training.train_mission_trajectory import (
        build_env_from_ckpt,
        load_actor_critic_from_ckpt,
    )

    ckpt_args = SimpleNamespace(
        ckpt=args.ckpt,
        traj=args.traj,
        control_mode=args.control_mode,
        observation_noise=1.0,
        domain_randomization=0.05,
        motor_thrust_noise=0.05,
        noise_start=0.0,
        noise_ramp_episodes=50,
        dr_start=0.0,
        dr_ramp_episodes=50,
    )
    env, _, _ = build_env_from_ckpt(ckpt_args)
    actor_critic, _ = load_actor_critic_from_ckpt(args.ckpt, env)
    actor_critic.eval()
    return env, actor_critic


def _build_random_env(args):
    render_mode = None
    env = gym.make(args.env, render_mode=render_mode)
    return env


def _resolve_target_position(env):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(base_env, "drone") and hasattr(base_env.drone, "xyz"):
        drone_xyz = np.asarray(base_env.drone.xyz, dtype=np.float32)
        return drone_xyz.tolist()
    return [0.0, 0.0, 0.75]


def _capture_frame(env, args):
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    if not hasattr(base_env, "bc"):
        raise RuntimeError("Environment does not expose PyBullet client as env.bc.")

    target = _resolve_target_position(env)
    view_matrix = base_env.bc.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=args.camera_distance,
        yaw=args.camera_yaw,
        pitch=args.camera_pitch,
        roll=0.0,
        upAxisIndex=2,
    )
    proj_matrix = base_env.bc.computeProjectionMatrixFOV(
        fov=args.fov,
        aspect=float(args.width) / float(args.height),
        nearVal=0.01,
        farVal=100.0,
    )

    renderer = base_env.bc.ER_TINY_RENDERER if args.renderer == "tiny" else base_env.bc.ER_BULLET_HARDWARE_OPENGL
    _, _, rgba, _, _ = base_env.bc.getCameraImage(
        width=args.width,
        height=args.height,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=renderer,
    )
    frame = np.reshape(rgba, (args.height, args.width, 4))[:, :, :3]
    return frame.astype(np.uint8)


def _policy_action(actor_critic, obs):
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    action, _, _ = actor_critic.step(obs_t)
    return action


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Record simulation rollouts and save as GIF.",
    )
    parser.add_argument("--out", type=str, default="docs/readme/sim_preview.gif", help="Output GIF path.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--max-steps", type=int, default=600, help="Max steps per episode.")
    parser.add_argument("--fps", type=int, default=20, help="GIF playback FPS.")
    parser.add_argument("--frame-skip", type=int, default=2, help="Capture every N simulator steps.")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--camera-distance", type=float, default=2.0)
    parser.add_argument("--camera-yaw", type=float, default=45.0)
    parser.add_argument("--camera-pitch", type=float, default=-30.0)
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument("--renderer", choices=["tiny", "opengl"], default="tiny")

    parser.add_argument("--env", type=str, default="DroneHoverBulletEnv-v0", help="Env id for random rollout mode.")

    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint dir from train_mission_trajectory.")
    parser.add_argument("--traj", type=str, default="flight_mission", help="Used if --ckpt is set.")
    parser.add_argument(
        "--control-mode",
        type=str,
        default="PWM",
        choices=["PWM", "Attitude", "AttitudeRate"],
        help="Used if --ckpt is set.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.frame_skip <= 0:
        raise ValueError("--frame-skip must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    register_all_envs()

    actor_critic = None
    if args.ckpt:
        env, actor_critic = _build_mission_ckpt_env_and_policy(args)
    else:
        env = _build_random_env(args)

    frames = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            step = 0
            while not done and step < args.max_steps:
                if step % args.frame_skip == 0:
                    frames.append(_capture_frame(env, args))

                if actor_critic is None:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = _policy_action(actor_critic, obs)

                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                step += 1

            print(f"Episode {ep + 1}: captured {step} steps.")
    finally:
        try:
            env.close()
        except Exception:
            pass

    if not frames:
        raise RuntimeError("No frames were captured. Nothing to save.")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    print(f"Saved GIF: {args.out}")
    print(f"Frames: {len(frames)}, FPS: {args.fps}")


if __name__ == "__main__":
    main()
