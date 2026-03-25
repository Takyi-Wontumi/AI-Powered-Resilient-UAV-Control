"""Train a PPO policy for takeoff + hover (Crazyflie / PyBullet).

Design:
- RL outputs high-level references: [x_ref, y_ref, z_ref, yaw_ref]
- A PID controller tracks those references and generates low-level commands
- Episode objective: takeoff from ground and reach stable hover altitude
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from AI_UAV_Tests.quadcopter_env import QuadcopterPID
from phoenix_drone_simulation.algs import core
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.loggers import setup_logger_kwargs


def _safe_close_env(env):
    try:
        env.close()
    except Exception:
        pass
    try:
        if hasattr(env, "bc"):
            env.bc.disconnect()
    except Exception:
        pass


class TakeoffHoverPIDEnv(DroneMissionEnv):
    """Ground takeoff + hover env with high-level RL actions and PID inner loop."""

    def __init__(
        self,
        target_z: float = 1.0,
        max_episode_s: float = 8.0,
        success_hold_s: float = 2.0,
        max_xy_ref: float = 0.30,
        max_z_ref_delta: float = 0.25,
        max_yaw_ref: float = np.pi / 4.0,
        hover_xy_tol: float = 0.15,
        hover_z_tol: float = 0.08,
        hover_vel_tol: float = 0.35,
        hover_tilt_tol_deg: float = 15.0,
        debug_interval: int = 25,
        **kwargs,
    ):
        self.target_z = float(target_z)
        self.max_episode_s = float(max_episode_s)
        self.success_hold_s = float(success_hold_s)
        self.max_xy_ref = float(max_xy_ref)
        self.max_z_ref_delta = float(max_z_ref_delta)
        self.max_yaw_ref = float(max_yaw_ref)
        self.hover_xy_tol = float(hover_xy_tol)
        self.hover_z_tol = float(hover_z_tol)
        self.hover_vel_tol = float(hover_vel_tol)
        self.hover_tilt_tol = np.deg2rad(float(hover_tilt_tol_deg))
        self.debug_interval = int(debug_interval)

        self.done_reason = "none"
        self.step_count = 0
        self.stable_hover_steps = 0
        self.required_hover_steps = 1
        self.prev_z = 0.0
        self.last_reward = 0.0
        self.last_u1 = 0.0
        self.current_target = np.array([0.0, 0.0, self.target_z], dtype=np.float32)
        # Give motors enough spin-up time before declaring failed takeoff.
        self.min_steps_before_ground_fail = 160

        # Keep a simple one-step observation, no stacked history.
        kwargs.setdefault("observation_history_size", 1)

        super().__init__(**kwargs)
        self.required_hover_steps = max(1, int(self.success_hold_s / self.TIME_STEP))

        # PID stabilization loop (inner control).
        self.pid = QuadcopterPID(dt=self.TIME_STEP)
        self.pid.reset()
        self.hover_thrust = self.pid.m * self.pid.g

        # RL action = [x_ref, y_ref, z_ref, yaw_ref] in normalized range [-1, 1].
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation = [pos(3), vel(3), ang_vel(3), quat(4), pos_err(3)] = 16
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(16,),
            dtype=np.float32,
        )

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.mission_time = 0.0
        self.step_count = 0
        self.done_reason = "none"
        self.stable_hover_steps = 0
        self.pid.reset()
        self.current_target = np.array([0.0, 0.0, self.target_z], dtype=np.float32)
        self.set_target(self.current_target)

        self.prev_z = float(self.drone.xyz[2])
        self.last_u1 = self.hover_thrust
        self.last_reward = 0.0

        obs = self.compute_observation()
        info = self.compute_info()
        return obs, info

    def _map_high_level_action(self, action: np.ndarray) -> Tuple[np.ndarray, float]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        x_ref = float(a[0] * self.max_xy_ref)
        y_ref = float(a[1] * self.max_xy_ref)
        z_ref = float(np.clip(self.target_z + a[2] * self.max_z_ref_delta, 0.05, 1.30))
        yaw_ref = float(a[3] * self.max_yaw_ref)

        pos_ref = np.array([x_ref, y_ref, z_ref], dtype=np.float32)
        return pos_ref, yaw_ref

    def _apply_hover_bias(self, u1_raw: float) -> float:
        """Apply hover-centered bias so thrust cannot collapse to zero."""
        delta = np.clip(
            u1_raw - self.hover_thrust,
            -0.35 * self.hover_thrust,
            0.35 * self.hover_thrust,
        )
        thrust = self.hover_thrust + delta
        # Hard floor above zero-thrust region.
        thrust = max(0.75 * self.hover_thrust, thrust)
        return float(thrust)

    def _thrust_to_action(self, thrust: float) -> float:
        """Map physical thrust [N] to AttitudeRate action[0] in [-1, 1]."""
        # Hover-centered normalization used in previous mission-control scripts.
        return float(np.clip((thrust / self.hover_thrust - 0.9) / 0.4, -1.0, 1.0))

    def _pid_to_low_level_action(self, pos_ref: np.ndarray, yaw_ref: float) -> np.ndarray:
        x = self.drone.xyz
        v = self.drone.xyz_dot
        ang = self.drone.rpy
        rate = self.drone.rpy_dot

        self.pid.inject_external_state(x, v, ang, rate)
        ctrl = self.pid.step(pos_ref=pos_ref, vel_ref=np.zeros(3), z_ref=float(pos_ref[2]))

        rates_des = np.array(ctrl["rates_des"], dtype=np.float32)

        # Yaw tracking term on top of PID desired rates.
        yaw_err = float(np.arctan2(np.sin(yaw_ref - ang[2]), np.cos(yaw_ref - ang[2])))
        rates_des[2] = np.clip(rates_des[2] + 1.2 * yaw_err, -self.pid.max_rate, self.pid.max_rate)

        u1_raw = float(ctrl["thrust_cmd"])
        u1 = self._apply_hover_bias(u1_raw)
        self.last_u1 = u1

        low = np.zeros(4, dtype=np.float32)
        low[0] = self._thrust_to_action(u1)
        low[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)
        return low

    def compute_observation(self) -> np.ndarray:
        # Required fields: pos, vel, angular velocity, quat, position error.
        pos = np.asarray(self.drone.xyz, dtype=np.float32)
        vel = np.asarray(self.drone.xyz_dot, dtype=np.float32)
        ang_vel = np.asarray(self.drone.rpy_dot, dtype=np.float32)
        quat = np.asarray(self.drone.quaternion, dtype=np.float32)
        pos_err = np.asarray(self.current_target - pos, dtype=np.float32)

        # Normalize to roughly [-1, 1].
        pos_n = np.clip(pos / 2.0, -1.0, 1.0)
        vel_n = np.clip(vel / 2.0, -1.0, 1.0)
        ang_vel_n = np.clip(ang_vel / 6.0, -1.0, 1.0)
        err_n = np.clip(pos_err / 2.0, -1.0, 1.0)

        obs = np.concatenate([pos_n, vel_n, ang_vel_n, quat, err_n]).astype(np.float32)
        return obs

    def compute_reward(self, action) -> float:
        x, y, z = map(float, self.drone.xyz)
        action_mag = float(np.linalg.norm(np.clip(action, -1.0, 1.0)))
        v_norm = float(np.linalg.norm(self.drone.xyz_dot))
        tilt = float(np.linalg.norm(self.drone.rpy[:2]))
        z_err = abs(z - self.target_z)
        xy_err = float(np.sqrt(x * x + y * y))

        reward = 0.0

        # reward upward movement
        reward += 5.0 * (z - self.prev_z)

        # reward being near target altitude
        reward -= 2.0 * abs(z - self.target_z)

        # penalize lateral drift slightly
        reward -= 0.5 * float(np.sqrt(x * x + y * y))

        # penalize crash / no lift
        if z < 0.05 and self.step_count > self.min_steps_before_ground_fail:
            reward -= 50.0

        # small action penalty
        reward -= 0.001 * action_mag

        # bonus for being in stable hover envelope
        in_hover = (
            xy_err < self.hover_xy_tol
            and z_err < self.hover_z_tol
            and v_norm < self.hover_vel_tol
            and tilt < self.hover_tilt_tol
        )
        if in_hover:
            reward += 0.5

        self.prev_z = z
        self.last_reward = float(reward)
        return float(reward)

    def compute_done(self) -> bool:
        z = float(self.drone.xyz[2])
        xy_err = float(np.linalg.norm(self.drone.xyz[:2]))
        z_err = abs(z - self.target_z)
        v_norm = float(np.linalg.norm(self.drone.xyz_dot))
        tilt = float(np.linalg.norm(self.drone.rpy[:2]))

        # Early failure: still on/near ground after grace period.
        if self.step_count > self.min_steps_before_ground_fail and z < 0.05:
            self.done_reason = "failed_takeoff"
            return True

        in_hover = (
            xy_err < self.hover_xy_tol
            and z_err < self.hover_z_tol
            and v_norm < self.hover_vel_tol
            and tilt < self.hover_tilt_tol
        )
        self.stable_hover_steps = self.stable_hover_steps + 1 if in_hover else 0

        # Success only after sustained stable hover.
        if self.stable_hover_steps >= self.required_hover_steps:
            self.done_reason = "success"
            return True

        # Safety guard.
        if np.linalg.norm(self.drone.rpy[:2]) > np.deg2rad(65.0):
            self.done_reason = "tilt_limit"
            return True

        # Timeout.
        if self.mission_time >= self.max_episode_s:
            self.done_reason = "timeout"
            return True

        self.done_reason = "none"
        return False

    def compute_info(self, action=None) -> Dict:
        info = {
            "cost": 0.0,
            "done_reason": self.done_reason,
            "thrust": float(self.last_u1),
            "z": float(self.drone.xyz[2]),
            "reward_step": float(self.last_reward),
            "action_mag": float(np.linalg.norm(np.clip(action, -1.0, 1.0))) if action is not None else 0.0,
        }
        return info

    def step(self, action: np.ndarray) -> tuple:
        self.step_count += 1
        self.mission_time += self.TIME_STEP

        pos_ref, yaw_ref = self._map_high_level_action(action)
        self.current_target = pos_ref.astype(np.float32)
        self.set_target(self.current_target)

        low_action = self._pid_to_low_level_action(pos_ref, yaw_ref)

        for _ in range(self.aggregate_phy_steps):
            self.physics.step_forward(low_action)
            self.compute_observation()
            self.iteration += 1

        obs = self.compute_observation()
        terminated = self.compute_done()
        reward = self.compute_reward(action)
        info = self.compute_info(action)
        truncated = False

        if self.debug_interval > 0 and (self.step_count % self.debug_interval == 0 or terminated):
            print(
                "[DBG] "
                f"step={self.step_count:4d} "
                f"z={info['z']:.3f} "
                f"thrust={info['thrust']:.3f} "
                f"reward={info['reward_step']:.3f} "
                f"|a|={info['action_mag']:.3f} "
                f"done={terminated} reason={info['done_reason']}",
                flush=True,
            )

        self.last_action = low_action
        return obs, reward, terminated, truncated, info


def get_alg_defaults(alg: str) -> dict:
    defaults_mod = utils.get_alg_module(alg, "defaults")
    for fn_name in ("defaults", "locomotion", "gym_locomotion_envs"):
        if hasattr(defaults_mod, fn_name):
            return getattr(defaults_mod, fn_name)()
    return {}


def load_actor_critic_from_ckpt(ckpt_dir: str, env):
    config_path = os.path.join(ckpt_dir, "config.json")
    model_path = os.path.join(ckpt_dir, "torch_save", "model.pt")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json at: {config_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Missing model.pt at: {model_path}")

    conf = utils.get_file_contents(config_path)
    ac = core.ActorCritic(
        actor_type=conf["actor"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_standardized_obs=conf.get("use_standardized_obs", True),
        use_scaled_rewards=conf.get("use_reward_scaling", True),
        use_shared_weights=False,
        ac_kwargs=conf["ac_kwargs"],
    )
    ac.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    ac.eval()
    return ac


def build_env(args, render_mode=None):
    if args.control_mode != "AttitudeRate":
        raise ValueError(
            "train_takeoff_hover.py currently supports only "
            "--control-mode AttitudeRate for the high-level RL + PID setup."
        )
    return TakeoffHoverPIDEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode=render_mode,
        observation_noise=args.observation_noise,
        domain_randomization=args.domain_randomization,
        motor_thrust_noise=args.motor_thrust_noise,
        target_z=args.target_z,
        max_episode_s=args.max_episode_s,
        success_hold_s=args.success_hold_s,
        max_xy_ref=args.max_xy_ref,
        max_z_ref_delta=args.max_z_ref_delta,
        max_yaw_ref=args.max_yaw_ref_deg * np.pi / 180.0,
        hover_xy_tol=args.hover_xy_tol,
        hover_z_tol=args.hover_z_tol,
        hover_vel_tol=args.hover_vel_tol,
        hover_tilt_tol_deg=args.hover_tilt_tol_deg,
        debug_interval=args.debug_interval,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--alg", type=str, default="ppo", choices=["ppo", "iwpg", "trpo", "npg"])
    # Compatibility flag: keep accepting previous commands.
    parser.add_argument(
        "--control-mode",
        type=str,
        default="AttitudeRate",
        choices=["AttitudeRate", "Attitude", "PWM"],
        help="Compatibility flag. High-level RL + PID currently runs with AttitudeRate.",
    )

    parser.add_argument("--target-z", type=float, default=1.0)
    parser.add_argument("--max-episode-s", type=float, default=6.0)
    parser.add_argument("--success-hold-s", type=float, default=2.0)
    parser.add_argument("--max-xy-ref", type=float, default=0.30)
    parser.add_argument("--max-z-ref-delta", type=float, default=0.25)
    parser.add_argument("--max-yaw-ref-deg", type=float, default=45.0)
    parser.add_argument("--hover-xy-tol", type=float, default=0.15)
    parser.add_argument("--hover-z-tol", type=float, default=0.08)
    parser.add_argument("--hover-vel-tol", type=float, default=0.35)
    parser.add_argument("--hover-tilt-tol-deg", type=float, default=15.0)

    parser.add_argument("--observation-noise", type=float, default=1.0)
    parser.add_argument("--domain-randomization", type=float, default=0.01)
    parser.add_argument("--motor-thrust-noise", type=float, default=0.03)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pi-hidden-sizes", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--val-hidden-sizes", nargs="+", type=int, default=[128, 128])

    parser.add_argument("--until-stable", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--target-success-rate", type=float, default=0.7)
    parser.add_argument("--eval-episodes", type=int, default=10)

    parser.add_argument("--render-training", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-episodes", type=int, default=3)
    parser.add_argument("--debug-interval", type=int, default=25)
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint run directory (contains config.json and torch_save/model.pt). If provided, skip training and play this model.",
    )

    parser.add_argument("--log-dir", type=str, default=os.path.join(ROOT_DIR, "runs"))
    return parser.parse_args()


def evaluate_policy(ac, args, episodes: int, render: bool = False):
    env = build_env(args, render_mode="human" if render else None)
    ac.eval()

    successes = 0
    reasons = {}
    returns = []

    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            last_info = {}

            while not done:
                with torch.no_grad():
                    action, _, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_ret += reward
                last_info = info
                if render:
                    time.sleep(1.0 / 120.0)

            reason = last_info.get("done_reason", "unknown")
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "success":
                successes += 1
            returns.append(ep_ret)
    finally:
        _safe_close_env(env)

    success_rate = successes / max(1, episodes)
    mean_ret = float(np.mean(returns)) if returns else 0.0
    return success_rate, mean_ret, reasons


def train_one_round(args, round_idx: int):
    seed = int(args.seed) + round_idx - 1
    exp_name = f"TakeoffHoverPID_round_{round_idx}"
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, base_dir=args.log_dir, seed=seed)

    env = build_env(args, render_mode="human" if args.render_training else None)

    max_ep_len = max(1, int(args.max_episode_s / env.TIME_STEP))
    steps_per_epoch = max(int(args.steps_per_epoch), max_ep_len)
    if steps_per_epoch != args.steps_per_epoch:
        print(
            f"[info] steps-per-epoch increased from {args.steps_per_epoch} "
            f"to {steps_per_epoch} (must be >= max_ep_len={max_ep_len})."
        )

    defaults = get_alg_defaults(args.alg)
    defaults.update(
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        seed=seed,
        max_ep_len=max_ep_len,
        logger_kwargs=logger_kwargs,
        use_entropy=True,
        entropy_coef=0.05,
        use_standardized_advantages=True,
        use_exploration_noise_anneal=False,
    )

    if "ac_kwargs" in defaults:
        defaults["ac_kwargs"]["pi"]["hidden_sizes"] = tuple(args.pi_hidden_sizes)
        defaults["ac_kwargs"]["val"]["hidden_sizes"] = tuple(args.val_hidden_sizes)

    try:
        alg = utils.get_alg_class(args.alg, env_id=env, **defaults)
        ac, _ = alg.learn()
    finally:
        _safe_close_env(env)

    return ac, logger_kwargs["log_dir"]


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    if args.ckpt:
        # Build env once to recover spaces for ActorCritic construction.
        env_for_load = build_env(args, render_mode=None)
        try:
            ac = load_actor_critic_from_ckpt(args.ckpt, env_for_load)
        finally:
            _safe_close_env(env_for_load)

        success_rate, mean_ret, reasons = evaluate_policy(
            ac,
            args,
            episodes=args.play_episodes,
            render=True,
        )
        print(
            f"Playback from ckpt complete: success_rate={success_rate:.2%}, "
            f"mean_return={mean_ret:.2f}, reasons={reasons}"
        )
        return

    max_rounds = args.max_rounds if args.until_stable else 1
    best = {"success_rate": -1.0, "round": -1, "log_dir": None, "ac": None}

    for round_idx in range(1, max_rounds + 1):
        print("=" * 80)
        print(f"Training round {round_idx}/{max_rounds}")
        print("=" * 80)

        ac, log_dir = train_one_round(args, round_idx=round_idx)
        success_rate, mean_ret, reasons = evaluate_policy(ac, args, episodes=args.eval_episodes, render=False)

        print(f"Round {round_idx} eval: success_rate={success_rate:.2%}, mean_return={mean_ret:.2f}, reasons={reasons}")
        print(f"Round {round_idx} logs: {log_dir}")

        if success_rate > best["success_rate"]:
            best.update({"success_rate": success_rate, "round": round_idx, "log_dir": log_dir, "ac": ac})

        if success_rate >= args.target_success_rate:
            print(f"Reached target success rate ({args.target_success_rate:.0%}) at round {round_idx}.")
            break

    print("-" * 80)
    print(f"Best round: {best['round']} | success_rate={best['success_rate']:.2%} | logs={best['log_dir']}")

    if args.play and best["ac"] is not None:
        print("Playing best policy...")
        evaluate_policy(best["ac"], args, episodes=args.play_episodes, render=True)


if __name__ == "__main__":
    main()
