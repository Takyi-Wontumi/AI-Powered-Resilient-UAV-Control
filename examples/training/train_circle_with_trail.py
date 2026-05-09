"""Train a circle-tracking policy with visible 3D trail in PyBullet.

- Uses Trajectories.circle_traj from AI_UAV_Tests.core.trajectories_library
- RL outputs high-level reference corrections around the circle reference
- Inner-loop PID controller handles low-level stabilization
- Optional trail visualization for both training and playback
"""

import argparse
import os
import sys
import time
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from AI_UAV_Tests.core.quadcopter_env import QuadcopterPID
from AI_UAV_Tests.core.trajectories_library import Trajectories as T
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


class CircleTrackingPIDEnv(DroneMissionEnv):
    """Circle mission env with high-level RL + PID + optional trail rendering."""

    def __init__(
        self,
        circle_radius: float = 0.8,
        circle_period: float = 10.0,
        target_z: float = 1.0,
        takeoff_s: float = 2.5,
        max_episode_s: float = 18.0,
        max_xy_ref_delta: float = 0.25,
        max_z_ref_delta: float = 0.20,
        max_yaw_ref: float = np.pi / 4.0,
        success_mean_err: float = 0.25,
        trail: bool = True,
        trail_color: Tuple[float, float, float] = (0.05, 0.7, 1.0),
        trail_width: float = 1.8,
        trail_max_points: int = 4000,
        debug_interval: int = 100,
        **kwargs,
    ):
        self.circle_radius = float(circle_radius)
        self.circle_period = float(circle_period)
        self.target_z = float(target_z)
        self.takeoff_s = float(takeoff_s)
        self.max_episode_s = float(max_episode_s)

        self.max_xy_ref_delta = float(max_xy_ref_delta)
        self.max_z_ref_delta = float(max_z_ref_delta)
        self.max_yaw_ref = float(max_yaw_ref)
        self.success_mean_err = float(success_mean_err)

        self.trail_enabled = bool(trail)
        self.trail_color = tuple(float(c) for c in trail_color)
        self.trail_width = float(trail_width)
        self.trail_max_points = int(trail_max_points)

        self.debug_interval = int(debug_interval)

        self.step_count = 0
        self.done_reason = "none"
        self.prev_z = 0.0
        self.last_reward = 0.0
        self.last_u1 = 0.0

        self.base_pos_ref = np.zeros(3, dtype=np.float32)
        self.base_vel_ref = np.zeros(3, dtype=np.float32)
        self.current_target = np.zeros(3, dtype=np.float32)

        self.error_accum = 0.0
        self.error_count = 0

        self.trail_ids = deque()
        self.prev_trail_pos = None

        # single-step observation (no stacked history)
        kwargs.setdefault("observation_history_size", 1)

        super().__init__(**kwargs)

        self.pid = QuadcopterPID(dt=self.TIME_STEP)
        self.pid.reset()
        self.hover_thrust = self.pid.m * self.pid.g

        # RL action is correction around circle reference: [dx, dy, dz, yaw]
        self.action_space = __import__("gymnasium").spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # obs = pos3 + vel3 + angvel3 + quat4 + pos_err3 + vel_err3 = 19
        self.observation_space = __import__("gymnasium").spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(19,),
            dtype=np.float32,
        )

    def _clear_trail(self):
        while self.trail_ids:
            line_id = self.trail_ids.popleft()
            try:
                self.bc.removeUserDebugItem(line_id)
            except Exception:
                pass
        self.prev_trail_pos = None

    def _update_trail(self):
        if not (self.trail_enabled and self.use_graphics):
            return
        curr = np.asarray(self.drone.xyz, dtype=np.float32)
        if self.prev_trail_pos is None:
            self.prev_trail_pos = curr.copy()
            return

        line_id = self.bc.addUserDebugLine(
            self.prev_trail_pos.tolist(),
            curr.tolist(),
            lineColorRGB=self.trail_color,
            lineWidth=self.trail_width,
            lifeTime=0,
        )
        self.trail_ids.append(line_id)
        if len(self.trail_ids) > self.trail_max_points:
            old = self.trail_ids.popleft()
            try:
                self.bc.removeUserDebugItem(old)
            except Exception:
                pass
        self.prev_trail_pos = curr.copy()

    def _base_reference(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        # Use the trajectory library circle directly after takeoff window.
        if t < self.takeoff_s:
            # climb and stage to first circle point
            pos_ref = np.array([self.circle_radius, 0.0, self.target_z], dtype=np.float32)
            vel_ref = np.zeros(3, dtype=np.float32)
            return pos_ref, vel_ref
        pos_ref, vel_ref = T.circle_traj(
            t - self.takeoff_s,
            radius=self.circle_radius,
            z=self.target_z,
            period=self.circle_period,
        )
        return np.asarray(pos_ref, dtype=np.float32), np.asarray(vel_ref, dtype=np.float32)

    def _map_action(self, action: np.ndarray, base_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        pos_ref = np.array(
            [
                base_pos[0] + a[0] * self.max_xy_ref_delta,
                base_pos[1] + a[1] * self.max_xy_ref_delta,
                np.clip(base_pos[2] + a[2] * self.max_z_ref_delta, 0.10, 1.50),
            ],
            dtype=np.float32,
        )
        yaw_ref = float(a[3] * self.max_yaw_ref)
        return pos_ref, yaw_ref

    def _thrust_to_action(self, thrust: float) -> float:
        return float(np.clip((thrust / self.hover_thrust - 0.9) / 0.4, -1.0, 1.0))

    def _apply_hover_bias(self, u1_raw: float) -> float:
        delta = np.clip(
            u1_raw - self.hover_thrust,
            -0.35 * self.hover_thrust,
            0.35 * self.hover_thrust,
        )
        thrust = self.hover_thrust + delta
        thrust = max(0.75 * self.hover_thrust, thrust)
        return float(thrust)

    def _pid_low_level_action(self, pos_ref: np.ndarray, yaw_ref: float) -> np.ndarray:
        x = self.drone.xyz
        v = self.drone.xyz_dot
        ang = self.drone.rpy
        rate = self.drone.rpy_dot

        self.pid.inject_external_state(x, v, ang, rate)
        ctrl = self.pid.step(pos_ref=pos_ref, vel_ref=self.base_vel_ref, z_ref=float(pos_ref[2]))
        rates_des = np.array(ctrl["rates_des"], dtype=np.float32)

        yaw_err = float(np.arctan2(np.sin(yaw_ref - ang[2]), np.cos(yaw_ref - ang[2])))
        rates_des[2] = np.clip(rates_des[2] + 1.0 * yaw_err, -self.pid.max_rate, self.pid.max_rate)

        u1 = self._apply_hover_bias(float(ctrl["thrust_cmd"]))
        self.last_u1 = u1

        low = np.zeros(4, dtype=np.float32)
        low[0] = self._thrust_to_action(u1)
        low[1:4] = np.clip(rates_des / (np.pi / 3.0), -1.0, 1.0)
        return low

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self.mission_time = 0.0
        self.step_count = 0
        self.done_reason = "none"
        self.prev_z = float(self.drone.xyz[2])
        self.last_reward = 0.0
        self.last_u1 = self.hover_thrust
        self.error_accum = 0.0
        self.error_count = 0

        self.pid.reset()

        self.base_pos_ref, self.base_vel_ref = self._base_reference(0.0)
        self.current_target = self.base_pos_ref.copy()
        self.set_target(self.current_target)

        self._clear_trail()
        self._update_trail()

        obs = self.compute_observation()
        info = self.compute_info()
        return obs, info

    def compute_observation(self) -> np.ndarray:
        pos = np.asarray(self.drone.xyz, dtype=np.float32)
        vel = np.asarray(self.drone.xyz_dot, dtype=np.float32)
        ang_vel = np.asarray(self.drone.rpy_dot, dtype=np.float32)
        quat = np.asarray(self.drone.quaternion, dtype=np.float32)

        pos_err = self.base_pos_ref - pos
        vel_err = self.base_vel_ref - vel

        # normalize/clamp
        pos_n = np.clip(pos / 2.0, -1.0, 1.0)
        vel_n = np.clip(vel / 2.0, -1.0, 1.0)
        ang_n = np.clip(ang_vel / 6.0, -1.0, 1.0)
        pos_err_n = np.clip(pos_err / 2.0, -1.0, 1.0)
        vel_err_n = np.clip(vel_err / 2.0, -1.0, 1.0)

        return np.concatenate([pos_n, vel_n, ang_n, quat, pos_err_n, vel_err_n]).astype(np.float32)

    def compute_reward(self, action) -> float:
        pos = np.asarray(self.drone.xyz, dtype=np.float32)
        vel = np.asarray(self.drone.xyz_dot, dtype=np.float32)
        base_err = float(np.linalg.norm(pos - self.base_pos_ref))
        vel_err = float(np.linalg.norm(vel - self.base_vel_ref))
        action_mag = float(np.linalg.norm(np.clip(action, -1.0, 1.0)))

        reward = 0.0

        # tracking objective
        reward -= 2.0 * base_err
        reward -= 0.2 * vel_err

        # encourage takeoff progress in staging window
        z = float(pos[2])
        if self.mission_time < self.takeoff_s:
            reward += 3.0 * (z - self.prev_z)
            reward -= 1.0 * abs(z - self.target_z)

        # regularization
        reward -= 0.001 * action_mag

        # shaping near path
        if base_err < 0.20:
            reward += 0.4

        # penalties
        if z < 0.05 and self.step_count > int(0.8 / self.TIME_STEP):
            reward -= 25.0

        self.prev_z = z
        self.last_reward = float(reward)
        return float(reward)

    def compute_done(self) -> bool:
        pos = np.asarray(self.drone.xyz, dtype=np.float32)
        z = float(pos[2])
        xy = float(np.linalg.norm(pos[:2]))

        base_err = float(np.linalg.norm(pos - self.base_pos_ref))
        self.error_accum += base_err
        self.error_count += 1

        # hard failures
        if self.step_count > int(0.8 / self.TIME_STEP) and z < 0.05:
            self.done_reason = "failed_takeoff"
            return True
        if xy > 2.5 or z > 2.2:
            self.done_reason = "out_of_bounds"
            return True
        if np.linalg.norm(self.drone.rpy[:2]) > np.deg2rad(65.0):
            self.done_reason = "tilt_limit"
            return True

        # mission end
        if self.mission_time >= self.max_episode_s:
            mean_err = self.error_accum / max(1, self.error_count)
            if mean_err <= self.success_mean_err:
                self.done_reason = "success"
            else:
                self.done_reason = "timeout"
            return True

        self.done_reason = "none"
        return False

    def compute_info(self, action=None) -> Dict:
        pos = np.asarray(self.drone.xyz, dtype=np.float32)
        base_err = float(np.linalg.norm(pos - self.base_pos_ref))
        info = {
            "cost": 0.0,
            "done_reason": self.done_reason,
            "base_error": base_err,
            "thrust": float(self.last_u1),
            "z": float(pos[2]),
            "reward_step": float(self.last_reward),
            "action_mag": float(np.linalg.norm(np.clip(action, -1.0, 1.0))) if action is not None else 0.0,
        }
        return info

    def step(self, action: np.ndarray) -> tuple:
        self.step_count += 1
        self.mission_time += self.TIME_STEP

        self.base_pos_ref, self.base_vel_ref = self._base_reference(self.mission_time)
        pos_ref, yaw_ref = self._map_action(action, self.base_pos_ref)
        self.current_target = pos_ref.copy()
        self.set_target(self.current_target)

        low_action = self._pid_low_level_action(pos_ref, yaw_ref)

        for _ in range(self.aggregate_phy_steps):
            self.physics.step_forward(low_action)
            self.compute_observation()
            self.iteration += 1

        self._update_trail()

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
                f"err={info['base_error']:.3f} "
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
    return CircleTrackingPIDEnv(
        physics="PyBulletPhysics",
        control_mode="AttitudeRate",
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode=render_mode,
        observation_noise=args.observation_noise,
        domain_randomization=args.domain_randomization,
        motor_thrust_noise=args.motor_thrust_noise,
        circle_radius=args.circle_radius,
        circle_period=args.circle_period,
        target_z=args.target_z,
        takeoff_s=args.takeoff_s,
        max_episode_s=args.max_episode_s,
        max_xy_ref_delta=args.max_xy_ref_delta,
        max_z_ref_delta=args.max_z_ref_delta,
        max_yaw_ref=args.max_yaw_ref_deg * np.pi / 180.0,
        success_mean_err=args.success_mean_err,
        trail=args.trail,
        trail_color=tuple(args.trail_color),
        trail_width=args.trail_width,
        trail_max_points=args.trail_max_points,
        debug_interval=args.debug_interval,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--alg", type=str, default="ppo", choices=["ppo", "iwpg", "trpo", "npg"])

    parser.add_argument("--circle-radius", type=float, default=0.8)
    parser.add_argument("--circle-period", type=float, default=10.0)
    parser.add_argument("--target-z", type=float, default=1.0)
    parser.add_argument("--takeoff-s", type=float, default=2.5)
    parser.add_argument("--max-episode-s", type=float, default=18.0)
    parser.add_argument("--success-mean-err", type=float, default=0.25)

    parser.add_argument("--max-xy-ref-delta", type=float, default=0.25)
    parser.add_argument("--max-z-ref-delta", type=float, default=0.20)
    parser.add_argument("--max-yaw-ref-deg", type=float, default=45.0)

    parser.add_argument("--observation-noise", type=float, default=1.0)
    parser.add_argument("--domain-randomization", type=float, default=0.01)
    parser.add_argument("--motor-thrust-noise", type=float, default=0.03)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pi-hidden-sizes", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--val-hidden-sizes", nargs="+", type=int, default=[128, 128])

    parser.add_argument("--until-stable", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--target-success-rate", type=float, default=0.7)
    parser.add_argument("--eval-episodes", type=int, default=10)

    parser.add_argument("--render-training", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-episodes", type=int, default=3)
    parser.add_argument("--debug-interval", type=int, default=100)

    parser.add_argument("--trail", action="store_true", help="Enable 3D motion trail in GUI")
    parser.add_argument("--trail-color", nargs=3, type=float, default=[0.05, 0.7, 1.0])
    parser.add_argument("--trail-width", type=float, default=1.8)
    parser.add_argument("--trail-max-points", type=int, default=4000)

    parser.add_argument("--ckpt", type=str, default=None)
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
    exp_name = f"CirclePID_round_{round_idx}"
    logger_kwargs = setup_logger_kwargs(exp_name=exp_name, base_dir=args.log_dir, seed=seed)

    env = build_env(args, render_mode="human" if args.render_training else None)

    max_ep_len = max(1, int(args.max_episode_s / env.TIME_STEP))
    steps_per_epoch = max(int(args.steps_per_epoch), max_ep_len)

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
        env_for_load = build_env(args, render_mode=None)
        try:
            ac = load_actor_critic_from_ckpt(args.ckpt, env_for_load)
        finally:
            _safe_close_env(env_for_load)
        success_rate, mean_ret, reasons = evaluate_policy(ac, args, episodes=args.play_episodes, render=True)
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

        print(
            f"Round {round_idx} eval: success_rate={success_rate:.2%}, "
            f"mean_return={mean_ret:.2f}, reasons={reasons}"
        )
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
