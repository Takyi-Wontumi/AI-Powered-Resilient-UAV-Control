"""
Train a policy on DroneMissionEnv using a trajectory from Trajectories.

Example:
    python examples/train_mission_trajectory.py --alg ppo --traj circle
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
from phoenix_drone_simulation.envs.mission import DroneMissionEnv
from phoenix_drone_simulation.envs.register_envs import register_all_envs
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.loggers import setup_logger_kwargs
from phoenix_drone_simulation.algs import core
from AI_UAV_Tests.trajectories_library import Trajectories as T
from AI_UAV_Tests.trajectories_library import FlightMission


# ---------------------------------------------------------
# Trajectory map (base functions)
# ---------------------------------------------------------
TRAJ_MAP = {
    "circle": T.circle_traj,
    "square": T.square_traj,
    "helix": T.helix_traj,
    "sine": T.sine_traj,
    "hover": T.hover_traj,
    # Placeholder to expose this option in argparse / checkpoint parsing.
    "flight_mission": T.hover_traj,
}


def _build_user_flight_mission():
    """Builds the exact mission requested by the user."""
    mission = FlightMission(default_z=1.0, ground_z=0.0)
    mission.add_takeoff(duration=3.0, target_z=1.0)
    mission.add_circle(duration=12.0, radius=1.0, period=12.0, z=1.0, center_xy=(-1.0, 0.0))
    mission.add_hover(duration=2.0, z=1.0)
    mission.add_point(duration=2.0, target=(-0.5, -0.5, 1.0))
    mission.add_square(duration=12.0, side=1.0, period=12.0, z=1.0, offset_xy=(-0.5, -0.5))
    mission.add_hover(duration=2.0, z=1.0)
    mission.add_landing(duration=4.0, ground_z=0.075)
    mission.add_hover(duration=1.0, z=0.075)
    mission.add_takeoff(duration=12.0, target_z=5.0)
    mission.add_hover(duration=3.0, z=5.0)
    mission.add_landing(duration=12.0, ground_z=0.055)
    return mission


def _make_traj_fn(traj_name: str, args):
    """Build a trajectory function with optional parameterization and takeoff."""
    if traj_name == "circle":
        def base_fn(t):
            return T.circle_traj(
                t,
                radius=args.traj_radius,
                z=args.traj_z,
                period=args.traj_period,
            )
    elif traj_name == "hover":
        def base_fn(t):
            return T.hover_traj(t, pos=(0.0, 0.0, args.traj_z))
    elif traj_name == "flight_mission":
        mission = _build_user_flight_mission()

        def base_fn(t):
            return mission(t)

        # Mission already includes explicit takeoff/hover/landing phases.
        return base_fn
    else:
        base_fn = TRAJ_MAP[traj_name]

    if args.takeoff_seconds > 0:
        takeoff_z = args.takeoff_z if args.takeoff_z is not None else min(args.traj_z, 0.3)

        def wrapped(t):
            if t < args.takeoff_seconds:
                return T.hover_traj(t, pos=(0.0, 0.0, takeoff_z))
            return base_fn(t - args.takeoff_seconds)

        return wrapped

    return base_fn


class DroneMissionTrajectoryEnv(DroneMissionEnv):
    """DroneMissionEnv that updates target from a trajectory function."""

    def __init__(
        self,
        trajectory_fn,
        noise_start=0.0,
        noise_end=1.0,
        noise_ramp_episodes=50,
        dr_start=0.0,
        dr_end=0.05,
        dr_ramp_episodes=50,
        **kwargs
    ):
        self.trajectory_fn = trajectory_fn
        self.noise_start = float(noise_start)
        self.noise_end = float(noise_end)
        self.noise_ramp_episodes = int(noise_ramp_episodes)
        self.dr_start = float(dr_start)
        self.dr_end = float(dr_end)
        self.dr_ramp_episodes = int(dr_ramp_episodes)
        self._episode_count = 0
        # Reward shaping + termination parameters
        self.success_xy = 0.05
        self.success_z = .05
        self.success_tilt = np.deg2rad(10)
        self.success_hold_steps = None  # set after TIME_STEP known
        self._success_counter = 0
        self.max_xy_err = 0.5
        self.max_z_err = 0.5
        self.max_tilt = np.deg2rad(30)
        self.takeoff_timeout = 2.0
        self.success_bonus = 5.0
        self.crash_penalty = 10.0
        self.done_reason = "none"
        super().__init__(**kwargs)
        self.success_hold_steps = int(2.0 / self.TIME_STEP)

    def reset(self, *args, **kwargs):
        self._episode_count += 1
        self.observation_noise = self._ramp_value(
            self.noise_start,
            self.noise_end,
            self.noise_ramp_episodes,
            self._episode_count
        )
        self.domain_randomization = self._ramp_value(
            self.dr_start,
            self.dr_end,
            self.dr_ramp_episodes,
            self._episode_count
        )
        if hasattr(self, "sensor_noise"):
            self.sensor_noise.bypass = not (self.observation_noise > 0)
        obs, info = super().reset(*args, **kwargs)
        self.mission_time = 0.0
        self._success_counter = 0
        pos_ref, _ = self.trajectory_fn(self.mission_time)
        self.set_target(pos_ref)
        return obs, info

    def step(self, action):
        self.mission_time += self.TIME_STEP
        pos_ref, _ = self.trajectory_fn(self.mission_time)
        self.set_target(pos_ref)
        return super().step(action)

    def _reference_errors(self):
        pos_ref, _ = self.trajectory_fn(self.mission_time)
        pos_err = pos_ref - self.drone.xyz
        xy_err = float(np.linalg.norm(pos_err[:2]))
        z_err = float(pos_err[2])
        tilt = float(np.linalg.norm(self.drone.rpy[:2]))
        return pos_err, xy_err, z_err, tilt

    @staticmethod
    def _ramp_value(start, end, ramp_episodes, episode_idx):
        if ramp_episodes <= 0 or start == end:
            return end
        frac = min(1.0, float(episode_idx) / float(ramp_episodes))
        return start + (end - start) * frac

    def compute_reward(self, action):
        _, xy_err, z_err, tilt = self._reference_errors()

        r = 2.0 * np.exp(-(z_err ** 2) / 0.05)
        r += 1.0 * np.exp(-(xy_err ** 2) / 0.05)
        r += 0.5 * np.exp(-(tilt ** 2) / (self.success_tilt ** 2))
        r -= 0.01 * float(np.sum(np.square(action)))

        if abs(z_err) < self.success_z and xy_err < self.success_xy and tilt < self.success_tilt:
            r += self.success_bonus

        if self.done_reason == "mission_success":
            r += self.success_bonus
        elif self.done_reason != "none":
            r -= self.crash_penalty

        return float(r)

    def compute_info(self, action=None):
        info = super().compute_info()
        info["done_reason"] = self.done_reason
        info["observation_noise"] = self.observation_noise
        info["domain_randomization"] = self.domain_randomization
        return info

    def compute_done(self) -> bool:
        # allow takeoff phase without early termination
        self.done_reason = "none"
        if not self.airborne and self.drone.xyz[2] > 0.15:
            self.airborne = True

        if not self.airborne and self.mission_time < self.takeoff_timeout:
            return False

        _, xy_err, z_err, tilt = self._reference_errors()
        in_tol = (
            abs(z_err) < self.success_z
            and xy_err < self.success_xy
            and tilt < self.success_tilt
        )
        self._success_counter = self._success_counter + 1 if in_tol else 0
        if self._success_counter >= self.success_hold_steps:
            self.done_reason = "mission_success"
            return True

        if abs(z_err) > self.max_z_err or xy_err > self.max_xy_err:
            self.done_reason = "out_of_bounds"
            return True
        if tilt > self.max_tilt:
            self.done_reason = "tilt_limit"
            return True
        if self.airborne and self.drone.xyz[2] <= self.ground_z:
            self.done_reason = "ground_contact"
            return True
        return False


def build_env(args):
    traj_fn = _make_traj_fn(args.traj, args)
    env = DroneMissionTrajectoryEnv(
        physics="PyBulletPhysics",
        control_mode=args.control_mode,
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human" if args.render else None,
        observation_noise=args.observation_noise,
        domain_randomization=args.domain_randomization,
        motor_thrust_noise=args.motor_thrust_noise,
        trajectory_fn=traj_fn,
        noise_start=args.noise_start,
        noise_end=args.observation_noise,
        noise_ramp_episodes=args.noise_ramp_episodes,
        dr_start=args.dr_start,
        dr_end=args.domain_randomization,
        dr_ramp_episodes=args.dr_ramp_episodes,
    )
    return env


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--alg", type=str, default="ppo",
                        help="Algorithm: ppo, iwpg, trpo, npg")
    parser.add_argument("--traj", type=str, default="circle",
                        choices=sorted(TRAJ_MAP.keys()))
    parser.add_argument("--traj-radius", type=float, default=1.0,
                        help="Circle radius (only for traj=circle).")
    parser.add_argument("--traj-z", type=float, default=1.0,
                        help="Z height for traj=circle/hover.")
    parser.add_argument("--traj-period", type=float, default=10.0,
                        help="Circle period in seconds (only for traj=circle).")
    parser.add_argument("--takeoff-seconds", type=float, default=0.0,
                        help="If >0, hover at (0,0,takeoff_z) before starting trajectory.")
    parser.add_argument("--takeoff-z", type=float, default=0.3,
                        help="Takeoff/initial hover height when takeoff-seconds > 0.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=4000)
    parser.add_argument("--max-ep-len", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(ROOT_DIR, "runs"))
    parser.add_argument("--control-mode", type=str, default="PWM",
                        choices=["PWM", "Attitude", "AttitudeRate"])
    parser.add_argument("--observation-noise", type=float, default=1.0)
    parser.add_argument("--domain-randomization", type=float, default=0.05)
    parser.add_argument("--motor-thrust-noise", type=float, default=0.05)
    parser.add_argument("--noise-start", type=float, default=0.0)
    parser.add_argument("--noise-ramp-episodes", type=int, default=50)
    parser.add_argument("--dr-start", type=float, default=0.0)
    parser.add_argument("--dr-ramp-episodes", type=int, default=50)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--use-entropy", action="store_true", default=True)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--no-exploration-anneal", dest="use_exploration_noise_anneal",
                        action="store_false")
    parser.add_argument("--no-lr-decay", dest="use_linear_lr_decay",
                        action="store_false")
    parser.set_defaults(use_exploration_noise_anneal=True, use_linear_lr_decay=True)
    parser.add_argument("--pi-hidden-sizes", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--val-hidden-sizes", nargs="+", type=int, default=[128, 128])
    parser.add_argument("--use-reward-scaling", type=str, default="true",
                        choices=["true", "false", "1", "0"])
    parser.add_argument("--print-done-reason", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a run directory containing config.json and torch_save/model.pt")
    parser.add_argument("--play", action="store_true",
                        help="Render a trained policy (requires --ckpt or trains then plays).")
    parser.add_argument("--episodes", type=int, default=3)
    return parser.parse_args()

def get_alg_defaults(alg: str) -> dict:
    defaults_mod = utils.get_alg_module(alg, "defaults")
    for fn_name in ("defaults", "locomotion", "gym_locomotion_envs"):
        if hasattr(defaults_mod, fn_name):
            return getattr(defaults_mod, fn_name)()
    return {}

def load_actor_critic_from_ckpt(ckpt_dir, env):
    conf = utils.get_file_contents(os.path.join(ckpt_dir, "config.json"))
    ac = core.ActorCritic(
        actor_type=conf["actor"],
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_standardized_obs=conf.get("use_standardized_obs", True),
        use_scaled_rewards=conf.get("use_reward_scaling", False),
        use_shared_weights=False,
        ac_kwargs=conf["ac_kwargs"],
    )
    model_path = os.path.join(ckpt_dir, "torch_save", "model.pt")
    ac.load_state_dict(torch.load(model_path), strict=False)
    ac.eval()
    return ac, conf

def play_policy(ac, env, episodes=3, show_done_reason=False):
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        steps = 0
        last_info = {}
        while not done:
            with torch.no_grad():
                action, _, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ret += r
            steps += 1
            time.sleep(1.0 / 120.0)
            last_info = _ if _ is not None else last_info
        msg = f"Episode {ep+1}: return={ret:.2f} steps={steps}"
        if show_done_reason and last_info:
            msg += f" done_reason={last_info.get('done_reason', 'unknown')}"
        print(msg)

def _parse_exp_name(exp_name):
    if not exp_name or not exp_name.startswith("Mission_"):
        return None, None
    parts = exp_name.split("_")
    if len(parts) < 3:
        return None, None
    traj = parts[1].lower()
    control_mode = parts[2]
    return traj, control_mode

def build_env_from_ckpt(args):
    config_path = os.path.join(args.ckpt, "config.json")
    conf = utils.get_file_contents(config_path)
    exp_name = conf.get("exp_name") or conf.get("logger_kwargs", {}).get("exp_name")
    traj_from_ckpt, control_from_ckpt = _parse_exp_name(exp_name)

    env_cfg_path = os.path.join(args.ckpt, "env_config.json")
    env_cfg = utils.get_file_contents(env_cfg_path) if os.path.isfile(env_cfg_path) else {}

    traj = traj_from_ckpt if traj_from_ckpt in TRAJ_MAP else args.traj
    control_mode = control_from_ckpt if control_from_ckpt in ["PWM", "Attitude", "AttitudeRate"] else args.control_mode

    traj_fn = _make_traj_fn(traj, args)
    env = DroneMissionTrajectoryEnv(
        physics="PyBulletPhysics",
        control_mode=control_mode,
        drone_model="cf21x_bullet",
        dropout_mode="NONE",
        render_mode="human",
        observation_noise=env_cfg.get("observation_noise", args.observation_noise),
        domain_randomization=env_cfg.get("domain_randomization", args.domain_randomization),
        motor_thrust_noise=env_cfg.get("motor_thrust_noise", args.motor_thrust_noise),
        trajectory_fn=traj_fn,
        noise_start=args.noise_start,
        noise_end=env_cfg.get("observation_noise", args.observation_noise),
        noise_ramp_episodes=args.noise_ramp_episodes,
        dr_start=args.dr_start,
        dr_end=env_cfg.get("domain_randomization", args.domain_randomization),
        dr_ramp_episodes=args.dr_ramp_episodes,
    )
    return env, traj, control_mode


def main():
    args = parse_args()

    # Ensure custom envs are registered for defaults lookup
    register_all_envs()

    if args.play:
        args.render = True

    if args.ckpt:
        env, traj_used, control_used = build_env_from_ckpt(args)
        if traj_used != args.traj or control_used != args.control_mode:
            print(
                f"Note: checkpoint was trained for traj={traj_used} "
                f"control_mode={control_used}. Overriding CLI settings."
            )
        ac, _ = load_actor_critic_from_ckpt(args.ckpt, env)
        play_policy(ac, env, episodes=args.episodes, show_done_reason=args.print_done_reason)
        return

    env = build_env(args)

    algorithm_kwargs = get_alg_defaults(args.alg)
    algorithm_kwargs.update(
        steps_per_epoch=args.steps_per_epoch,
        max_ep_len=args.max_ep_len,
        epochs=args.epochs,
        seed=args.seed,
        use_entropy=args.use_entropy,
        entropy_coef=args.entropy_coef,
        use_exploration_noise_anneal=args.use_exploration_noise_anneal,
        use_linear_lr_decay=args.use_linear_lr_decay,
        use_reward_scaling=args.use_reward_scaling.lower() in ("true", "1"),
    )
    if "ac_kwargs" in algorithm_kwargs:
        algorithm_kwargs["ac_kwargs"]["pi"]["hidden_sizes"] = tuple(args.pi_hidden_sizes)
        algorithm_kwargs["ac_kwargs"]["val"]["hidden_sizes"] = tuple(args.val_hidden_sizes)

    logger_kwargs = setup_logger_kwargs(
        exp_name=f"Mission_{args.traj}_{args.control_mode}",
        base_dir=args.log_dir,
        seed=args.seed,
    )
    algorithm_kwargs["logger_kwargs"] = logger_kwargs

    alg = utils.get_alg_class(args.alg, env_id=env, **algorithm_kwargs)
    ac, _ = alg.learn()

    if args.play:
        play_policy(ac, env, episodes=args.episodes, show_done_reason=args.print_done_reason)

    if args.print_done_reason:
        import torch
        obs, _ = env.reset()
        done = False
        last_info = {}
        while not done:
            with torch.no_grad():
                action, _, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_info = info
        print(f"done_reason={last_info.get('done_reason', 'unknown')}")


if __name__ == "__main__":
    main()
