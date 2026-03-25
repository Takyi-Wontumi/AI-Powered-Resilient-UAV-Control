"""Python module to train RL algorithms.

Author:     Sven Gronauer (sven.gronauer@gmail.com)
"""
import argparse
import os
import sys
import time
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import psutil
import torch

# Ensure repo-root imports work when executing this file directly.
HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# local imports
import phoenix_drone_simulation  # noqa: F401 - ensures package import side-effects
from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.envs.register_envs import register_all_envs
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.utils.mpi_tools import mpi_fork, mpi_print


TRAJECTORY_TO_ENV = {
    "circle": "DroneFollowPathEnv-v0",
    "hover": "DroneHoverEnv-v0",
    "square": "DroneSquareEnv-v0",
    "helix": "DroneHelixEnv-v0",
    "sine": "DroneSineEnv-v0",
}


def _default_results_dir() -> str:
    """
    Returns an absolute path to '<repo-root>/results'.
    This avoids writing to /var/tmp on Windows and makes it easy
    to find checkpoints and logs.
    """
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))
    results = os.path.join(repo_root, "results")
    return results


def _resolve_env_id(args: argparse.Namespace) -> str:
    if args.env:
        return args.env
    return TRAJECTORY_TO_ENV[args.trajectory]


def _play_trained_policy(actor_critic, env_id: str, episodes: int = 3) -> None:
    """Run a short visual rollout after training."""
    env = gym.make(env_id, render_mode="human")
    actor_critic.eval()
    try:
        for i in range(episodes):
            done = False
            obs, _ = env.reset()
            ep_ret = 0.0
            ep_len = 0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    action, *_ = actor_critic(obs_t)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += reward
                ep_len += 1
                time.sleep(1.0 / 120.0)
            print(f"Play episode {i + 1}: return={ep_ret:.3f}, length={ep_len}")
    finally:
        env.close()


def get_training_command_line_args(
    alg: Optional[str] = None,
    env: Optional[str] = None,
) -> Tuple[argparse.Namespace, list]:
    r"""Fetches command line arguments from sys.argv."""

    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    # Seed must be < 2**32 => use 2**16 to allow seed += 10000*proc_id() for MPI
    random_seed = int(time.time()) % 2**16
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Algorithm argument is set to passed argument `alg`
    if alg is not None:
        parser.add_argument("--alg", type=str, default=alg)
    else:
        parser.add_argument(
            "--alg", type=str, required=True,
            help="Choose from: {iwpg, ppo, trpo, npg}")

    parser.add_argument(
        "--cores", "-c", type=int, default=physical_cores,
        help="Number of cores used for calculations.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Show debug prints during training.")

    # Environment argument can be explicit, or selected from --trajectory
    if env is not None:
        parser.add_argument("--env", type=str, default=env)
    else:
        parser.add_argument(
            "--env", type=str, default=None,
            help="Optional explicit Gym env id (overrides --trajectory).")

    parser.add_argument(
        "--trajectory", type=str, default="circle",
        choices=sorted(TRAJECTORY_TO_ENV.keys()),
        help="Trajectory task from AI_UAV_Tests.trajectories_library.")

    parser.add_argument(
        "--no-mpi", action="store_true",
        help="Do not use MPI for parallel execution.")
    parser.add_argument(
        "--pi", nargs="+",
        help="Structure of policy network. Usage: --pi 64 64 relu")
    parser.add_argument(
        "--play", action="store_true",
        help="Visualize a few episodes after training.")
    parser.add_argument(
        "--play-episodes", type=int, default=3,
        help="Number of visualized episodes after training.")
    parser.add_argument(
        "--render-training", action="store_true",
        help="Render PyBullet GUI during training.")
    parser.add_argument(
        "--seed", default=random_seed, type=int,
        help=f"Define the init seed, e.g. {random_seed}")
    parser.add_argument(
        "--search", action="store_true",
        help="If given search over learning rates.")

    default_log_dir = _default_results_dir()
    parser.add_argument(
        "--log-dir", type=str, default=default_log_dir,
        help="Directory for logs & checkpoints (default: repo_root/results).")

    _args, _unparsed_args = parser.parse_known_args()

    # Ensure the directory exists and is absolute
    _args.log_dir = os.path.abspath(os.path.expanduser(_args.log_dir))
    os.makedirs(_args.log_dir, exist_ok=True)

    return _args, _unparsed_args


def run_training(args, unparsed_args, exp_name=None):
    r"""Executes one training loop with given parameters."""
    env_id = _resolve_env_id(args)

    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))

    # Use number of physical cores as default.
    use_number_of_threads = True if args.cores > physical_cores else False
    if mpi_fork(args.cores, use_number_of_threads=use_number_of_threads):
        sys.exit()

    mpi_print("Unknowns:", unparsed_args)

    # Update algorithm kwargs with unparsed arguments from command line
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_kwargs = {k: v for k, v in zip(keys, values)}

    algorithm_kwargs = utils.get_defaults_kwargs(alg=args.alg, env_id=env_id)
    algorithm_kwargs.update(**unparsed_kwargs)

    if args.render_training:
        # IWPG/PPO forward kwargs to gym.make(...), enabling live PyBullet GUI.
        algorithm_kwargs["render_mode"] = "human"

    if args.pi is not None:
        hidden_sizes = tuple(eval(s) for s in args.pi[:-1])
        assert np.all([isinstance(s, int) for s in hidden_sizes]), \
            "Hidden sizes must be of type: int"
        activation = args.pi[-1]
        assert isinstance(activation, str), "Activation expected as string."

        algorithm_kwargs["ac_kwargs"]["pi"]["hidden_sizes"] = hidden_sizes
        algorithm_kwargs["ac_kwargs"]["pi"]["activation"] = activation

    mpi_print("=" * 70)
    mpi_print("Parsed algorithm kwargs:")
    mpi_print(algorithm_kwargs)
    mpi_print(f"Env: {env_id}")
    mpi_print("=" * 70)

    mpi_print(f"Logging & checkpoints will be written under:\n  {args.log_dir}")
    os.makedirs(args.log_dir, exist_ok=True)

    model = Model(
        alg=args.alg,
        env_id=env_id,
        log_dir=args.log_dir,
        init_seed=args.seed,
        algorithm_kwargs=algorithm_kwargs,
        use_mpi=not args.no_mpi,
    )
    model.compile(num_cores=args.cores, exp_name=exp_name)

    model.fit()
    model.eval()
    if args.play:
        _play_trained_policy(
            model.actor_critic,
            env_id=env_id,
            episodes=args.play_episodes,
        )


if __name__ == "__main__":
    register_all_envs()
    args, unparsed_args = get_training_command_line_args()
    run_training(args, unparsed_args)
