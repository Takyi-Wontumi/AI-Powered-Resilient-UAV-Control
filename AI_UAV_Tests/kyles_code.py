"""Python module to train RL algorithms.
 
Author:     Sven Gronauer (sven.gronauer@gmail.com)
"""
import argparse
import numpy as np
import psutil
import sys
import time
import warnings
import logging
import os
import getpass
from typing import Optional, Tuple
 
# local imports
import phoenix_drone_simulation  # import environments
import phoenix_drone_simulation.utils.loggers as loggers
from phoenix_drone_simulation.utils import utils
from phoenix_drone_simulation.algs.model import Model
from phoenix_drone_simulation.utils.mpi_tools import mpi_fork, mpi_print, is_root_process
 
# ---------- helpers to place results in repo by default ----------  # <<< added
def _default_results_dir() -> str:
    """
    Returns an absolute path to '<repo-root>/results'.
    This avoids writing to /var/tmp on Windows and makes it easy
    to find checkpoints and logs.
    """
    # train.py lives in <repo-root>/phoenix_drone_simulation/train.py
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir))  # go up one (to repo root)
    results = os.path.join(repo_root, "results")
    return results
# ------------------------------------------------------------------  # <<< added
 
 
def get_training_command_line_args(
        alg: Optional[str] = None,
        env: Optional[str] = None
) -> Tuple[argparse.Namespace, list]:
    r"""Fetches command line arguments from sys.argv.
 
    Parameters
    ----------
    alg: over-writes console
    env
    num_runs
 
    Returns
    -------
    Tuple of two lists
    """
 
    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
    # Seed must be < 2**32 => use 2**16 to allow seed += 10000*proc_id() for MPI
    random_seed = int(time.time()) % 2**16
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
 
    # Algorithm argument is set to passed argument `alg`
    if alg is not None:
        parser.add_argument('--alg', type=str, default=alg)
    else:  # --add alg as required console argument
        parser.add_argument(
            '--alg', type=str, required=True,
            help='Choose from: {iwpg, ppo, trpo, npg}')
 
    parser.add_argument(
        '--cores', '-c', type=int, default=physical_cores,
        help=f'Number of cores used for calculations.')
    parser.add_argument(
        '--debug', action='store_true',
        help='Show debug prints during training.')
 
    # Environment argument is set to passed argument `env`
    if env is not None:
        parser.add_argument('--env', type=str, default=env)
    else:
        parser.add_argument(
            '--env', type=str, required=True,
            help='Example: HopperBulletEnv-v0')
 
    parser.add_argument(
        '--no-mpi', action='store_true',
        help='Do not use MPI for parallel execution.')
    parser.add_argument(
        '--pi', nargs='+',  # creates args as list: pi=['64,', '64,', 'relu']
        help='Structure of policy network. Usage: --pi 64 64 relu')
    parser.add_argument(
        '--play', action='store_true',
        help='Visualize agent after training.')
    parser.add_argument(
        '--seed', default=random_seed, type=int,
        help=f'Define the init seed, e.g. {random_seed}')
    parser.add_argument(
        '--search', action='store_true',
        help='If given search over learning rates.')
 
    # --- changed default log dir to repo-local "results"  ---        # <<< added
    user_name = getpass.getuser()
    default_log_dir = _default_results_dir()
    parser.add_argument(
        '--log-dir', type=str, default=default_log_dir,
        help='Directory for logs & checkpoints (default: repo_root/results).')
    # ----------------------------------------------------------------  # <<< added
 
    _args, _unparsed_args = parser.parse_known_args()
 
    # Ensure the directory exists and is absolute                       # <<< added
    _args.log_dir = os.path.abspath(os.path.expanduser(_args.log_dir))
    os.makedirs(_args.log_dir, exist_ok=True)
 
    return _args, _unparsed_args
 
from gymnasium.wrappers import TransformObservation
import numpy as np
def run_training(args, unparsed_args, exp_name=None):
    r"""Executes one training loop with given parameters."""
 
    # Exclude hyper-threading and round cores to anything in: [2, 4, 8, 16, ...]
    physical_cores = 2 ** int(np.log2(psutil.cpu_count(logical=False)))
 
    # Use number of physical cores as default. If also hardware threading CPUs
    # should be used, enable this by the use_number_of_threads=True
    use_number_of_threads = True if args.cores > physical_cores else False
    if mpi_fork(args.cores, use_number_of_threads=use_number_of_threads):
        # Re-launches the current script with workers linked by MPI
        sys.exit()
    mpi_print('Unknowns:', unparsed_args)
 
    # update algorithm kwargs with unparsed arguments from command line
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [eval(v) for v in unparsed_args[1::2]]
    unparsed_kwargs = {k: v for k, v in zip(keys, values)}
 
    algorithm_kwargs = utils.get_defaults_kwargs(alg=args.alg, env_id=args.env)
 
    # update algorithm_kwargs with unparsed arguments from command line:
    algorithm_kwargs.update(**unparsed_kwargs)
 
    if args.pi is not None:
        hidden_sizes = tuple(eval(s) for s in args.pi[:-1])
        assert np.all([isinstance(s, int) for s in hidden_sizes]), \
            f'Hidden sizes must be of type: int'
        activation = args.pi[-1]
        assert isinstance(activation, str), 'Activation expected as string.'
 
        algorithm_kwargs['ac_kwargs']['pi']['hidden_sizes'] = hidden_sizes
        algorithm_kwargs['ac_kwargs']['pi']['activation'] = activation
 
    mpi_print('=' * 70)
    mpi_print('Parsed algorithm kwargs:')
    mpi_print(algorithm_kwargs)
    mpi_print('=' * 70)
 
    # Print and ensure the log directory                                 # <<< added
    mpi_print(f'Logging & checkpoints will be written under:\n  {args.log_dir}')
    os.makedirs(args.log_dir, exist_ok=True)
 
    model = Model(
        alg=args.alg,
        env_id=args.env,
        log_dir=args.log_dir,        # this controls where the repo writes runs
        init_seed=args.seed,
        algorithm_kwargs=algorithm_kwargs,
        use_mpi=not args.no_mpi
    )
    model.compile(num_cores=args.cores, exp_name=exp_name)
 
    model.fit()
    model.eval()
    if args.play:
        model.play()
 
 
if __name__ == '__main__':
    args, unparsed_args = get_training_command_line_args()
    run_training(args, unparsed_args)