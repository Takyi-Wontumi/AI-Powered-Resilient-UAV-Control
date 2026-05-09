"""Quick smoke runner for residual PPO wiring.
Runs a few steps with random actions and prints PID, delta, final commands, and EKF trace.
"""
import os
import sys
import numpy as np

# Ensure project root is on sys.path so AI_UAV_Tests can be imported
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from AI_UAV_Tests.rl.rl_dropout_policy import DroneDropoutRLEnv


def run(steps=20):
    env = DroneDropoutRLEnv(render_mode=None, dropout_randomize=False)
    state, _ = env.reset()
    # Make decoupling threshold short for the smoke test so we can observe it
    env.ekf.decouple_after_s = 0.1

    for i in range(steps):
        # small random actions around zero
        action = 0.01 * np.random.randn(4).astype(np.float32)
        # Force a dropout starting at step 5
        if i == 5:
            env.env.dropout_mgr.mode = "HOV"
            env.env.trigger_dropout()
            print("-- triggered dropout --")

        state, reward, done, _, info = env.step(action)

        # extract logging fields
        cov_trace = info.get("cov_trace", None)
        pos_err = info.get("pos_error", None)
        drop = info.get("dropout_active", False)

        # Check EKF dropout internal state
        dropout_time = getattr(env.ekf, "dropout_time", None)
        # detect decoupling by checking cross-covariance block norm
        P = env.ekf.ekf.P
        cross_norm = float(np.linalg.norm(P[0:6, 6:12]))

        print(
            f"step={i:02d} drop={drop} pos_err={pos_err:.3f} cov_trace={cov_trace:.6f} "
            f"dropout_time={dropout_time:.4f} cross_norm={cross_norm:.6e} reward={reward:.3f}"
        )

        if done:
            print("Episode finished")
            break

    env.close()


if __name__ == '__main__':
    run(steps=50)
