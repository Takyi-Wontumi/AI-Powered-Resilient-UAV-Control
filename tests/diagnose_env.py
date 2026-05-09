#!/usr/bin/env python3
"""
Brutal environment diagnostic.
No PPO. Just raw simulation.
"""

import os
import sys
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from AI_UAV_Tests.rl.rl_dropout_policy import DroneDropoutRLEnv


def main():
    print("\n" + "="*60)
    print("  ENVIRONMENT DIAGNOSTIC")
    print("="*60 + "\n")

    env = DroneDropoutRLEnv(render_mode=None, dropout_randomize=False)

    for episode in range(3):
        print(f"\nEpisode {episode + 1}")
        print("-" * 60)

        obs, info = env.reset()
        print(f"Reset successful. Initial obs shape: {obs.shape}")
        print(f"Initial obs (first 5 dims): {obs[:5]}")

        episode_reward = 0.0
        dropout_started = False

        for step in range(500):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            is_dropout = info.get('dropout_active', False)

            # Alert when dropout starts
            if is_dropout and not dropout_started:
                print(f"  >>> DROPOUT TRIGGERED AT STEP {step} <<<")
                dropout_started = True

            # Print every 10 steps OR when dropout changes
            if step % 10 == 0 or step < 5 or done or (is_dropout != dropout_started):
                dropout_str = " [DROPOUT]" if is_dropout else ""
                print(f"  Step {step:3d}: reward={reward:7.4f}, done={done}, "
                      f"z={info.get('drone_z', 0):6.3f}, obs_sum={np.sum(obs):8.2f}{dropout_str}")

            if done:
                print(f"\n  >>> EPISODE ENDED AT STEP {step + 1} <<<")
                print(f"  Total reward: {episode_reward:.3f}")
                print(f"  Crash: {info.get('crashed', False)}")
                print(f"  Dropout active: {info.get('dropout_active', False)}")
                break

        if not done:
            print(f"  Episode completed 100 steps (no early termination)")
            print(f"  Total reward: {episode_reward:.3f}")

    env.close()
    print("\n" + "="*60)
    print("  DIAGNOSTIC COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
