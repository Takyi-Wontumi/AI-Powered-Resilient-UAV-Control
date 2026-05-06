import numpy as np
from AI_UAV_Tests.rl_dropout_policy import DroneDropoutRLEnv


def test_zero_action_keeps_pid():
    env = DroneDropoutRLEnv(render_mode=None, dropout_randomize=False)
    state, _ = env.reset()

    pos_ref, vel_ref = env._reference_for_policy()
    pid_out = env._compute_pid_output(pos_ref, vel_ref)
    U1_pid = float(pid_out["thrust_cmd"])
    tau_pid = np.asarray(pid_out.get("tau_cmd", np.zeros(3)), dtype=float)

    zero_action = np.zeros(4, dtype=float)
    out = env._action_to_control(zero_action, pid_out)
    U1_final = float(out["thrust_cmd"])
    tau_final = np.asarray(out.get("tau_cmd", np.zeros(3)), dtype=float)

    assert abs(U1_final - U1_pid) < 1e-6 or np.isclose(U1_final, U1_pid)
    assert np.allclose(tau_final, tau_pid, atol=1e-6)

    env.close()
