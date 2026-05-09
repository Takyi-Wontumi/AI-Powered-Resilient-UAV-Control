import numpy as np

def run_flight_simulation(ekf, duration=5.0):
    """
    Simulates a flight and compares EKF output to Ground Truth.
    Returns: Dictionary of metrics (RMSE, NEES, NIS).
    """
    steps = int(duration / ekf.dt)
    nees_history = []
    nis_history = []
    errors = []

    # Simulated 'Ground Truth' Trajectory (a simple circle)
    for t_step in range(steps):
        t = t_step * ekf.dt
        true_pos = np.array([np.sin(t), np.cos(t), -1.0]) # Circular path
        
        # 1. Generate Noisy Measurement
        z = true_pos + np.random.normal(0, 0.05, 3) 
        # (Assuming velocity measurements are also available for 6D z)
        z_full = np.concatenate([z, np.zeros(3)]) 

        # 2. EKF Step
        ekf.predict(u=np.zeros(4))
        nis = ekf.update(z_full)

        # 3. Collect Statistics
        # NEES = (true_x - est_x).T * P^-1 * (true_x - est_x)
        err = true_pos - ekf.x[:3]
        inv_P = np.linalg.inv(ekf.P[:3, :3])
        nees = err.T @ inv_P @ err
        
        nees_history.append(nees)
        nis_history.append(nis)
        errors.append(np.linalg.norm(err))

    return {
        'mean_nees': np.mean(nees_history),
        'mean_nis': np.mean(nis_history),
        'rmse_pos': np.sqrt(np.mean(np.square(errors)))
    }