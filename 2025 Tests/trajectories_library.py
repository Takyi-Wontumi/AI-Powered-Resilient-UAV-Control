import numpy as np

class Trajectories:
    """Collection of reference trajectory generators for QuadcopterSim."""

    @staticmethod
    def square_traj(t: float, side=1.0, period=16.0, z=1.0):
        """Square path on XY plane."""
        T = period / 4
        tm = t % period
        if tm < T:
            s = tm / T; x, y, vx, vy = side * s, 0, side / T, 0
        elif tm < 2 * T:
            s = (tm - T) / T; x, y, vx, vy = side, side * s, 0, side / T
        elif tm < 3 * T:
            s = (tm - 2 * T) / T; x, y, vx, vy = side * (1 - s), side, -side / T, 0
        else:
            s = (tm - 3 * T) / T; x, y, vx, vy = 0, side * (1 - s), 0, -side / T
        return np.array([x, y, z]), np.array([vx, vy, 0])

    @staticmethod
    def circle_traj(t: float, radius=1.0, z=1.0, period=10.0):
        """Constant-altitude circular path."""
        omega = 2 * np.pi / period
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        vx = -radius * omega * np.sin(omega * t)
        vy =  radius * omega * np.cos(omega * t)
        return np.array([x, y, z]), np.array([vx, vy, 0])

    @staticmethod
    def helix_traj(t: float, radius=1.0, height=1.5, period=10.0):
        """Helical ascent trajectory."""
        omega = 2 * np.pi / period
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = height * (t % period) / period
        vx = -radius * omega * np.sin(omega * t)
        vy =  radius * omega * np.cos(omega * t)
        vz = height / period
        return np.array([x, y, z]), np.array([vx, vy, vz])

    @staticmethod
    def sine_traj(t: float, amplitude=0.5, freq=0.2, z=1.0):
        """Sine-wave path along X."""
        x = t * 0.1
        y = amplitude * np.sin(2 * np.pi * freq * t)
        vx = 0.1
        vy = amplitude * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
        return np.array([x, y, z]), np.array([vx, vy, 0])
