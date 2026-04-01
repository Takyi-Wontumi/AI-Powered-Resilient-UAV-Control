import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

# pyproj is required for GPS -> ECEF conversion
try:
    from pyproj import Transformer
except Exception as e:
    Transformer = None


# =========================================================
# Local analytic trajectories (keep these for simulation/RL)
# =========================================================
class Trajectories:
    """Collection of local reference trajectory generators returning (pos, vel)."""

    @staticmethod
    def hover_traj(t: float = 0.0, pos=(0.0, 0.0, 1.0)):
        x, y, z = pos
        return np.array([x, y, z], dtype=float), np.zeros(3, dtype=float)

    @staticmethod
    def point_traj(target=(1.0, 1.0, 1.0)):
        return np.array(target, dtype=float), np.zeros(3, dtype=float)

    @staticmethod
    def square_traj(t: float, side=1.0, period=16.0, z=1.0):
        T = period / 4.0
        tm = t % period
        if tm < T:
            s = tm / T
            x, y, vx, vy = side * s, 0.0, side / T, 0.0
        elif tm < 2 * T:
            s = (tm - T) / T
            x, y, vx, vy = side, side * s, 0.0, side / T
        elif tm < 3 * T:
            s = (tm - 2 * T) / T
            x, y, vx, vy = side * (1 - s), side, -side / T, 0.0
        else:
            s = (tm - 3 * T) / T
            x, y, vx, vy = 0.0, side * (1 - s), 0.0, -side / T
        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float)

    @staticmethod
    def circle_traj(t: float, radius=1.0, z=1.0, period=10.0):
        omega = 2.0 * np.pi / period
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float)

    @staticmethod
    def helix_traj(t: float, radius=1.0, height=1.5, period=10.0):
        omega = 2.0 * np.pi / period
        x = radius * np.cos(omega * t)
        y = radius * np.sin(omega * t)
        z = height * (t % period) / period
        vx = -radius * omega * np.sin(omega * t)
        vy = radius * omega * np.cos(omega * t)
        vz = height / period
        return np.array([x, y, z], dtype=float), np.array([vx, vy, vz], dtype=float)

    @staticmethod
    def sine_traj(t: float, amplitude=0.5, freq=0.2, z=1.0):
        x = t * 0.1
        y = amplitude * np.sin(2.0 * np.pi * freq * t)
        vx = 0.1
        vy = amplitude * 2.0 * np.pi * freq * np.cos(2.0 * np.pi * freq * t)
        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float)


# =========================================================
# Composable mission framework (local analytic trajectories)
# =========================================================
@dataclass
class FlightMissionPhase:
    name: str
    t_start: float
    t_end: float
    trajectory_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]]

    def contains(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        local_t = np.clip(float(t) - self.t_start, 0.0, self.t_end - self.t_start)
        return self.trajectory_fn(local_t)


class FlightMission:
    """
    Compose a full flight mission from local trajectory primitives.

    The mission is callable:
        pos_ref, vel_ref = mission(t)

    Typical usage:
        mission = FlightMission(default_z=1.0)
        mission.add_takeoff(duration=3.0, target_z=1.0)
        mission.add_circle(duration=20.0, radius=1.0, period=10.0)
        mission.add_point(duration=4.0, target=(1.0, 1.0, 1.2))
        mission.add_hover(duration=5.0, z=1.2)
        mission.add_square(duration=20.0, side=1.0, period=16.0, z=1.2)
        mission.add_hover(duration=3.0, z=1.2)
        mission.add_landing(duration=6.0, ground_z=0.0)
    """
    DEFAULT_PHASE_COLORS = {
        "circle": (0.20, 0.60, 1.00),
        "hover": (0.10, 0.85, 0.25),
        "square": (1.00, 0.55, 0.05),
        "point": (0.10, 0.85, 0.85),
        "landing": (0.95, 0.20, 0.20),
        "takeoff": (0.80, 0.30, 0.95),
        "pre_start": (0.80, 0.80, 0.80),
        "post_mission": (0.60, 0.60, 0.60),
        "dropout": (0.95, 0.95, 0.20),
        "unknown": (1.00, 1.00, 1.00),
    }

    def __init__(
        self,
        start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        default_z: float = 1.0,
        ground_z: float = 0.0,
        auto_takeoff: bool = True,
        auto_takeoff_duration: float = 3.0,
    ):
        self.start_pos = np.asarray(start_pos, dtype=float)
        self.default_z = float(default_z)
        self.ground_z = float(ground_z)
        self.auto_takeoff = bool(auto_takeoff)
        self.auto_takeoff_duration = float(auto_takeoff_duration)
        self.phases: List[FlightMissionPhase] = []

        self._terminal_pos = self.start_pos.copy()
        self._terminal_vel = np.zeros(3, dtype=float)

    @property
    def total_time(self) -> float:
        if not self.phases:
            return 0.0
        return float(self.phases[-1].t_end)

    def summary(self) -> dict:
        return {
            "total_time": self.total_time,
            "num_phases": len(self.phases),
            "auto_takeoff": bool(self.auto_takeoff),
            "auto_takeoff_duration": float(self.auto_takeoff_duration),
            "phases": [
                {
                    "name": p.name,
                    "t_start": float(p.t_start),
                    "t_end": float(p.t_end),
                    "duration": float(p.t_end - p.t_start),
                }
                for p in self.phases
            ],
        }

    def __call__(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        t = float(t)
        if not self.phases:
            return self._terminal_pos.copy(), np.zeros(3, dtype=float)

        if t <= self.phases[0].t_start:
            return self.phases[0].eval(self.phases[0].t_start)

        for phase in self.phases:
            if phase.contains(t):
                return phase.eval(t)

        # after mission end: hold terminal state
        return self._terminal_pos.copy(), np.zeros(3, dtype=float)

    def phase_at(self, t: float) -> Optional[FlightMissionPhase]:
        t = float(t)
        for phase in self.phases:
            if phase.contains(t):
                return phase
        return None

    def phase_name_at(self, t: float) -> str:
        phase = self.phase_at(t)
        if phase is not None:
            return str(phase.name)
        if t < 0.0:
            return "pre_start"
        return "post_mission"

    def phase_info_at(self, t: float) -> Tuple[str, float, float]:
        phase = self.phase_at(t)
        if phase is None:
            return self.phase_name_at(t), 0.0, 0.0
        elapsed = np.clip(float(t) - phase.t_start, 0.0, phase.t_end - phase.t_start)
        total = float(phase.t_end - phase.t_start)
        return str(phase.name), float(elapsed), total

    def get_phase_color(self, phase_name: str) -> Tuple[float, float, float]:
        color = self.DEFAULT_PHASE_COLORS.get(str(phase_name), self.DEFAULT_PHASE_COLORS["unknown"])
        return float(color[0]), float(color[1]), float(color[2])

    def trail_color_at_time(self, t: float, dropout_active: bool = False) -> Tuple[float, float, float]:
        if dropout_active:
            return self.get_phase_color("dropout")
        return self.get_phase_color(self.phase_name_at(t))

    def preflight_check(
        self,
        speedup: float = 8.0,
        sample_hz: float = 200.0,
        line_width: float = 2.0,
        block: bool = True,
    ) -> dict:
        """
        Fast mission preview dashboard (no physics).

        Shows a window that animates reference position in accelerated time,
        with phase colors and a live dashboard text panel.
        """
        if not self.phases:
            raise ValueError("Mission has no phases. Add at least one phase before preflight_check().")

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise ImportError("matplotlib is required for preflight_check().") from exc

        speedup = max(float(speedup), 1e-3)
        dt = 1.0 / max(float(sample_hz), 1e-3)
        total_time = float(self.total_time)
        n_steps = max(2, int(np.ceil(total_time / dt)) + 1)
        t_grid = np.linspace(0.0, total_time, n_steps)

        pos_grid = np.zeros((n_steps, 3), dtype=float)
        vel_grid = np.zeros((n_steps, 3), dtype=float)
        for i, t in enumerate(t_grid):
            p, v = self(t)
            pos_grid[i] = p
            vel_grid[i] = v
        speed_grid = np.linalg.norm(vel_grid, axis=1)

        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.0, 1.0])
        ax_xy = fig.add_subplot(gs[:, 0])
        ax_z = fig.add_subplot(gs[0, 1])
        ax_dash = fig.add_subplot(gs[1, 1])

        ax_xy.set_title("Preflight Mission Preview (XY)")
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.grid(True)
        ax_xy.set_aspect("equal", adjustable="box")
        ax_xy.plot(pos_grid[:, 0], pos_grid[:, 1], "--", color="0.8", linewidth=1.0)

        pad = 0.5
        ax_xy.set_xlim(float(np.min(pos_grid[:, 0]) - pad), float(np.max(pos_grid[:, 0]) + pad))
        ax_xy.set_ylim(float(np.min(pos_grid[:, 1]) - pad), float(np.max(pos_grid[:, 1]) + pad))

        marker, = ax_xy.plot([pos_grid[0, 0]], [pos_grid[0, 1]], "o", color=self.get_phase_color(self.phase_name_at(0.0)))

        ax_z.set_title("Altitude Reference")
        ax_z.set_xlabel("Time [s]")
        ax_z.set_ylabel("Z [m]")
        ax_z.grid(True)
        ax_z.plot(t_grid, pos_grid[:, 2], "--", color="0.8", linewidth=1.0)
        z_prog, = ax_z.plot([t_grid[0]], [pos_grid[0, 2]], color="tab:blue", linewidth=2.0)
        ax_z.set_xlim(0.0, total_time)
        z_min = float(np.min(pos_grid[:, 2]))
        z_max = float(np.max(pos_grid[:, 2]))
        z_pad = 0.15 + 0.05 * max(1.0, z_max - z_min)
        ax_z.set_ylim(z_min - z_pad, z_max + z_pad)

        ax_dash.axis("off")
        header = (
            f"{'task':<10}  {'task_time(s)':>12}  {'mission_time(s)':>15}  "
            f"{'speed(m/s)':>10}  {'x(m)':>8}  {'y(m)':>8}  {'z(m)':>8}"
        )
        tally_header = "TASK TALLY (all phases)"
        dashboard_text = ax_dash.text(
            0.01,
            0.95,
            "Starting preflight preview...",
            va="top",
            ha="left",
            family="monospace",
            fontsize=8.5,
        )

        def _task_tally_text(t_now: float, mark_complete: bool = False) -> str:
            rows = [tally_header]
            for idx, phase in enumerate(self.phases, start=1):
                if mark_complete or t_now >= phase.t_end - 1e-9:
                    status = "DONE"
                elif phase.contains(t_now):
                    status = "ACTIVE"
                else:
                    status = "PENDING"
                rows.append(
                    f"{idx:02d}. {phase.name:<10}  "
                    f"{phase.t_start:6.2f}->{phase.t_end:6.2f}s  [{status}]"
                )
            return "\n".join(rows)

        for i, t in enumerate(t_grid):
            phase_name, phase_elapsed, phase_total = self.phase_info_at(float(t))
            pos = pos_grid[i]
            speed = float(speed_grid[i])

            marker.set_data([pos[0]], [pos[1]])
            marker.set_color(self.get_phase_color(phase_name))

            if i > 0:
                ax_xy.plot(
                    [pos_grid[i - 1, 0], pos_grid[i, 0]],
                    [pos_grid[i - 1, 1], pos_grid[i, 1]],
                    color=self.get_phase_color(phase_name),
                    linewidth=line_width,
                )

            z_prog.set_data(t_grid[: i + 1], pos_grid[: i + 1, 2])

            row = (
                f"{phase_name:<10}  {phase_elapsed:5.2f}/{phase_total:5.2f}  "
                f"{t:6.2f}/{total_time:6.2f}  {speed:10.3f}  "
                f"{pos[0]:8.3f}  {pos[1]:8.3f}  {pos[2]:8.3f}"
            )
            dashboard_text.set_text(
                "PRE-FLIGHT DASHBOARD (accelerated)\n"
                f"speedup={speedup:.1f}x  sample_hz={1.0 / dt:.1f}\n"
                f"{header}\n{row}\n\n{_task_tally_text(float(t))}"
            )
            plt.pause(max(1e-3, dt / speedup))

        # Keep the completed state visible on the dashboard.
        phase_name, phase_elapsed, phase_total = self.phase_info_at(total_time)
        pos = pos_grid[-1]
        speed = float(speed_grid[-1])
        row = (
            f"{phase_name:<10}  {phase_elapsed:5.2f}/{phase_total:5.2f}  "
            f"{total_time:6.2f}/{total_time:6.2f}  {speed:10.3f}  "
            f"{pos[0]:8.3f}  {pos[1]:8.3f}  {pos[2]:8.3f}"
        )
        dashboard_text.set_text(
            "PRE-FLIGHT DASHBOARD (completed)\n"
            f"speedup={speedup:.1f}x  sample_hz={1.0 / dt:.1f}  status=COMPLETED\n"
            f"{header}\n{row}\n\n{_task_tally_text(total_time, mark_complete=True)}"
        )
        fig.canvas.draw_idle()
        plt.pause(0.001)

        if block:
            plt.show()
        else:
            plt.show(block=False)

        return {
            "t": t_grid,
            "pos": pos_grid,
            "vel": vel_grid,
            "speed": speed_grid,
        }

    def _ensure_takeoff_if_needed(self, target_z: float):
        if not self.auto_takeoff:
            return
        target_z = float(target_z)
        current_z = float(self._terminal_pos[2])
        if target_z > current_z + 1e-6:
            self.add_takeoff(duration=self.auto_takeoff_duration, target_z=target_z)

    def add_takeoff(
        self,
        duration: float = 3.0,
        target_z: Optional[float] = None,
        profile: str = "min_jerk",
    ):
        """
        Vertical takeoff from current terminal position to target_z.

        Can be used as first phase or between other mission phases.
        """
        z_target = self.default_z if target_z is None else float(target_z)
        p0 = self._terminal_pos.copy()
        z0 = float(p0[2])

        # If already at/above target altitude, keep mission unchanged.
        if z_target <= z0 + 1e-6:
            return self

        T = float(duration)
        if T <= 0.0:
            raise ValueError("Takeoff duration must be > 0.")

        mode = str(profile).strip().lower()

        def _traj(local_t):
            tau = np.clip(float(local_t) / T, 0.0, 1.0)
            if mode == "min_jerk":
                s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                ds_dt = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T
            elif mode == "linear":
                s = tau
                ds_dt = 1.0 / T
            else:
                raise ValueError(f"Unknown takeoff profile: {profile}")

            pos = p0.copy()
            pos[2] = z0 + s * (z_target - z0)
            vel = np.zeros(3, dtype=float)
            vel[2] = ds_dt * (z_target - z0)
            return pos, vel

        return self.add_phase("takeoff", T, _traj)

    def add_phase(self, name: str, duration: float, trajectory_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]]):
        duration = float(duration)
        if duration <= 0:
            raise ValueError("Phase duration must be > 0.")

        t_start = self.total_time
        t_end = t_start + duration
        phase = FlightMissionPhase(
            name=str(name),
            t_start=t_start,
            t_end=t_end,
            trajectory_fn=trajectory_fn,
        )
        self.phases.append(phase)

        pos_end, vel_end = phase.eval(t_end)
        self._terminal_pos = np.asarray(pos_end, dtype=float).copy()
        self._terminal_vel = np.asarray(vel_end, dtype=float).copy()
        return self

    def add_circle(
        self,
        duration: float,
        radius: float = 1.0,
        period: float = 10.0,
        z: Optional[float] = None,
        center_xy: Tuple[float, float] = (0.0, 0.0),
    ):
        z_use = self.default_z if z is None else float(z)
        self._ensure_takeoff_if_needed(z_use)
        cx, cy = float(center_xy[0]), float(center_xy[1])

        def _traj(local_t):
            pos, vel = Trajectories.circle_traj(local_t, radius=radius, z=z_use, period=period)
            pos = pos.copy()
            pos[0] += cx
            pos[1] += cy
            return pos, vel

        return self.add_phase("circle", duration, _traj)

    def add_square(
        self,
        duration: float,
        side: float = 1.0,
        period: float = 16.0,
        z: Optional[float] = None,
        offset_xy: Tuple[float, float] = (0.0, 0.0),
    ):
        z_use = self.default_z if z is None else float(z)
        self._ensure_takeoff_if_needed(z_use)
        ox, oy = float(offset_xy[0]), float(offset_xy[1])

        def _traj(local_t):
            pos, vel = Trajectories.square_traj(local_t, side=side, period=period, z=z_use)
            pos = pos.copy()
            pos[0] += ox
            pos[1] += oy
            return pos, vel

        return self.add_phase("square", duration, _traj)

    def add_hover(
        self,
        duration: float,
        z: Optional[float] = None,
        pos: Optional[Tuple[float, float, float]] = None,
    ):
        if pos is not None:
            hover_pos = np.asarray(pos, dtype=float)
            self._ensure_takeoff_if_needed(float(hover_pos[2]))
        else:
            hover_pos = self._terminal_pos.copy()
            if z is not None:
                z_use = float(z)
                self._ensure_takeoff_if_needed(z_use)
                hover_pos = self._terminal_pos.copy()
                hover_pos[2] = z_use

        def _traj(local_t):
            return Trajectories.hover_traj(local_t, pos=tuple(hover_pos))

        return self.add_phase("hover", duration, _traj)

    def add_point(
        self,
        duration: float,
        target: Tuple[float, float, float],
        profile: str = "min_jerk",
    ):
        """
        Move from current terminal position to an arbitrary 3D point.

        Works at mission start or between any other phases.
        """
        p1 = np.asarray(target, dtype=float).reshape(3)
        self._ensure_takeoff_if_needed(float(p1[2]))
        p0 = self._terminal_pos.copy()

        T = float(duration)
        if T <= 0.0:
            raise ValueError("Point-move duration must be > 0.")

        mode = str(profile).strip().lower()
        if np.linalg.norm(p1 - p0) <= 1e-9:
            return self.add_hover(duration=T, pos=tuple(p1))

        def _traj(local_t):
            tau = np.clip(float(local_t) / T, 0.0, 1.0)
            if mode == "min_jerk":
                s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
                ds_dt = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / T
            elif mode == "linear":
                s = tau
                ds_dt = 1.0 / T
            else:
                raise ValueError(f"Unknown point-move profile: {profile}")

            pos = (1.0 - s) * p0 + s * p1
            vel = ds_dt * (p1 - p0)
            return pos, vel

        return self.add_phase("point", T, _traj)

    def add_go_to(
        self,
        duration: float,
        target: Tuple[float, float, float],
        profile: str = "min_jerk",
    ):
        """Alias for add_point()."""
        return self.add_point(duration=duration, target=target, profile=profile)

    def add_landing(
        self,
        duration: float,
        ground_z: Optional[float] = None,
    ):
        gz = self.ground_z if ground_z is None else float(ground_z)
        p0 = self._terminal_pos.copy()
        p1 = p0.copy()
        p1[2] = gz

        def _traj(local_t):
            T = float(duration)
            s = np.clip(local_t / T, 0.0, 1.0)
            pos = (1.0 - s) * p0 + s * p1
            vel = (p1 - p0) / T
            return pos, vel

        return self.add_phase("landing", duration, _traj)

    @classmethod
    def circle_hover_square_hover_land(
        cls,
        circle_duration: float = 20.0,
        hover1_duration: float = 5.0,
        square_duration: float = 20.0,
        hover2_duration: float = 5.0,
        landing_duration: float = 6.0,
        circle_radius: float = 1.0,
        circle_period: float = 10.0,
        square_side: float = 1.0,
        square_period: float = 16.0,
        mission_z: float = 1.0,
        ground_z: float = 0.0,
        takeoff_duration: float = 3.0,
    ):
        """
        Convenience constructor:
        takeoff -> circle -> hover -> square -> hover -> landing.
        """
        mission = cls(default_z=mission_z, ground_z=ground_z)
        mission.add_takeoff(duration=takeoff_duration, target_z=mission_z)
        mission.add_circle(
            duration=circle_duration,
            radius=circle_radius,
            period=circle_period,
            z=mission_z,
        )
        mission.add_hover(duration=hover1_duration, z=mission_z)
        mission.add_square(
            duration=square_duration,
            side=square_side,
            period=square_period,
            z=mission_z,
        )
        mission.add_hover(duration=hover2_duration, z=mission_z)
        mission.add_landing(duration=landing_duration, ground_z=ground_z)
        return mission


# =========================================================
# Segment framework (for inserting takeoff/hover/landing)
# =========================================================
@dataclass
class TrajectorySegment:
    t_start: float
    t_end: float

    def contains(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class HoverSegment(TrajectorySegment):
    pos: np.ndarray

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        return self.pos.copy(), np.zeros(3, dtype=float)


@dataclass
class LandingSegment(TrajectorySegment):
    p0: np.ndarray
    ground_z: float = 0.0

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        T = (self.t_end - self.t_start)
        s = np.clip((t - self.t_start) / T, 0.0, 1.0)
        p1 = self.p0.copy()
        p1[2] = float(self.ground_z)
        pos = (1.0 - s) * self.p0 + s * p1
        vel = (p1 - self.p0) / T
        return pos, vel


@dataclass
class MinimumJerkTakeoffSegment(TrajectorySegment):
    p0: np.ndarray
    target_z: float

    def eval(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        T = (self.t_end - self.t_start)
        tau = np.clip((t - self.t_start) / T, 0.0, 1.0)

        # min-jerk scaling
        s = 10*tau**3 - 15*tau**4 + 6*tau**5
        ds_dt = (30*tau**2 - 60*tau**3 + 30*tau**4) / T

        z0 = float(self.p0[2])
        z1 = float(self.target_z)

        pos = self.p0.copy()
        pos[2] = z0 + s * (z1 - z0)

        vel = np.zeros(3, dtype=float)
        vel[2] = ds_dt * (z1 - z0)
        return pos, vel


# =========================================================
# Mission Planner trajectory
# =========================================================
class MissionPlannerTrajectory:
    """
    Mission trajectory loaded from:
      - Mission Planner JSON (.mission)
      - QGC WPL 110 text (.waypoints)

    Outputs (pos_ref, vel_ref) in LOCAL ENU meters:
      +x East, +y North, +z Up

    Supports time-shifting segment insertions:
      add_hover, add_takeoff_min_jerk, add_landing

    Enforces "no overlap" on inserted segments.
    """

    CMD_NAV_WAYPOINT = 16
    CMD_DO_CHANGE_SPEED = 178

    def __init__(
        self,
        path: str,
        total_time: Optional[float] = None,
        default_speed: float = 0.5
    ):
        if Transformer is None:
            raise ImportError(
                "pyproj is required for MissionPlannerTrajectory. Install with: pip install pyproj"
            )

        self.path = path
        self.total_time = None if total_time is None else float(total_time)
        self.default_speed = float(default_speed)

        # parsed content
        self.gps_points: np.ndarray = np.zeros((0, 3), dtype=float)
        self.speed_cmds: List[Tuple[int, float]] = []
        self.wp_indices: List[int] = []

        # trajectory in ENU
        self.points: np.ndarray = np.zeros((0, 3), dtype=float)
        self.times: np.ndarray = np.zeros((0,), dtype=float)

        # base mission (time-shifted after insertions)
        self.base_points: np.ndarray = np.zeros((0, 3), dtype=float)
        self.base_times: np.ndarray = np.zeros((0,), dtype=float)

        # inserted segments
        self.segments: List[TrajectorySegment] = []

        self._load_any()
        self._gps_to_enu()
        self._assign_times()

        self.base_points = self.points.copy()
        self.base_times = self.times.copy()

    # -------------------------
    # Public API
    # -------------------------
    def __call__(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        t = float(t)

        # inserted segments have priority
        for seg in self.segments:
            if seg.contains(t):
                return seg.eval(t)

        return self._eval_base(t)

    def summary(self) -> dict:
        self._sort_segments()
        return {
            "num_waypoints": int(len(self.points)),
            "base_mission_end_time": float(self.base_times[-1]) if len(self.base_times) else 0.0,
            "default_speed": float(self.default_speed),
            "speed_changes": list(self.speed_cmds),
            "segments": [
                {"type": type(s).__name__, "t_start": float(s.t_start), "t_end": float(s.t_end)}
                for s in self.segments
            ],
        }

    # -------------------------
    # Segment insertion
    # -------------------------
    def add_hover(self, t_start: float, duration: float):
        t_start = float(t_start)
        duration = float(duration)
        pos0, _ = self._eval_base(t_start)
        seg = HoverSegment(t_start=t_start, t_end=t_start + duration, pos=pos0)
        self.segments.append(seg)
        self._shift_base_after(t_start, duration)
        self._assert_no_overlap()

    def add_takeoff_min_jerk(self, t_start: float, duration: float, target_z: float):
        t_start = float(t_start)
        duration = float(duration)
        pos0, _ = self._eval_base(t_start)
        seg = MinimumJerkTakeoffSegment(
            t_start=t_start,
            t_end=t_start + duration,
            p0=pos0,
            target_z=float(target_z),
        )
        self.segments.append(seg)
        self._shift_base_after(t_start, duration)
        self._assert_no_overlap()

    def add_landing(self, t_start: float, duration: float, ground_z: float = 0.0):
        t_start = float(t_start)
        duration = float(duration)
        pos0, _ = self._eval_base(t_start)
        seg = LandingSegment(
            t_start=t_start,
            t_end=t_start + duration,
            p0=pos0,
            ground_z=float(ground_z),
        )
        self.segments.append(seg)
        self._shift_base_after(t_start, duration)
        self._assert_no_overlap()

    # -------------------------
    # Internal: parsing
    # -------------------------
    def _load_any(self):
        # Detect QGC WPL vs JSON
        with open(self.path, "r") as f:
            head = f.readline().strip()

        if head.startswith("QGC WPL"):
            self._load_qgc_wpl()
        else:
            self._load_json_mission()

        if len(self.gps_points) < 2:
            raise ValueError("Mission must contain at least two NAV_WAYPOINTs.")

    def _load_json_mission(self):
        with open(self.path, "r") as f:
            data = json.load(f)

        items = data["mission"]["items"]

        gps = []
        wp_indices = []
        speed_cmds = []

        for idx, item in enumerate(items):
            cmd = int(item["command"])
            p = item["params"]

            if cmd == self.CMD_NAV_WAYPOINT:
                lat, lon, alt = p[4:7]
                gps.append([lat, lon, alt])
                wp_indices.append(idx)

            elif cmd == self.CMD_DO_CHANGE_SPEED:
                speed = float(p[1])  # param2
                speed_cmds.append((idx, speed))

        self.gps_points = np.asarray(gps, dtype=float)
        self.wp_indices = wp_indices
        self.speed_cmds = speed_cmds

    def _load_qgc_wpl(self):
        gps = []
        wp_indices = []
        speed_cmds = []

        with open(self.path, "r") as f:
            lines = f.readlines()

        # line0 is header "QGC WPL 110"
        for idx, line in enumerate(lines[1:], start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                continue

            # format:
            # seq current frame cmd p1 p2 p3 p4 x(lat) y(lon) z(alt) autocontinue
            cmd = int(float(parts[3]))
            p1 = float(parts[4])
            p2 = float(parts[5])
            # p3 p4 not used

            lat = float(parts[8])
            lon = float(parts[9])
            alt = float(parts[10])

            if cmd == self.CMD_NAV_WAYPOINT:
                gps.append([lat, lon, alt])
                wp_indices.append(idx)

            elif cmd == self.CMD_DO_CHANGE_SPEED:
                speed_cmds.append((idx, p2))

        self.gps_points = np.asarray(gps, dtype=float)
        self.wp_indices = wp_indices
        self.speed_cmds = speed_cmds

    # -------------------------
    # Internal: conversion GPS -> ENU
    # -------------------------
    def _gps_to_enu(self):
        tf = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)

        lat0, lon0, alt0 = self.gps_points[0]
        x0, y0, z0 = tf.transform(lon0, lat0, alt0)

        lat0_rad = np.deg2rad(lat0)
        lon0_rad = np.deg2rad(lon0)

        sLat = np.sin(lat0_rad)
        cLat = np.cos(lat0_rad)
        sLon = np.sin(lon0_rad)
        cLon = np.cos(lon0_rad)

        # ECEF -> ENU rotation
        R = np.array([
            [-sLon,          cLon,         0.0],
            [-sLat * cLon,  -sLat * sLon,  cLat],
            [ cLat * cLon,   cLat * sLon,  sLat],
        ], dtype=float)

        enu = []
        for lat, lon, alt in self.gps_points:
            x, y, z = tf.transform(lon, lat, alt)
            d = np.array([x - x0, y - y0, z - z0], dtype=float)
            e, n, u = R @ d
            enu.append([e, n, u])

        self.points = np.asarray(enu, dtype=float)

    # -------------------------
    # Internal: timing
    # -------------------------
    def _assign_times(self):
        dists = np.linalg.norm(np.diff(self.points, axis=0), axis=1)

        if self.total_time is not None:
            L = float(np.sum(dists))
            if L < 1e-9:
                raise ValueError("Total path length is ~0; cannot allocate times.")
            times = [0.0]
            for d in dists:
                times.append(times[-1] + self.total_time * float(d) / L)
            self.times = np.asarray(times, dtype=float)
            return

        # Speed-based timing using DO_CHANGE_SPEED (best-effort)
        times = [0.0]
        current_speed = float(self.default_speed)
        speed_cmds = sorted(self.speed_cmds, key=lambda x: x[0])
        sc_idx = 0

        for i in range(1, len(self.points)):
            mission_idx = self.wp_indices[i]

            while sc_idx < len(speed_cmds) and speed_cmds[sc_idx][0] <= mission_idx:
                current_speed = float(speed_cmds[sc_idx][1])
                sc_idx += 1

            dt = float(dists[i - 1]) / max(current_speed, 1e-6)
            times.append(times[-1] + dt)

        self.times = np.asarray(times, dtype=float)

    def _eval_base(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        t = float(t)

        if t <= self.base_times[0]:
            return self.base_points[0].copy(), np.zeros(3, dtype=float)
        if t >= self.base_times[-1]:
            return self.base_points[-1].copy(), np.zeros(3, dtype=float)

        i = int(np.searchsorted(self.base_times, t) - 1)
        t0, t1 = self.base_times[i], self.base_times[i + 1]
        p0, p1 = self.base_points[i], self.base_points[i + 1]

        s = (t - t0) / (t1 - t0)
        pos = (1.0 - s) * p0 + s * p1
        vel = (p1 - p0) / (t1 - t0)
        return pos, vel

    # -------------------------
    # Internal: timeline integrity
    # -------------------------
    def _shift_base_after(self, t_start: float, delta_t: float):
        mask = self.base_times >= float(t_start)
        self.base_times[mask] += float(delta_t)

    def _sort_segments(self):
        self.segments.sort(key=lambda s: (s.t_start, s.t_end))

    def _assert_no_overlap(self):
        self._sort_segments()
        for i in range(len(self.segments) - 1):
            a = self.segments[i]
            b = self.segments[i + 1]
            if b.t_start < a.t_end:
                raise AssertionError(
                    f"Trajectory segments overlap:\n"
                    f"  {type(a).__name__} [{a.t_start:.3f}, {a.t_end:.3f}]\n"
                    f"  {type(b).__name__} [{b.t_start:.3f}, {b.t_end:.3f}]"
                )
