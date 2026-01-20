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
