import numpy as np
from trajectories_library import Trajectories as T

class TrajectoryManager:
    def __init__(self, default="hover"):
        self.active = default
        self.start_time = 0.0   # local time reset on switching

    # ---------------------------------------------------------
    # Switch active trajectory at runtime
    # ---------------------------------------------------------
    def switch(self, name, now):
        self.active = name
        self.start_time = now

    # ---------------------------------------------------------
    # Fetch reference from your trajectory library
    # ---------------------------------------------------------
    def get_reference(self, now):
        # Local time inside this trajectory segment
        t = now - self.start_time

        # === Hover (constant position) ===
        if self.active == "hover":
            pos, vel = T.hover_traj(t)

        # === Static point (waypoint) ===
        elif self.active == "point":
            pos, vel = T.point_traj()

        # === Square ===
        elif self.active == "square":
            pos, vel = T.square_traj(t)

        # === Circle ===
        elif self.active == "circle":
            pos, vel = T.circle_traj(t)

        # === Helix ===
        elif self.active == "helix":
            pos, vel = T.helix_traj(t)

        # === Sine wave ===
        elif self.active == "sine":
            pos, vel = T.sine_traj(t)

        else:
            # fallback = hover
            pos, vel = T.hover_traj(t)

        # (pos = [x,y,z], vel = [vx,vy,vz])
        return pos, vel
