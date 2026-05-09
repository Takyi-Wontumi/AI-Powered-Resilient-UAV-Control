"""
Motor-mapping layer for applying a user-selected actuator layout on top of the
controller's body-level thrust/torque commands.

There are two ways to define a mapping:

1. Easy corner shorthand
   Example: "M1=FR,M2=RR,M3=RL,M4=FL"
   Use this when the frame is a standard X quad and you only want to say which
   named motor lives at which corner.

2. Full explicit geometry
   Example:
   "M1:0.707:0.707:1,M2:0.707:-0.707:-1,M3:-0.707:-0.707:1,M4:-0.707:0.707:-1"
   Use this when you want direct control over each motor's position and yaw
   direction contribution.
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


SQRT_HALF = 2.0 ** -0.5
# Standard X-layout corner definitions used by the simple shorthand parser.
# Each value is:
#   (x_right, y_front, yaw_sign)
# where yaw_sign is the signed drag contribution for that motor.
CORNER_TO_CHANNEL = {
    "FR": (SQRT_HALF, SQRT_HALF, 1.0),
    "FRONT_RIGHT": (SQRT_HALF, SQRT_HALF, 1.0),
    "RR": (SQRT_HALF, -SQRT_HALF, -1.0),
    "REAR_RIGHT": (SQRT_HALF, -SQRT_HALF, -1.0),
    "RL": (-SQRT_HALF, -SQRT_HALF, 1.0),
    "REAR_LEFT": (-SQRT_HALF, -SQRT_HALF, 1.0),
    "FL": (-SQRT_HALF, SQRT_HALF, -1.0),
    "FRONT_LEFT": (-SQRT_HALF, SQRT_HALF, -1.0),
}


@dataclass(frozen=True)
class MotorChannel:
    # Human-readable channel label used in logs and diagnostics.
    name: str
    # Positive values mean the motor is on the right side of the frame.
    x_right: float
    # Positive values mean the motor is toward the front of the frame.
    y_front: float
    # Signed yaw contribution, typically +1 or -1 depending on spin direction.
    yaw_sign: float


class MotorMapLayer:
    def __init__(self, channels: Iterable[MotorChannel], thrust_coeff: float, drag_coeff: float, arm_length: float):
        self.channels = tuple(channels)
        if len(self.channels) != 4:
            raise ValueError("MotorMapLayer requires exactly 4 motor channels")

        self.thrust_coeff = float(thrust_coeff)
        self.drag_coeff = float(drag_coeff)
        self.arm_length = float(arm_length)
        self._allocation_matrix = self._build_allocation_matrix()
        self._allocation_inv = np.linalg.inv(self._allocation_matrix)

    def _build_allocation_matrix(self) -> np.ndarray:
        # Map per-motor squared speeds into:
        #   total thrust, roll torque, pitch torque, yaw torque
        matrix = np.zeros((4, 4), dtype=float)
        for idx, channel in enumerate(self.channels):
            matrix[0, idx] = self.thrust_coeff
            matrix[1, idx] = -channel.x_right * self.arm_length * self.thrust_coeff
            matrix[2, idx] = -channel.y_front * self.arm_length * self.thrust_coeff
            matrix[3, idx] = channel.yaw_sign * self.drag_coeff
        return matrix

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    def mixer(self, thrust_cmd: float, tau_cmd) -> np.ndarray:
        # Re-allocate the controller's body-level command into motor speeds for
        # the user-selected layout. This sits on top of the controller instead
        # of changing the controller itself.
        u = np.array(
            [
                float(thrust_cmd),
                float(tau_cmd[0]),
                float(tau_cmd[1]),
                float(tau_cmd[2]),
            ],
            dtype=float,
        )
        omega_sq = self._allocation_inv @ u
        omega_sq = np.clip(omega_sq, 0.0, np.inf)
        return np.sqrt(omega_sq)

    def motor_forces(self, omega_cmd) -> np.ndarray:
        omega_cmd = np.asarray(omega_cmd, dtype=float)
        return self.thrust_coeff * omega_cmd ** 2

    def map_control(self, ctrl: dict) -> dict:
        if "tau_cmd" in ctrl:
            omega_cmd = self.mixer(ctrl["thrust_cmd"], ctrl["tau_cmd"])
            motor_forces = self.motor_forces(omega_cmd)
            tau_cmd = np.asarray(ctrl["tau_cmd"], dtype=float).copy()
        elif "omega_cmd" in ctrl:
            # Fall back to controller-native motor outputs when body torque
            # commands are unavailable in older controller variants.
            omega_cmd = np.asarray(ctrl["omega_cmd"], dtype=float).copy()
            if omega_cmd.shape != (4,):
                raise ValueError("control dict 'omega_cmd' must have shape (4,)")
            motor_forces = (
                np.asarray(ctrl["motor_forces"], dtype=float).copy()
                if "motor_forces" in ctrl
                else self.motor_forces(omega_cmd)
            )
            tau_cmd = None
        else:
            raise KeyError(
                "control dict must include 'tau_cmd' for remapping, or "
                "'omega_cmd' to log controller-native motor outputs"
            )

        return {
            "channel_names": self.channel_names,
            "omega_cmd": omega_cmd,
            "motor_forces": motor_forces,
            "thrust_cmd": float(ctrl["thrust_cmd"]),
            "tau_cmd": tau_cmd,
        }


def internal_plus_channels() -> tuple[MotorChannel, ...]:
    return (
        MotorChannel("w1/front", 0.0, 1.0, 1.0),
        MotorChannel("w2/right", 1.0, 0.0, -1.0),
        MotorChannel("w3/rear", 0.0, -1.0, 1.0),
        MotorChannel("w4/left", -1.0, 0.0, -1.0),
    )


def crazyflie_x_channels() -> tuple[MotorChannel, ...]:
    return (
        MotorChannel("M1/front_right", SQRT_HALF, SQRT_HALF, 1.0),
        MotorChannel("M2/rear_right", SQRT_HALF, -SQRT_HALF, -1.0),
        MotorChannel("M3/rear_left", -SQRT_HALF, -SQRT_HALF, 1.0),
        MotorChannel("M4/front_left", -SQRT_HALF, SQRT_HALF, -1.0),
    )


def parse_motor_map_spec(spec: str) -> tuple[MotorChannel, ...]:
    # Support both user-facing formats with one entry point:
    #   1. "M1=FR,M2=RR,..." shorthand
    #   2. "M1:0.707:0.707:1,..." explicit geometry
    if "=" in spec and ":" not in spec:
        return parse_motor_corner_spec(spec)

    channels = []
    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            name, x_right, y_front, yaw_sign = entry.split(":")
        except ValueError as exc:
            raise ValueError(
                "motor map spec entries must look like "
                "'name:x_right:y_front:yaw_sign'"
            ) from exc
        channels.append(
            MotorChannel(
                name=name.strip(),
                x_right=float(x_right),
                y_front=float(y_front),
                yaw_sign=float(yaw_sign),
            )
        )

    if len(channels) != 4:
        raise ValueError("motor map spec must define exactly 4 motor channels")

    return tuple(channels)


def parse_motor_corner_spec(spec: str) -> tuple[MotorChannel, ...]:
    # Easy format for standard X quads:
    #   M1=FR,M2=RR,M3=RL,M4=FL
    # The corner name is expanded into the corresponding coordinates and yaw
    # sign using CORNER_TO_CHANNEL above.
    channels = []
    used_corners = set()

    for raw_entry in spec.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            name, corner = entry.split("=")
        except ValueError as exc:
            raise ValueError(
                "corner motor map entries must look like 'name=FR' or 'M4=FL'"
            ) from exc

        corner_key = corner.strip().upper()
        if corner_key not in CORNER_TO_CHANNEL:
            raise ValueError(
                "unknown corner label "
                f"'{corner.strip()}'. Use one of FL, FR, RL, RR"
            )
        if corner_key in used_corners:
            raise ValueError(f"corner '{corner.strip()}' was assigned more than once")
        used_corners.add(corner_key)

        x_right, y_front, yaw_sign = CORNER_TO_CHANNEL[corner_key]
        channels.append(
            MotorChannel(
                name=name.strip(),
                x_right=x_right,
                y_front=y_front,
                yaw_sign=yaw_sign,
            )
        )

    if len(channels) != 4:
        raise ValueError("corner motor map spec must define exactly 4 motor channels")

    return tuple(channels)


def make_motor_map_layer(controller, preset: str = "internal_plus", spec: Optional[str] = None) -> MotorMapLayer:
    # Priority order:
    #   1. explicit user spec if provided
    #   2. named preset otherwise
    if spec:
        channels = parse_motor_map_spec(spec)
    elif preset == "internal_plus":
        channels = internal_plus_channels()
    elif preset == "crazyflie_x":
        channels = crazyflie_x_channels()
    else:
        raise ValueError(f"unknown motor map preset: {preset}")

    return MotorMapLayer(
        channels=channels,
        thrust_coeff=controller.b,
        drag_coeff=controller.d,
        arm_length=controller.l,
    )
