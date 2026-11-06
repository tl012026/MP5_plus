"""Fit a pinhole field of view from one known angle. The axis is chosen by the caller."""

from __future__ import annotations

import math


def fov_deg_from_pixel(pixel: float, size: int, angle_deg: float) -> float:
    """angle_deg is the ray angle from the optical axis that landed on this pixel.

    u = (pixel + 0.5) / size - 0.5, and u = 0.5 * tan(angle) / tan(fov/2).
    """
    u = (float(pixel) + 0.5) / float(size) - 0.5
    if abs(u) < 1e-4:
        raise ValueError("a pixel on the optical axis does not determine the field of view")
    half = math.atan(math.tan(math.radians(angle_deg)) / (2.0 * u))
    return math.degrees(2.0 * half)


def ray_angles_deg(direction) -> tuple[float, float]:
    """Return (yaw_deg, pitch_deg) for a camera-frame unit ray. +x right, +y down, +z forward."""
    x, y, z = direction
    yaw = math.degrees(math.atan2(x, z))
    pitch = math.degrees(math.atan2(y, math.hypot(x, z)))
    return yaw, pitch
