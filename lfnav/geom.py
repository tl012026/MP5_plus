"""Patch-center ray directions. FOV axis is a config value; calibrate_camera.py fits it."""

from __future__ import annotations

import math


def patch_grid(height: int, width: int, patch: int = 14) -> tuple[int, int]:
    if height % patch or width % patch:
        raise ValueError(f"image {width}x{height} is not divisible by patch {patch}")
    return height // patch, width // patch


def ray_directions(n_h: int, n_w: int, fov_deg: float, fov_axis: str = "vertical") -> list[list[tuple[float, float, float]]]:
    """Camera looks along +z, +x is image right, +y is image down.

    fov_axis is "vertical" or "horizontal". The other axis follows the patch aspect ratio,
    which matches the image aspect ratio when patches are square.
    """
    if fov_axis not in ("vertical", "horizontal"):
        raise ValueError(f"fov_axis must be vertical or horizontal, got {fov_axis}")
    fov = math.radians(float(fov_deg))
    aspect = n_w / n_h
    if fov_axis == "vertical":
        fov_v = fov
        fov_h = 2.0 * math.atan(math.tan(fov_v / 2.0) * aspect)
    else:
        fov_h = fov
        fov_v = 2.0 * math.atan(math.tan(fov_h / 2.0) / aspect)
    rows = []
    for i in range(n_h):
        v = (i + 0.5) / n_h - 0.5
        pitch = v * fov_v
        row = []
        for j in range(n_w):
            u = (j + 0.5) / n_w - 0.5
            yaw = u * fov_h
            x = math.tan(yaw)
            y = math.tan(pitch)
            z = 1.0
            norm = math.sqrt(x * x + y * y + z * z)
            row.append((x / norm, y / norm, z / norm))
        rows.append(row)
    return rows


def center_ray(n_h: int, n_w: int, fov_deg: float, fov_axis: str, row: int, col: int) -> tuple[float, float, float]:
    return ray_directions(n_h, n_w, fov_deg, fov_axis)[row][col]
