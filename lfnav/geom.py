"""Patch-center ray directions. FOV axis or a full pinhole K."""

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


def scale_intrinsics(intr: dict, height: int, width: int, aspect_tol: float = 0.03) -> dict:
    """Map a K captured at (intr height, width) onto the runtime image.

    Scale fx,fy,cx,cy by w/W and h/H when the aspect ratios match. Raise if they do not.
    """
    ih, iw = int(intr["height"]), int(intr["width"])
    if ih < 1 or iw < 1:
        raise ValueError("intrinsics width/height must be positive")
    src_aspect = iw / ih
    dst_aspect = width / height
    if abs(src_aspect - dst_aspect) > aspect_tol:
        raise ValueError(
            f"intrinsics {iw}x{ih} aspect {src_aspect:.4f} does not match "
            f"image {width}x{height} aspect {dst_aspect:.4f}"
        )
    sx = width / iw
    sy = height / ih
    return {
        "fx": float(intr["fx"]) * sx,
        "fy": float(intr["fy"]) * sy,
        "cx": float(intr["cx"]) * sx,
        "cy": float(intr["cy"]) * sy,
        "width": int(width),
        "height": int(height),
    }


def ray_directions_from_K(n_h: int, n_w: int, fx, fy, cx, cy, width: int, height: int):
    """Unproject the centre of each patch. Pixel (0,0) is the top-left corner."""
    rows = []
    for i in range(n_h):
        v = (i + 0.5) * height / n_h
        row = []
        for j in range(n_w):
            u = (j + 0.5) * width / n_w
            x = (u - float(cx)) / float(fx)
            y = (v - float(cy)) / float(fy)
            z = 1.0
            norm = math.sqrt(x * x + y * y + z * z)
            row.append((x / norm, y / norm, z / norm))
        rows.append(row)
    return rows


def rays_from_cfg(cfg: dict) -> list[list[tuple[float, float, float]]]:
    height, width = cfg.get("image_hw", [224, 364])
    height, width = int(height), int(width)
    n_h, n_w = patch_grid(height, width, int(cfg.get("patch", 14)))
    intr = cfg.get("intrinsics")
    if intr:
        k = scale_intrinsics(intr, height, width)
        return ray_directions_from_K(n_h, n_w, k["fx"], k["fy"], k["cx"], k["cy"], k["width"], k["height"])
    return ray_directions(n_h, n_w, float(cfg.get("fov_deg", 70)), str(cfg.get("fov_axis", "vertical")))
