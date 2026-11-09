"""Synthetic K recovery, yaw sign, and K vs fov rays. No MineDojo."""

from __future__ import annotations

import math
import unittest

import numpy as np

from lfnav.env.minedojo_env import aligned_lidar_rays
from lfnav.geom import ray_directions, ray_directions_from_K, scale_intrinsics
from lfnav.gt.camera import (
    hud_mask,
    initial_K,
    match_numpy,
    optimize_K,
    params_to_K,
    project_points,
    ransac_inliers,
    relative_rotation,
    valid_mask,
    world_to_camera,
)


def _pano(h=240, w=480):
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (90 + 70 * np.sin(xx * 0.17) + 40 * np.sin(yy * 0.23)).astype(np.uint8)
    img[..., 1] = (80 + 90 * np.sin(xx * 0.05 + yy * 0.09) + 30 * np.cos(xx * 0.31)).astype(np.uint8)
    img[..., 2] = ((xx * 3 + yy * 7) % 180 + 40).astype(np.uint8)
    img[(yy // 12 + xx // 16) % 2 == 0] = np.clip(img[(yy // 12 + xx // 16) % 2 == 0] + 40, 0, 255)
    return img


def _bilinear(pano, u, v):
    h, w = pano.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    au, av = u - x0, v - y0
    c00 = pano[y0, x0].astype(np.float64)
    c10 = pano[y0, x1].astype(np.float64)
    c01 = pano[y1, x0].astype(np.float64)
    c11 = pano[y1, x1].astype(np.float64)
    top = c00 * (1 - au)[:, None] + c10 * au[:, None]
    bot = c01 * (1 - au)[:, None] + c11 * au[:, None]
    return (top * (1 - av)[:, None] + bot * av[:, None]).astype(np.uint8)


def render_sized(pano, K, yaw, pitch, height, width, noise=2.5, hud=True):
    yy, xx = np.mgrid[0:height, 0:width]
    u = xx.astype(np.float64) + 0.5
    v = yy.astype(np.float64) + 0.5
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    z = np.ones_like(x)
    d = np.stack([x, y, z], -1)
    d = d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-9)
    r = world_to_camera(yaw, pitch)
    world = (r.T @ d.reshape(-1, 3).T).T
    wx, wy, wz = world[:, 0], world[:, 1], world[:, 2]
    lon = np.arctan2(wx, wz)
    lat = np.arctan2(wy, np.hypot(wx, wz))
    pu = (lon / math.pi + 1.0) * 0.5 * (pano.shape[1] - 1)
    pv = (lat / (math.pi / 2) + 1.0) * 0.5 * (pano.shape[0] - 1)
    rgb = _bilinear(pano, pu, pv).reshape(height, width, 3)
    if hud:
        mask = hud_mask(height, width)
        rgb = rgb.copy()
        rgb[~mask] = np.array([40, 40, 40], np.uint8)
        rgb[~mask] = (rgb[~mask] * 0 + np.array([30, 90, 40], np.uint8))
        cy, cx = height // 2, width // 2
        rgb[cy, cx - 3 : cx + 4] = (220, 220, 220)
        rgb[cy - 3 : cy + 4, cx] = (220, 220, 220)
    rng = np.random.default_rng(4)
    rgb = np.clip(rgb.astype(np.float64) + rng.normal(0, noise, rgb.shape), 0, 255).astype(np.uint8)
    return rgb


def recover(K_true, height, width, poses, pano=None):
    pano = pano if pano is not None else _pano()
    frames = [(yaw, pitch, render_sized(pano, K_true, yaw, pitch, height, width)) for yaw, pitch in poses]
    k0 = initial_K(height, width, 70.0)
    calib = {"bottom_frac": 0.18, "crosshair_frac": 0.03, "hand_width_frac": 0.22, "hand_height_frac": 0.28, "grad_min": 4.0}
    pairs = []
    for i, (y1, p1, im1) in enumerate(frames):
        for y2, p2, im2 in frames[i + 1 : i + 3]:
            dy = abs(((y1 - y2 + 180) % 360) - 180)
            dp = abs(p1 - p2)
            if not (8.0 <= max(dy, dp) <= 40.0):
                continue
            r = relative_rotation(y1, p1, y2, p2)
            mask = valid_mask(im1, calib)
            src, dst = match_numpy(im1, im2, r, k0, mask, patch=6, window=8, max_pts=70)
            if len(src) < 12:
                continue
            keep = ransac_inliers(src, dst, k0, r)
            if keep.sum() < 8:
                continue
            pairs.append((src[keep], dst[keep], r))
    k_fit, _rms = optimize_K(pairs, k0, width, height)
    return k_fit, pairs, len(pairs) >= 2


class YawSignTest(unittest.TestCase):
    def test_right_yaw_moves_principal_point_left(self):
        h, w = 100, 160
        k = initial_K(h, w, 70)
        r = relative_rotation(0, 0, 15, 0)
        x2 = project_points(k, r, [[w / 2, h / 2]])[0]
        self.assertLess(x2[0], w / 2)
        self.assertAlmostEqual(x2[1], h / 2, delta=1.5)


class RayKVsFovTest(unittest.TestCase):
    def test_centered_square_pixels_match_vertical_fov(self):
        h, w, patch = 224, 364, 14
        n_h, n_w = h // patch, w // patch
        fy = (h / 2.0) / math.tan(math.radians(70) / 2.0)
        fx = fy
        cx, cy = w / 2.0, h / 2.0
        from_k = ray_directions_from_K(n_h, n_w, fx, fy, cx, cy, w, h)
        from_fov = ray_directions(n_h, n_w, 70, "vertical")
        for i in range(n_h):
            for j in range(n_w):
                a = np.array(from_k[i][j])
                b = np.array(from_fov[i][j])
                self.assertGreater(float(np.dot(a, b)), 0.97)

    def test_aligned_lidar_rays_same_length(self):
        cfg = {"image_hw": [224, 364], "patch": 14, "fov_deg": 70, "fov_axis": "vertical", "aligned_ray_range": 16}
        h, w = 224, 364
        fy = (h / 2.0) / math.tan(math.radians(70) / 2.0)
        cfg_k = dict(cfg)
        cfg_k["intrinsics"] = {"fx": fy, "fy": fy, "cx": w / 2, "cy": h / 2, "width": w, "height": h}
        a = aligned_lidar_rays(cfg)
        b = aligned_lidar_rays(cfg_k)
        self.assertEqual(len(a), len(b))
        self.assertEqual(len(a), (224 // 14) * (364 // 14))
        for ra, rb in zip(a, b):
            self.assertAlmostEqual(ra[2], rb[2])
            self.assertLess(abs(ra[0] - rb[0]) + abs(ra[1] - rb[1]), 0.12)

    def test_aspect_mismatch_raises(self):
        intr = {"fx": 100, "fy": 100, "cx": 50, "cy": 50, "width": 100, "height": 100}
        with self.assertRaises(ValueError):
            scale_intrinsics(intr, 224, 364)


class SyntheticFitTest(unittest.TestCase):
    def _assert_close(self, k_fit, k_true, pixel=2.0, frac=0.01):
        fx, fy = k_true[0, 0], k_true[1, 1]
        self.assertLess(abs(k_fit[0, 0] - fx) / fx, frac)
        self.assertLess(abs(k_fit[1, 1] - fy) / fy, frac)
        self.assertLess(abs(k_fit[0, 2] - k_true[0, 2]), pixel)
        self.assertLess(abs(k_fit[1, 2] - k_true[1, 2]), pixel)

    def test_square_pixels(self):
        h, w = 72, 96
        k = initial_K(h, w, 70)
        poses = [(0, 0), (15, 0), (30, 0), (-15, 0), (0, 15), (15, 15)]
        k_fit, pairs, ok = recover(k, h, w, poses)
        self.assertTrue(ok)
        self.assertGreaterEqual(len(pairs), 2)
        self._assert_close(k_fit, k)

    def test_fx_not_fy(self):
        h, w = 72, 96
        fy = (h / 2.0) / math.tan(math.radians(70) / 2.0)
        k = params_to_K([fy * 1.12, fy, w / 2 + 1.5, h / 2 - 1.0], w, h)
        poses = [(0, 0), (15, 0), (30, 0), (-15, 0), (0, 15), (15, -15)]
        k_fit, pairs, ok = recover(k, h, w, poses)
        self.assertTrue(ok)
        self.assertGreaterEqual(len(pairs), 2)
        self._assert_close(k_fit, k)

    def test_yaw_while_pitched(self):
        h, w = 72, 96
        k = initial_K(h, w, 70)
        poses = [(0, 15), (15, 15), (30, 15), (-15, 15), (0, 0), (0, 30)]
        k_fit, pairs, ok = recover(k, h, w, poses)
        self.assertTrue(ok)
        self.assertGreaterEqual(len(pairs), 2)
        self._assert_close(k_fit, k)


if __name__ == "__main__":
    unittest.main()
