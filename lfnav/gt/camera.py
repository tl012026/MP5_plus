"""Pinhole camera for Minecraft-style yaw/pitch.

Camera frame (same as lfnav.gt.fov): +x right, +y down, +z forward.
Minecraft: yaw increases when turning right; pitch positive looks down.
World Y is up. World-to-camera is FlipY, then yaw about world up (so a right
turn sends the previous principal point to the left of the image), then pitch
about camera x (looking down sends the previous center toward the top of the
image, negative y).

Relative rotation of two poses: R_rel = R2 @ R1.T maps camera-1 rays into
camera-2, so x2 ~ K R_rel K^{-1} x1 for points at infinity.
"""

from __future__ import annotations

import math

import numpy as np


def world_to_camera(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    # Right yaw: world +Z maps to camera x < 0 (center moves left).
    r_yaw = np.array([[cy, 0.0, -sy], [0.0, 1.0, 0.0], [sy, 0.0, cy]], np.float64)
    flip_y = np.diag([1.0, -1.0, 1.0])
    # Positive pitch looks down: world +Z maps to camera y < 0.
    r_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], np.float64)
    return r_pitch @ flip_y @ r_yaw


def relative_rotation(yaw1, pitch1, yaw2, pitch2) -> np.ndarray:
    r1 = world_to_camera(yaw1, pitch1)
    r2 = world_to_camera(yaw2, pitch2)
    return r2 @ r1.T


def initial_K(height: int, width: int, fov_v_deg: float = 70.0) -> np.ndarray:
    fy = (float(height) / 2.0) / math.tan(math.radians(float(fov_v_deg)) / 2.0)
    fx = fy
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]], np.float64)


def params_to_K(params, width: int, height: int) -> np.ndarray:
    fx, fy, cx, cy = [float(v) for v in params]
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)


def K_to_params(K) -> np.ndarray:
    K = np.asarray(K, np.float64)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float64)


def fov_from_K(K, height: int, width: int) -> dict:
    fx, fy, cx, cy = K_to_params(K)
    fov_v = math.degrees(2.0 * math.atan((height / 2.0) / fy))
    fov_h = math.degrees(2.0 * math.atan((width / 2.0) / fx))
    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "fov_v_deg": float(fov_v),
        "fov_h_deg": float(fov_h),
        "fx_over_fy": float(fx / fy),
        "cx_offset": float(cx - width / 2.0),
        "cy_offset": float(cy - height / 2.0),
        "width": int(width),
        "height": int(height),
    }


def project_points(K, R, xy1) -> np.ndarray:
    """Project image points xy1 (N,2) through H = K R K^{-1}."""
    K = np.asarray(K, np.float64)
    R = np.asarray(R, np.float64)
    pts = np.asarray(xy1, np.float64)
    ones = np.ones((len(pts), 1), np.float64)
    xh = np.concatenate([pts, ones], 1)
    mapped = (K @ R @ np.linalg.inv(K) @ xh.T).T
    mapped = mapped[:, :2] / np.maximum(mapped[:, 2:3], 1e-9)
    return mapped


def hud_mask(height: int, width: int, calib: dict | None = None) -> np.ndarray:
    calib = calib or {}
    bottom = float(calib.get("bottom_frac", 0.18))
    cross = float(calib.get("crosshair_frac", 0.03))
    hand_w = float(calib.get("hand_width_frac", 0.22))
    hand_h = float(calib.get("hand_height_frac", 0.28))
    mask = np.ones((height, width), bool)
    y0 = int((1.0 - bottom) * height)
    mask[y0:, :] = False
    cx, cy = width // 2, height // 2
    r = max(2, int(cross * min(height, width)))
    mask[cy - r:cy + r + 1, cx - r:cx + r + 1] = False
    x0 = int((1.0 - hand_w) * width)
    y1 = int((1.0 - hand_h) * height)
    mask[y1:, x0:] = False
    return mask


def gradient_mask(rgb: np.ndarray, grad_min: float = 8.0) -> np.ndarray:
    gray = rgb.astype(np.float64).mean(-1)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
    gy[1:, :] = gray[1:, :] - gray[:-1, :]
    return np.hypot(gx, gy) >= float(grad_min)


def valid_mask(rgb: np.ndarray, calib: dict | None = None) -> np.ndarray:
    h, w = rgb.shape[:2]
    calib = calib or {}
    return hud_mask(h, w, calib) & gradient_mask(rgb, float(calib.get("grad_min", 8.0)))


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return -1.0
    return float(np.dot(a, b) / denom)


def match_numpy(rgb1, rgb2, R, K, mask1, patch: int = 8, window: int = 10, max_pts: int = 180):
    """NCC around the location predicted by K0 and R_rel."""
    h, w = rgb1.shape[:2]
    g1 = rgb1.astype(np.float64).mean(-1)
    g2 = rgb2.astype(np.float64).mean(-1)
    ys, xs = np.where(mask1)
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(xs), size=min(max_pts * 3, len(xs)), replace=False)
    src, dst = [], []
    half = patch
    for idx in pick:
        x, y = int(xs[idx]), int(ys[idx])
        if x < half + window or y < half + window or x >= w - half - window or y >= h - half - window:
            continue
        pred = project_points(K, R, [[x, y]])[0]
        px, py = int(round(pred[0])), int(round(pred[1]))
        if px < half + window or py < half + window or px >= w - half - window or py >= h - half - window:
            continue
        templ = g1[y - half:y + half, x - half:x + half]
        best, bx, by = -1.0, px, py
        for dy in range(-window, window + 1, 2):
            for dx in range(-window, window + 1, 2):
                yy, xx = py + dy, px + dx
                crop = g2[yy - half:yy + half, xx - half:xx + half]
                if crop.shape != templ.shape:
                    continue
                score = _ncc(templ, crop)
                if score > best:
                    best, bx, by = score, xx, yy
        if best > 0.45:
            src.append([x, y])
            dst.append([bx, by])
        if len(src) >= max_pts:
            break
    return np.asarray(src, np.float64), np.asarray(dst, np.float64)


def match_opencv(rgb1, rgb2, mask1):
    import cv2
    g1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY)
    m = mask1.astype(np.uint8) * 255
    if hasattr(cv2, "SIFT_create"):
        det = cv2.SIFT_create(nfeatures=800)
    else:
        det = cv2.ORB_create(nfeatures=800)
    k1, d1 = det.detectAndCompute(g1, m)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    if d1.dtype != np.uint8:
        bf = cv2.BFMatcher(cv2.NORM_L2)
        pairs = bf.knnMatch(d1, d2, k=2)
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = bf.match(d1, d2)
        good = sorted(good, key=lambda m: m.distance)[:400]
    src = np.array([k1[m.queryIdx].pt for m in good], np.float64)
    dst = np.array([k2[m.trainIdx].pt for m in good], np.float64)
    return src, dst


def match_pair(rgb1, rgb2, R, K, calib, use_opencv: bool = True):
    mask = valid_mask(rgb1, calib)
    if use_opencv:
        try:
            import cv2  # noqa: F401
            src, dst = match_opencv(rgb1, rgb2, mask)
            if len(src) >= 12:
                return src, dst
        except Exception:
            pass
    return match_numpy(rgb1, rgb2, R, K, mask)


def ransac_inliers(src, dst, K, R, thresh: float = 3.0, iters: int = 80):
    """Keep matches consistent with H = K0 R K0^{-1}."""
    if len(src) < 8:
        return np.zeros(len(src), bool)
    pred = project_points(K, R, src)
    err = np.linalg.norm(pred - dst, axis=1)
    best = err < float(thresh)
    rng = np.random.default_rng(1)
    n = len(src)
    for _ in range(iters):
        idx = rng.choice(n, size=min(12, n), replace=False)
        pred = project_points(K, R, src)
        # K is fixed in this pass; inliers are just the threshold set. The loop
        # exists so a slightly looser band around the median error can recover
        # a pair that is mostly good.
        med = float(np.median(err[idx])) if len(idx) else float(thresh)
        cand = err < max(float(thresh), 2.5 * med)
        if cand.sum() > best.sum():
            best = cand
    return best


def huber_weights(r, delta: float = 2.0) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r)
    big = a > delta
    w[big] = delta / np.maximum(a[big], 1e-9)
    return w


def residuals(params, pairs, width, height):
    K = params_to_K(params, width, height)
    chunks = []
    for src, dst, R in pairs:
        pred = project_points(K, R, src)
        chunks.append((pred - dst).ravel())
    if not chunks:
        return np.zeros(0)
    return np.concatenate(chunks)


def jacobian(params, pairs, width, height, eps: float = 0.5):
    base = residuals(params, pairs, width, height)
    jac = np.zeros((len(base), 4), np.float64)
    for i in range(4):
        step = np.array(params, np.float64)
        step[i] += eps
        jac[:, i] = (residuals(step, pairs, width, height) - base) / eps
    return jac, base


def optimize_K(pairs, K0, width, height, iters: int = 12):
    params = K_to_params(K0)
    try:
        from scipy.optimize import least_squares
        result = least_squares(
            lambda p: residuals(p, pairs, width, height),
            params,
            loss="huber",
            f_scale=2.0,
            max_nfev=80,
        )
        return params_to_K(result.x, width, height), float(np.sqrt(np.mean(result.fun ** 2))) if len(result.fun) else None
    except Exception:
        pass
    rms = None
    for _ in range(iters):
        jac, r = jacobian(params, pairs, width, height)
        if len(r) == 0:
            break
        w = huber_weights(r)
        jw = jac * w[:, None]
        rw = r * w
        try:
            delta = np.linalg.lstsq(jw, -rw, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        params = params + np.clip(delta, -40, 40)
        params[0] = max(params[0], 10.0)
        params[1] = max(params[1], 10.0)
        rms = float(np.sqrt(np.mean(r ** 2)))
    return params_to_K(params, width, height), rms


def suggest_config(report: dict) -> dict:
    fx, fy = report["fx"], report["fy"]
    width, height = report["width"], report["height"]
    square = abs(fx / fy - 1.0) < 0.01
    centered = abs(report["cx_offset"]) < 0.01 * width and abs(report["cy_offset"]) < 0.01 * height
    if square and centered:
        fov = round(report["fov_v_deg"], 1)
        return {
            "mode": "fov",
            "fov_axis": "vertical",
            "fov_deg": fov,
            "note": f"square pixels and centered principal point; set fov_deg={fov} fov_axis=vertical",
        }
    return {
        "mode": "intrinsics",
        "intrinsics": {k: report[k] for k in ("fx", "fy", "cx", "cy", "width", "height")},
        "note": "pixels are not square or the principal point is off-center; store fx,fy,cx,cy and use ray_directions_from_K",
    }
