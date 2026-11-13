"""Frozen image encoders. DINOv2 and MineCLIP load only when requested.

The fake encoder is the stand-in used by --dry-run and the login-node tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DINOV2_HUB_NAME = "facebookresearch_dinov2_main"
DINOV2_GITHUB = "facebookresearch/dinov2:main"


class FakeEncoder:
    name = "fake"

    def __init__(self, dim: int = 32, grid: tuple[int, int] = (2, 3)):
        self.dim = dim
        self.grid = grid

    def __call__(self, rgb) -> dict:
        image = np.asarray(rgb, dtype=np.float32)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        mean = float(image.mean()) if image.size else 0.0
        cls = np.full(self.dim, mean / 255.0, dtype=np.float32)
        gh, gw = self.grid
        tokens = np.zeros((gh, gw, self.dim), dtype=np.float32)
        tokens[..., 0] = mean / 255.0
        pooled = tokens.mean(axis=(0, 1))
        return {"cls": cls, "tokens": tokens, "pooled": pooled}


def load_encoder(name: str, cfg: dict | None = None):
    cfg = cfg or {}
    if name in ("fake", "dry-run"):
        return FakeEncoder()
    if name.startswith("dinov2"):
        return _dinov2(name)
    if name == "mineclip":
        return _mineclip(cfg)
    raise KeyError(name)


def dinov2_local_repo():
    """$TORCH_HOME/hub/facebookresearch_dinov2_main, or torch.hub.get_dir() if TORCH_HOME is unset."""
    import torch

    home = os.environ.get("TORCH_HOME")
    if home:
        path = Path(home) / "hub" / DINOV2_HUB_NAME
    else:
        path = Path(torch.hub.get_dir()) / DINOV2_HUB_NAME
    if path.is_dir():
        return path
    return None


def load_dinov2_backbone(entry: str):
    import torch

    local = dinov2_local_repo()
    if local is not None:
        return torch.hub.load(str(local), entry, source="local")
    return torch.hub.load(DINOV2_GITHUB, entry, skip_validation=True)


def _dinov2(name: str):
    hub = {"dinov2_vits14": "dinov2_vits14", "dinov2_vitb14": "dinov2_vitb14"}[name]
    model = load_dinov2_backbone(hub)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    encoder_name = name

    class _Wrap:
        patch = 14

        def __init__(self):
            self.model = model

        def to(self, device):
            self.model.to(device)
            return self

        def __call__(self, rgb):
            tensor = _as_tensor(rgb)
            device = next(self.model.parameters()).device
            tensor = tensor.to(device)
            with torch.no_grad():
                feats = self.model.forward_features(tensor)
            cls = feats["x_norm_clstoken"][0].detach().cpu().numpy()
            tokens = feats["x_norm_patchtokens"][0].detach().cpu().numpy()
            _, _, height, width = tensor.shape
            grid = reshape_patch_tokens(tokens, height, width, self.patch)
            pooled = grid.mean(axis=(0, 1))
            return {"cls": cls, "tokens": grid, "pooled": pooled}

    _Wrap.name = encoder_name
    return _Wrap()


def _mineclip(cfg: dict):
    path = (cfg.get("perception") or {}).get("mineclip_dir")
    raise RuntimeError(
        f"MineCLIP weights were not wired. Expected a local checkout under {path}. "
        "The login-node tests use the fake encoder. Skipping MineCLIP is recorded in REPORT.md."
    )


def reshape_patch_tokens(tokens, height: int, width: int, patch: int = 14):
    """(N, D) patch tokens → (H/patch, W/patch, D). 224×364 / 14 is 16×26, not square."""
    tokens = np.asarray(tokens)
    n_h, n_w = int(height) // int(patch), int(width) // int(patch)
    if tokens.ndim != 2:
        raise ValueError(f"expected (N, D) tokens, got {tokens.shape}")
    if tokens.shape[0] != n_h * n_w:
        raise ValueError(f"token count {tokens.shape[0]} != {n_h}x{n_w} for {height}x{width} patch {patch}")
    return tokens.reshape(n_h, n_w, tokens.shape[-1])


def token_pool_size(cfg: dict | None = None) -> int:
    return int(((cfg or {}).get("joint") or {}).get("token_pool") or 1)


def pool_patch_tokens(tokens, pool: int = 2):
    """Non-overlapping k×k mean pool. 16×26 with k=2 is 8×13."""
    arr = np.asarray(tokens)
    k = int(pool or 1)
    if k <= 1:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"expected (H, W, D) tokens, got {arr.shape}")
    h, w, d = arr.shape
    n_h, n_w = h // k, w // k
    if n_h == 0 or n_w == 0:
        raise ValueError(f"cannot pool {h}x{w} by {k}")
    cropped = arr[: n_h * k, : n_w * k]
    return cropped.reshape(n_h, k, n_w, k, d).mean(axis=(1, 3))


def _as_tensor(rgb):
    import torch
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] in (1, 3):
        image = np.transpose(image, (2, 0, 1))
    if image.max() > 1.5:
        image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    image = (image - mean) / std
    return torch.from_numpy(image).unsqueeze(0)
