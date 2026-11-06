"""Frozen image encoders. DINOv2 and MineCLIP load only when requested.

The fake encoder is the stand-in used by --dry-run and the login-node tests.
"""

from __future__ import annotations

import numpy as np


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
        return {"cls": cls, "tokens": tokens}


def load_encoder(name: str, cfg: dict | None = None):
    cfg = cfg or {}
    if name in ("fake", "dry-run"):
        return FakeEncoder()
    if name.startswith("dinov2"):
        return _dinov2(name)
    if name == "mineclip":
        return _mineclip(cfg)
    raise KeyError(name)


def _dinov2(name: str):
    import torch
    hub = {"dinov2_vits14": "dinov2_vits14", "dinov2_vitb14": "dinov2_vitb14"}[name]

    model = torch.hub.load("facebookresearch/dinov2", hub)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    class _Wrap:
        name = name

        def __call__(self, rgb):
            tensor = _as_tensor(rgb)
            with torch.no_grad():
                feats = model.forward_features(tensor)
            cls = feats["x_norm_clstoken"][0].cpu().numpy()
            tokens = feats["x_norm_patchtokens"][0].cpu().numpy()
            side = int(tokens.shape[0] ** 0.5)
            return {"cls": cls, "tokens": tokens.reshape(side, -1, tokens.shape[-1])}

    return _Wrap()


def _mineclip(cfg: dict):
    path = (cfg.get("perception") or {}).get("mineclip_dir")
    raise RuntimeError(
        f"MineCLIP weights were not wired. Expected a local checkout under {path}. "
        "The login-node tests use the fake encoder. Skipping MineCLIP is recorded in REPORT.md."
    )


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
