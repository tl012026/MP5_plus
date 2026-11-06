"""Resize shapes for a frozen ViT. The encoder itself is loaded only on a GPU node."""

from __future__ import annotations


def target_hw(height: int, width: int, out_h: int = 224, patch: int = 14) -> tuple[int, int]:
    """Keep the aspect ratio. Height is out_h and width is a positive multiple of patch."""
    out_w = int(round((width * out_h / max(height, 1)) / patch)) * patch
    return out_h, max(patch, out_w)
