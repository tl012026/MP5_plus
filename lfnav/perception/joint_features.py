"""Shared frozen-encoder + pool path for joint training extract and eval.

Training extract does **not** resize again: it decodes the on-disk JPEG written at
collect time by ``preprocess_frame`` (224×364, bilinear, JPEG quality 90).
Live eval runs ``preprocess_frame`` on MineDojo RGB, then this function.

Color order (HWC uint8 → CHW) and ImageNet mean/std live in the encoder
(``_as_tensor``). Spatial pooling is ``pool_patch_tokens``.
"""

from __future__ import annotations

import numpy as np

from lfnav.perception.encoder import pool_patch_tokens


def encode_joint_tokens(rgb, encoder, pool: int) -> dict:
    encoded = encoder(rgb)
    tokens = pool_patch_tokens(np.asarray(encoded["tokens"], dtype=np.float32), int(pool))
    pooled = tokens.mean(axis=(0, 1))
    out = dict(encoded)
    out["tokens"] = tokens
    out["pooled"] = np.asarray(pooled, dtype=np.float32)
    return out
