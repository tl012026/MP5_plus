"""Dependency-free PNG encode/decode and resize (numpy + zlib only).

The MineDreamer Imaginator expects a PNG upload and returns base64 PNGs written by PIL.
The decoder handles 8-bit RGB/RGBA, non-interlaced, all five row filters.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def encode_png(img: np.ndarray, filters: list[int] | None = None) -> bytes:
    """img: HxWx3 uint8. filters: optional per-row filter ids (0-4), default all 0."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8, got {img.shape}")
    h, w, _ = img.shape
    bpp = 3
    rows = []
    prev = np.zeros(w * bpp, dtype=np.int32)
    for y in range(h):
        cur = img[y].reshape(-1).astype(np.int32)
        ftype = 0 if filters is None else int(filters[y % len(filters)])
        if ftype == 0:
            out = cur
        elif ftype == 1:
            left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]])
            out = cur - left
        elif ftype == 2:
            out = cur - prev
        elif ftype == 3:
            left = np.concatenate([np.zeros(bpp, np.int32), cur[:-bpp]])
            out = cur - (left + prev) // 2
        elif ftype == 4:
            out = np.empty_like(cur)
            for i in range(len(cur)):
                a = cur[i - bpp] if i >= bpp else 0
                c = prev[i - bpp] if i >= bpp else 0
                out[i] = cur[i] - _paeth(int(a), int(prev[i]), int(c))
        else:
            raise ValueError(f"bad filter {ftype}")
        rows.append(bytes([ftype]) + (out % 256).astype(np.uint8).tobytes())
        prev = cur
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return _SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + _chunk(b"IEND", b"")


def decode_png(data: bytes) -> np.ndarray:
    """Returns HxWx3 uint8 (alpha dropped)."""
    if data[:8] != _SIG:
        raise ValueError("not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if ihdr is None:
        raise ValueError("PNG without IHDR")
    w, h, depth, ctype, _comp, _filt, interlace = ihdr
    if depth != 8 or ctype not in (2, 6) or interlace != 0:
        raise ValueError(f"unsupported PNG (depth={depth}, color type={ctype}, interlace={interlace})")
    bpp = 3 if ctype == 2 else 4
    raw = zlib.decompress(b"".join(idat))
    stride = w * bpp
    out = np.zeros((h, stride), dtype=np.uint8)
    prev = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        base = y * (stride + 1)
        ftype = raw[base]
        line = np.frombuffer(raw, dtype=np.uint8, count=stride, offset=base + 1).astype(np.int32)
        if ftype == 0:
            cur = line
        elif ftype == 2:
            cur = (line + prev) % 256
        else:
            cur = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                a = int(cur[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                c = int(prev[i - bpp]) if i >= bpp else 0
                if ftype == 1:
                    pred = a
                elif ftype == 3:
                    pred = (a + b) // 2
                elif ftype == 4:
                    pred = _paeth(a, b, c)
                else:
                    raise ValueError(f"bad filter {ftype}")
                cur[i] = (int(line[i]) + pred) % 256
        out[y] = cur.astype(np.uint8)
        prev = cur
    img = out.reshape(h, w, bpp)
    return np.ascontiguousarray(img[:, :, :3])


def resize_nearest(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = img.shape[:2]
    th, tw = int(size_hw[0]), int(size_hw[1])
    ys = (np.arange(th) * h // th).clip(0, h - 1)
    xs = (np.arange(tw) * w // tw).clip(0, w - 1)
    return img[ys][:, xs]
