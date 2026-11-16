"""The three MineDreamer pieces the performer calls, each with a mock and a real side.

Imaginator       current frame + text -> imagined goal frame(s)
                 real: HTTP POST {url}goal_image (current_image PNG, text_prompt, is_del)
                       -> {"result": 1, "goal_image_list": [base64 PNG]}   (port 25547)
PromptGenerator  text + current + goal -> 512-d goal embedding for STEVE-1 (MineDreamer's CVAE)
Policy           obs + goal embedding -> one MineRL action dict (STEVE-1 on VPT)

Mocks never pretend to be the models: the mock Imaginator echoes the current frame and
the mock policy is a pixel heuristic, so a mock run only checks plumbing and file format.
"""

from __future__ import annotations

import base64
import random
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from minedreamer_bridge import actions as A
from minedreamer_bridge import fake_world as W
from minedreamer_bridge.imageio import decode_png, encode_png, resize_nearest

DREAMER_IMAGE_SIZE = (256, 256)  # minedreamer/config.py
EMBED_DIM = 512


@dataclass
class GoalSpec:
    text_prompt: str
    step: int
    goal_image: np.ndarray | None = None
    embed: np.ndarray | None = None
    imagine_latency_s: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- Imaginator
class MockImaginator:
    name = "mock_imaginator(echo current frame)"

    def imagine(self, text_prompt: str, frame: np.ndarray) -> list[np.ndarray]:
        return [frame.copy()]


class HttpImaginator:
    """Client for imaginator/test/minedreamer_backend_{MLLMSD,IP2P}.py."""

    name = "minedreamer_http"

    def __init__(self, url: str, timeout: float = 300.0, session=None):
        self.url = url if url.endswith("/") else url + "/"
        self.timeout = float(timeout)
        self._session = session

    def _post(self, *args, **kwargs):
        if self._session is not None:
            return self._session.post(*args, **kwargs)
        import requests
        return requests.post(*args, **kwargs)

    def imagine(self, text_prompt: str, frame: np.ndarray) -> list[np.ndarray]:
        png = encode_png(resize_nearest(frame, DREAMER_IMAGE_SIZE))
        resp = self._post(
            self.url + "goal_image",
            files={"current_image": ("current.png", png, "image/png")},
            data={"text_prompt": text_prompt, "is_del": 1},
            timeout=self.timeout,
        )
        payload = resp.json()
        if "goal_image_list" not in payload:
            raise RuntimeError(f"Imaginator returned no goal_image_list: {str(payload)[:200]}")
        return [decode_png(base64.b64decode(item)) for item in payload["goal_image_list"]]


# ---------------------------------------------------------- Prompt Generator
class MockPromptGenerator:
    name = "mock_prompt_generator(hash)"

    def embed(self, text_prompt: str, current: np.ndarray, goal: np.ndarray) -> np.ndarray:
        seed = zlib.crc32(text_prompt.encode()) ^ int(goal.mean() * 1000)
        vec = np.random.default_rng(seed).standard_normal(EMBED_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


# -------------------------------------------------------------------- Policy
def _mask(img: np.ndarray, color, tol: int = 12) -> np.ndarray:
    return (np.abs(img.astype(np.int16) - np.array(color, np.int16)) <= tol).all(axis=2)


class MockSteve1:
    """Pixel heuristic standing in for STEVE-1. It sees only obs['pov'] and the text prompt."""

    name = "mock_steve1(pixel heuristic)"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.reset()

    def reset(self):
        self._rng = random.Random(self.seed)
        self._t = 0

    def act(self, obs: dict, goal: GoalSpec) -> dict:
        self._t += 1
        img = obs["pov"]
        text = goal.text_prompt.lower()
        if any(k in text for k in ("tree", "chop", "wood", "log")):
            # "go to"/"find" only approach; "chop" approaches and then attacks
            return self._tree(img, attack_ok=not any(k in text for k in ("go to", "find")))
        if any(k in text for k in ("dig down", "dirt")):
            return self._dig(img)
        return self._explore()

    def _explore(self) -> dict:
        phase = self._t % 54
        if phase < 24:
            return A.make(camera=[0.0, 15.0])
        return A.make(forward=1, jump=1 if self._rng.random() < 0.05 else 0)

    def _tree(self, img: np.ndarray, attack_ok: bool = True) -> dict:
        h, w = img.shape[:2]
        trunk = _mask(img, W.TRUNK)
        cols = np.flatnonzero(trunk.any(axis=0))
        sky_rows = np.flatnonzero(_mask(img, W.SKY).any(axis=1))
        pitch_fix = 0.0
        if sky_rows.size == 0:          # looking at the ground: raise the view
            pitch_fix = -15.0
        if cols.size == 0:
            act = self._explore()
            act["camera"][0] = pitch_fix
            return act
        # nearest trunk = widest contiguous run of trunk columns
        runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        run = max(runs, key=len)
        offset_deg = (run.mean() - w / 2) / (w / 2) * W.HALF_FOV
        yaw = float(np.clip(offset_deg, -10.0, 10.0))
        if attack_ok and len(run) >= 0.2 * w and abs(offset_deg) < 5.0:   # about 3.3 blocks: in reach
            return A.make(attack=1, camera=[pitch_fix, yaw])
        if not attack_ok and len(run) >= 0.3 * w:                         # about 2.2 blocks: stand still
            return A.make(camera=[pitch_fix, yaw])
        return A.make(forward=1, camera=[pitch_fix, yaw])

    def _dig(self, img: np.ndarray) -> dict:
        if _mask(img, W.SKY).any() or _mask(img, W.GRASS).mean() > 0.0:
            return A.make(camera=[15.0, 0.0])
        return A.make(attack=1)


# ----------------------------------------------------------- real MineDreamer
def load_real_stack(cfg: dict):
    """Build (env, policy, prompt_generator) from a MineDreamer checkout. Untested here: needs the
    MineDreamer repo on PYTHONPATH, MineRL 1.0, torch, and a GPU (see minedreamer_bridge/README.md)."""
    try:
        import torch
        from minedreamer.config import CVAE_INFO, DEVICE
        from minedreamer.text_alignment.cvae import load_cvae_model
        from minedreamer.utils.embed_utils import (
            embed_one_frame, get_cvae_embed_from_text_instruction_with_current_and_goal_image_embed)
        from minedreamer.utils.mineclip_agent_env_utils import load_mineclip_agent_env
    except ImportError as exc:  # pragma: no cover - depends on the MineDreamer env
        raise RuntimeError(
            "Real MineDreamer stack is not importable. Run inside the MineDreamer/MineRL environment "
            f"with the repo on PYTHONPATH ({exc})."
        ) from exc

    agent, mineclip, env = load_mineclip_agent_env(
        cfg["in_model"], cfg["in_weights"], cfg["env_seed"], cfg.get("cond_scale", 6.0),
        cfg.get("inventory"), cfg.get("preferred_spawn_biome"))
    CVAE_INFO["model_path"] = cfg["prior_weights_path"]
    cvae = load_cvae_model(CVAE_INFO)

    class RealPromptGenerator:
        name = "minedreamer_cvae"

        def embed(self, text_prompt, current, goal):
            cur = embed_one_frame(current, mineclip, DEVICE)
            gol = embed_one_frame(goal, mineclip, DEVICE)
            with torch.cuda.amp.autocast():
                return get_cvae_embed_from_text_instruction_with_current_and_goal_image_embed(
                    text_prompt, cur, gol, mineclip, cvae, DEVICE)

    class RealSteve1:
        name = "steve1"

        def reset(self):
            if hasattr(agent, "reset"):
                agent.reset(cfg.get("cond_scale", 6.0))

        def act(self, obs, goal):
            with torch.cuda.amp.autocast():
                return agent.get_action(obs, goal.embed)

    return env, RealSteve1(), RealPromptGenerator()


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0
