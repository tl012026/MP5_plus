"""Count env.step calls and abort on timeout or death. Does not change the action."""

from __future__ import annotations

import numpy as np


class EpisodeTimeout(Exception):
    def __init__(self, steps):
        super().__init__(f"episode exceeded max_steps at step {steps}")
        self.steps = steps


class EpisodeDeath(Exception):
    def __init__(self, steps, life):
        super().__init__(f"agent died at step {steps} (life={life})")
        self.steps = steps
        self.life = life


def _as_float(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return float(np.asarray(value).reshape(-1)[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_life(obs):
    """
    MineDojo privileged observation `life_stats` (see MineDojo Observation Space docs /
    minedojo/sim/wrappers/ar_nn/nn_observation.py): `life` is health in [0, 20].
    Death is life <= 0. Some builds also expose `is_alive` (bool).
    """
    if obs is None or not isinstance(obs, dict):
        return None
    stats = obs.get("life_stats")
    if stats is None:
        return None
    if isinstance(stats, dict):
        if "is_alive" in stats:
            alive = stats["is_alive"]
            if isinstance(alive, np.ndarray):
                alive = bool(np.asarray(alive).reshape(-1)[0])
            if alive is False:
                return 0.0
        return _as_float(stats.get("life"))
    return None


def is_dead(obs):
    life = read_life(obs)
    return life is not None and life <= 0.0


class EnvWrapper:
    def __init__(self, env, max_steps, logger=None):
        self._env = env
        self.max_steps = int(max_steps)
        self.steps = 0
        self.logger = logger
        self.last_obs = None

    def step(self, action):
        result = self._env.step(action)
        self.steps += 1
        obs = result[0] if isinstance(result, (tuple, list)) else result
        self.last_obs = obs
        if self.steps > self.max_steps:
            raise EpisodeTimeout(self.steps)
        if is_dead(obs):
            raise EpisodeDeath(self.steps, read_life(obs))
        return result

    def reset(self, *args, **kwargs):
        obs = self._env.reset(*args, **kwargs)
        self.last_obs = obs
        return obs

    def __getattr__(self, name):
        return getattr(self._env, name)
