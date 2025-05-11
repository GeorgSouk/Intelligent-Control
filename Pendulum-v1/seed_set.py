"""
seed_setter.py

Utility module providing a `SeedSetter` class that synchronises random seeds across
Python's `random`, NumPy, PyTorch and (optionally) Gymnasium environments to ensure
reproducible experiments.

Example
-------
>>> from seed_setter import SeedSetter
>>> seed = SeedSetter(42)          # sets global seeds immediately
>>> env = gym.make("CartPole-v1")
>>> seed.apply_to_env(env)         # seeds env, its action & observation spaces
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch

__all__: list[str] = ["SeedSetter"]


class SeedSetter:
    """Propagate a single integer seed across common pseudo‑random number generators.

    Parameters
    ----------
    seed : int
        The seed to use for all PRNGs.

    Notes
    -----
    * ``torch.cuda.manual_seed_all`` is called unconditionally—PyTorch will silently
      ignore it if CUDA is not available, so it's safe on CPU‑only machines.
    * The :py:meth:`apply_to_env` method targets libraries that follow the *Gymnasium*
      API. It first tries ``env.reset(seed=seed)`` (Gymnasium >= 0.26), and then seeds
      the action and observation spaces if they expose a ``seed`` method.
    """

    def __init__(self, seed: int) -> None:
        self.seed: int = seed
        # Eagerly set global seeds upon instantiation
        self.set_global_seed()

    # ---------------------------------------------------------------------
    # Global seeding helpers
    # ---------------------------------------------------------------------
    def set_global_seed(self) -> None:
        """Seed Python's ``random``, NumPy, and PyTorch PRNGs.

        You can call this method again later if you ever need to re‑seed the
        global state.
        """
        random.seed(self.seed)                # built‑in RNG
        np.random.seed(self.seed)             # NumPy RNG
        torch.manual_seed(self.seed)          # PyTorch CPU RNG
        torch.cuda.manual_seed_all(self.seed)  # all CUDA devices (no‑op if unavailable)

    # ---------------------------------------------------------------------
    # Environment seeding (Gymnasium‑style APIs)
    # ---------------------------------------------------------------------
    def apply_to_env(self, env: Any) -> None:
        """Attempt to seed a Gymnasium‑style environment and its spaces.

        Parameters
        ----------
        env : Any
            The environment whose internal RNGs should be seeded. It is
            expected to provide at least a ``reset`` method that accepts a
            ``seed`` keyword argument, as well as ``action_space`` and/or
            ``observation_space`` attributes exposing a ``seed`` method.

        Notes
        -----
        This method performs duck typing—no hard dependency on Gymnasium is
        introduced. Attributes are checked with :py:func:`hasattr` before use.
        """
        # Seed the environment's internal RNG (Gymnasium >= 0.26)
        if hasattr(env, "reset"):
            try:
                env.reset(seed=self.seed)
            except TypeError:
                # Older Gym versions may not accept the keyword argument
                pass

        # Seed the action space RNG, if present
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(self.seed)

        # Seed the observation space RNG, if present
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(self.seed)
