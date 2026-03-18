"""Tiny exact recurrent model for checkpoint-resume validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ToyRecurrentModel:
    dim: int = 4
    seed: int = 42

    def token_features(self, token: int) -> tuple[float, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 97 * token)
        lam = float(rng.uniform(0.1, 0.9))
        v = rng.normal(size=self.dim)
        k = rng.normal(size=self.dim)
        return lam, v, k

    def step(self, state: np.ndarray, token: int) -> np.ndarray:
        lam, v, k = self.token_features(token)
        return lam * state + np.outer(v, k)

    def run(self, tokens: tuple[int, ...], start_state: np.ndarray | None = None) -> np.ndarray:
        state = np.zeros((self.dim, self.dim), dtype=float) if start_state is None else start_state.copy()
        for token in tokens:
            state = self.step(state, token)
        return state


def checkpoint_resume_error(tokens: tuple[int, ...], checkpoint_depth: int, dim: int = 4, seed: int = 42) -> float:
    model = ToyRecurrentModel(dim=dim, seed=seed)
    full = model.run(tokens)
    checkpoint_state = model.run(tokens[:checkpoint_depth])
    resumed = model.run(tokens[checkpoint_depth:], start_state=checkpoint_state)
    return float(np.max(np.abs(full - resumed)))

