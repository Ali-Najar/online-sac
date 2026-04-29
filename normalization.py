from __future__ import annotations

import numpy as np


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean,
            "var": self.var,
            "count": self.count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float32)
        self.var = np.asarray(state["var"], dtype=np.float32)
        self.count = float(state["count"])
