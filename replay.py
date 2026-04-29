from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch


@dataclass
class SegmentBatch:
    obs: torch.Tensor          # (B, M, T+1, obs_dim), normalized
    tokens: torch.Tensor       # (B, M, T, token_dim)
    actions: torch.Tensor      # (B, M, T, action_dim)
    rewards: torch.Tensor      # (B, M, T)
    dones: torch.Tensor        # (B, M, T)


class SegmentReplayBuffer:
    """Stores online windows.

    Each stored item is one vector-env member's window:
      M MDP episodes, each T steps long.

    This preserves the sequence structure needed by TTT.
    """

    def __init__(self, capacity_segments: int):
        self.capacity = int(capacity_segments)
        self.storage: List[dict] = []
        self.next_idx = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add_segment(
        self,
        obs: np.ndarray,
        tokens: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        item = {
            "obs": obs.astype(np.float32),
            "tokens": tokens.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "dones": dones.astype(np.float32),
        }

        if len(self.storage) < self.capacity:
            self.storage.append(item)
        else:
            self.storage[self.next_idx] = item

        self.next_idx = (self.next_idx + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> SegmentBatch:
        if len(self.storage) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")

        indices = np.random.randint(0, len(self.storage), size=batch_size)
        obs = np.stack([self.storage[i]["obs"] for i in indices], axis=0)
        tokens = np.stack([self.storage[i]["tokens"] for i in indices], axis=0)
        actions = np.stack([self.storage[i]["actions"] for i in indices], axis=0)
        rewards = np.stack([self.storage[i]["rewards"] for i in indices], axis=0)
        dones = np.stack([self.storage[i]["dones"] for i in indices], axis=0)

        return SegmentBatch(
            obs=torch.tensor(obs, dtype=torch.float32, device=device),
            tokens=torch.tensor(tokens, dtype=torch.float32, device=device),
            actions=torch.tensor(actions, dtype=torch.float32, device=device),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device),
        )
