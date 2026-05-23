"""
rl/buffer.py  —  Rollout buffer with Generalised Advantage Estimation (GAE).
"""
import numpy as np


class RolloutBuffer:
    """
    Stores (state, reward, done, value) tuples from one or more games.
    Converts to numpy arrays ready for a TD/PPO update.
    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam   = lam
        self.clear()

    def clear(self):
        self.states  = []
        self.rewards = []
        self.dones   = []
        self.values  = []

    def add(self, state: list, reward: float, done: bool, value: float):
        self.states.append(state)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def compute_gae(self, last_value: float = 0.0):
        """Returns (returns, advantages) both as plain lists."""
        advantages, returns = [], []
        gae      = 0.0
        next_val = last_value
        for r, d, v in zip(reversed(self.rewards),
                           reversed(self.dones),
                           reversed(self.values)):
            delta = r + self.gamma * next_val * (1.0 - d) - v
            gae   = delta + self.gamma * self.lam * (1.0 - d) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_val = v
        return returns, advantages

    def as_arrays(self, last_value: float = 0.0):
        returns, advantages = self.compute_gae(last_value)
        return (
            np.array(self.states,     dtype=np.float32),
            np.array(returns,          dtype=np.float32),
            np.array(advantages,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.rewards)
