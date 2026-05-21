"""
rl/ppo.py
=========
Rollout buffer and PPO update algorithm.

Hyperparameter defaults
-----------------------
  clip_eps      = 0.2     (PPO clipping)
  n_epochs      = 4       (passes over each rollout)
  n_mini_batches= 4
  value_coeff   = 0.5
  entropy_coeff = 0.01    (encourages exploration)
  max_grad_norm = 0.5
  gamma         = 0.99
  gae_lambda    = 0.95
"""

import numpy as np
import torch
import torch.nn as nn

from rl.features import MAX_PLANETS, PLANET_FEATURES, FLEET_FEATURES, GLOBAL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
class RolloutBuffer:
    """
    Stores one rollout of experience for a single PPO update.
    All data is stored as raw numpy arrays and converted to tensors
    only when get_tensors() is called.
    """

    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self.reset()

    def reset(self):
        self.planet_tensors  = []
        self.fleet_tensors   = []
        self.global_tensors  = []
        self.valid_masks     = []
        self.my_planet_masks = []

        self.send_actions    = []
        self.angle_actions   = []
        self.frac_actions    = []

        self.log_probs       = []
        self.values          = []
        self.rewards         = []
        self.dones           = []

    def add(self,
            planet_t, fleet_t, global_t,
            valid_mask, my_mask,
            send_a, angle_a, frac_a,
            log_prob, value,
            reward, done):

        self.planet_tensors.append(planet_t)
        self.fleet_tensors.append(fleet_t)
        self.global_tensors.append(global_t)
        self.valid_masks.append(valid_mask)
        self.my_planet_masks.append(my_mask)

        self.send_actions.append(send_a.astype(np.int64))
        self.angle_actions.append(angle_a.astype(np.int64))
        self.frac_actions.append(frac_a.astype(np.int64))

        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def compute_gae(self, last_value: float, gamma=0.99, gae_lambda=0.95):
        """
        Compute Generalised Advantage Estimates (GAE) and returns.

        Returns
        -------
        advantages : float32 np.array [n_steps]
        returns    : float32 np.array [n_steps]
        """
        n           = len(self.rewards)
        advantages  = np.zeros(n, dtype=np.float32)
        last_gae    = 0.0

        for t in reversed(range(n)):
            non_terminal = 1.0 - float(self.dones[t])
            next_val     = self.values[t + 1] if t < n - 1 else last_value

            delta    = self.rewards[t] + gamma * next_val * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def get_tensors(self, device):
        """Pack all stored data into torch tensors on `device`."""
        return {
            "planet"    : torch.FloatTensor(np.array(self.planet_tensors)).to(device),
            "fleet"     : torch.FloatTensor(np.array(self.fleet_tensors)).to(device),
            "global"    : torch.FloatTensor(np.array(self.global_tensors)).to(device),
            "valid_mask": torch.BoolTensor(np.array(self.valid_masks)).to(device),
            "my_mask"   : torch.BoolTensor(np.array(self.my_planet_masks)).to(device),
            "send_a"    : torch.LongTensor(np.array(self.send_actions)).to(device),
            "angle_a"   : torch.LongTensor(np.array(self.angle_actions)).to(device),
            "frac_a"    : torch.LongTensor(np.array(self.frac_actions)).to(device),
            "old_lp"    : torch.FloatTensor(np.array(self.log_probs)).to(device),
            "old_values": torch.FloatTensor(np.array(self.values)).to(device),
        }


# ─────────────────────────────────────────────────────────────────────────────
class PPO:
    """
    Proximal Policy Optimisation trainer.

    Parameters
    ----------
    model          : ActorCritic
    lr             : learning rate
    clip_eps       : PPO clip parameter ε
    n_epochs       : passes over each rollout buffer
    n_mini_batches : number of mini-batches per epoch
    value_coeff    : weight of value loss
    entropy_coeff  : weight of entropy bonus (exploration)
    max_grad_norm  : gradient clipping
    device         : "cpu" or "cuda"
    """

    def __init__(self, model,
                 lr=3e-4,
                 clip_eps=0.2,
                 n_epochs=4,
                 n_mini_batches=4,
                 value_coeff=0.5,
                 entropy_coeff=0.01,
                 max_grad_norm=0.5,
                 device="cpu"):

        self.model          = model
        self.clip_eps       = clip_eps
        self.n_epochs       = n_epochs
        self.n_mini_batches = n_mini_batches
        self.value_coeff    = value_coeff
        self.entropy_coeff  = entropy_coeff
        self.max_grad_norm  = max_grad_norm
        self.device         = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    def update(self, buffer: RolloutBuffer, last_value: float,
               gamma=0.99, gae_lambda=0.95):
        """
        Run PPO update on a completed rollout.

        Returns a dict with mean losses for logging.
        """
        advantages, returns = buffer.compute_gae(last_value, gamma, gae_lambda)

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns    = torch.FloatTensor(returns).to(self.device)

        data = buffer.get_tensors(self.device)
        n    = advantages.shape[0]

        pg_losses, v_losses, ent_losses = [], [], []

        for _ in range(self.n_epochs):
            indices = torch.randperm(n, device=self.device)
            mb_size = max(1, n // self.n_mini_batches)

            for start in range(0, n, mb_size):
                idx = indices[start : start + mb_size]

                log_probs, values, entropy = self.model.evaluate(
                    data["planet"][idx],
                    data["fleet"][idx],
                    data["global"][idx],
                    data["valid_mask"][idx],
                    data["my_mask"][idx],
                    data["send_a"][idx],
                    data["angle_a"][idx],
                    data["frac_a"][idx],
                )

                # ── PPO clipped surrogate objective ───────────────────────
                ratio = torch.exp(log_probs - data["old_lp"][idx])
                adv   = advantages[idx]

                pg_loss = -torch.min(
                    adv * ratio,
                    adv * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps),
                ).mean()

                # ── Value loss ────────────────────────────────────────────
                v_loss = 0.5 * (values - returns[idx]).pow(2).mean()

                # ── Entropy bonus (exploration) ───────────────────────────
                ent_loss = -entropy.mean()

                loss = pg_loss + self.value_coeff * v_loss + self.entropy_coeff * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())

        return {
            "pg_loss" : float(np.mean(pg_losses)),
            "v_loss"  : float(np.mean(v_losses)),
            "ent_loss": float(np.mean(ent_losses)),
        }
