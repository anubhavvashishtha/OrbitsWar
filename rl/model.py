"""
rl/model.py
===========
ActorCritic neural network for the PPO Orbit Wars agent.

Architecture
------------
  PlanetEncoder  [48 × 13]  → masked mean-pool → [64]
  FleetEncoder   [100 × 12] → masked mean-pool → [64]
  GlobalMLP      [10]
        concat [138]  →  SharedTrunk  →  [256]
                          │
                          ├── CriticHead  → scalar value V(s)
                          │
                          └── ActorHead (per-planet × 48)
                               ├── send_logit   [1]   → Bernoulli
                               ├── angle_logits [32]  → Categorical
                               └── frac_logits  [6]   → Categorical
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Categorical

from rl.features import (
    MAX_PLANETS, MAX_FLEETS,
    PLANET_FEATURES, FLEET_FEATURES, GLOBAL_FEATURES,
    ANGLE_BINS, N_FRACTIONS,
)

PLANET_EMBED = 64
FLEET_EMBED  = 64
SHARED_DIM   = 256


def _mlp(sizes, activation=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [activation()]
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Combined actor-critic network for PPO."""

    def __init__(self):
        super().__init__()

        # ── Encoders ────────────────────────────────────────────────────────
        self.planet_enc = nn.Sequential(
            nn.Linear(PLANET_FEATURES, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, PLANET_EMBED),
            nn.ReLU(),
        )

        self.fleet_enc = nn.Sequential(
            nn.Linear(FLEET_FEATURES, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, FLEET_EMBED),
            nn.ReLU(),
        )

        # ── Shared trunk ────────────────────────────────────────────────────
        trunk_in = PLANET_EMBED + FLEET_EMBED + GLOBAL_FEATURES
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, SHARED_DIM),
            nn.ReLU(),
            nn.Linear(SHARED_DIM, SHARED_DIM),
            nn.ReLU(),
        )

        # ── Per-planet actor heads ──────────────────────────────────────────
        per_planet_in = PLANET_EMBED + SHARED_DIM
        self.head_send  = nn.Linear(per_planet_in, 1)          # Bernoulli
        self.head_angle = nn.Linear(per_planet_in, ANGLE_BINS)  # Categorical
        self.head_frac  = nn.Linear(per_planet_in, N_FRACTIONS) # Categorical

        # ── Critic ──────────────────────────────────────────────────────────
        self.critic = nn.Linear(SHARED_DIM, 1)

        self._init_weights()

    # ── Weight initialisation ───────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for head in (self.head_send, self.head_angle, self.head_frac):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    # ── Internal encoder ────────────────────────────────────────────────────
    def _encode(self, planet_t, fleet_t, global_t, valid_mask):
        """
        Encode board state into per-planet embeddings + shared latent.

        planet_t   : [B, MAX_PLANETS, PLANET_FEATURES]
        fleet_t    : [B, MAX_FLEETS,  FLEET_FEATURES]
        global_t   : [B, GLOBAL_FEATURES]
        valid_mask : [B, MAX_PLANETS] bool

        Returns
        -------
        p_emb   : [B, MAX_PLANETS, PLANET_EMBED]
        shared  : [B, SHARED_DIM]
        """
        # Per-planet embeddings
        p_emb = self.planet_enc(planet_t)                        # [B,48,64]

        # Masked mean-pool (ignore padding rows)
        vm     = valid_mask.unsqueeze(-1).float()                # [B,48,1]
        p_pool = (p_emb * vm).sum(1) / (vm.sum(1) + 1e-6)       # [B,64]

        # Fleet embeddings — use the is_valid channel (last feature) as mask
        f_emb   = self.fleet_enc(fleet_t)                        # [B,100,64]
        f_valid = fleet_t[..., -1:]                              # [B,100,1]
        f_pool  = (f_emb * f_valid).sum(1) / (f_valid.sum(1) + 1e-6)  # [B,64]

        # Shared trunk
        shared = self.trunk(
            torch.cat([p_pool, f_pool, global_t], dim=-1)
        )                                                        # [B,256]

        return p_emb, shared

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, planet_t, fleet_t, global_t, valid_mask, _my_mask=None):
        """
        Returns (send_logits, angle_logits, frac_logits, value).

        send_logits  : [B, MAX_PLANETS]
        angle_logits : [B, MAX_PLANETS, ANGLE_BINS]
        frac_logits  : [B, MAX_PLANETS, N_FRACTIONS]
        value        : [B, 1]
        """
        p_emb, shared = self._encode(planet_t, fleet_t, global_t, valid_mask)

        # Expand shared for per-planet concat
        shared_exp  = shared.unsqueeze(1).expand(-1, MAX_PLANETS, -1)  # [B,48,256]
        per_planet  = torch.cat([p_emb, shared_exp], dim=-1)           # [B,48,320]

        send_logits  = self.head_send(per_planet).squeeze(-1)           # [B,48]
        angle_logits = self.head_angle(per_planet)                      # [B,48,32]
        frac_logits  = self.head_frac(per_planet)                       # [B,48,6]
        value        = self.critic(shared)                              # [B,1]

        return send_logits, angle_logits, frac_logits, value

    # ── Value only (for bootstrap) ──────────────────────────────────────────
    def get_value(self, planet_t, fleet_t, global_t, valid_mask):
        _, shared = self._encode(planet_t, fleet_t, global_t, valid_mask)
        return self.critic(shared)

    # ── Act (inference / rollout) ───────────────────────────────────────────
    @torch.no_grad()
    def act(self, planet_t, fleet_t, global_t, valid_mask, my_mask, deterministic=False):
        """
        Sample (or greedily select) actions for a single time step.

        All inputs: batch=1 tensors, i.e. [1, ...].

        Returns
        -------
        send_actions  : bool  np.array [MAX_PLANETS]
        angle_actions : int   np.array [MAX_PLANETS]
        frac_actions  : int   np.array [MAX_PLANETS]
        log_prob      : scalar tensor
        value         : scalar tensor
        """
        send_logits, angle_logits, frac_logits, value = self.forward(
            planet_t, fleet_t, global_t, valid_mask
        )

        # Squeeze batch dim
        send_logits  = send_logits.squeeze(0)   # [48]
        angle_logits = angle_logits.squeeze(0)  # [48, 32]
        frac_logits  = frac_logits.squeeze(0)   # [48, 6]
        my_mask_1d   = my_mask.squeeze(0)       # [48]

        # Sample / argmax
        send_dist = Bernoulli(logits=send_logits)
        if deterministic:
            send_a = (send_logits > 0).bool()
            angle_a = angle_logits.argmax(dim=-1)
            frac_a  = frac_logits.argmax(dim=-1)
        else:
            send_a  = send_dist.sample().bool()
            angle_a = Categorical(logits=angle_logits).sample()
            frac_a  = Categorical(logits=frac_logits).sample()

        # Only owned planets may send
        send_a = send_a & my_mask_1d

        # Log probability (sum over all owned planet decisions)
        log_prob = torch.tensor(0.0)
        if my_mask_1d.any():
            send_lp  = send_dist.log_prob(send_a.float())[my_mask_1d].sum()

            sending  = send_a & my_mask_1d
            if sending.any():
                angle_lp = Categorical(logits=angle_logits[sending]).log_prob(angle_a[sending]).sum()
                frac_lp  = Categorical(logits=frac_logits[sending]).log_prob(frac_a[sending]).sum()
            else:
                angle_lp = frac_lp = torch.tensor(0.0)

            log_prob = send_lp + angle_lp + frac_lp

        return (
            send_a.cpu().numpy(),
            angle_a.cpu().numpy(),
            frac_a.cpu().numpy(),
            log_prob,
            value.squeeze(),
        )

    # ── Evaluate (PPO update) ───────────────────────────────────────────────
    def evaluate(self, planet_t, fleet_t, global_t, valid_mask, my_mask,
                 send_a, angle_a, frac_a):
        """
        Evaluate stored actions for the PPO loss calculation.
        Fully vectorised — no Python loops.

        All inputs are batched [B, ...].

        Returns
        -------
        log_probs : [B]
        values    : [B]
        entropies : [B]
        """
        send_logits, angle_logits, frac_logits, values = self.forward(
            planet_t, fleet_t, global_t, valid_mask
        )
        values = values.squeeze(-1)  # [B]

        my_mask_f    = my_mask.float()                     # [B, 48]
        sending_mask = send_a.bool() & my_mask             # [B, 48]
        sending_f    = sending_mask.float()                 # [B, 48]

        # ── Send log probs ───────────────────────────────────────────────────
        send_dist = Bernoulli(logits=send_logits)
        send_lp   = (send_dist.log_prob(send_a.float()) * my_mask_f).sum(dim=1)
        send_ent  = (send_dist.entropy() * my_mask_f).sum(dim=1)

        # ── Angle log probs ──────────────────────────────────────────────────
        angle_dist = Categorical(logits=angle_logits)          # over [B,48,32]
        angle_lp   = (angle_dist.log_prob(angle_a) * sending_f).sum(dim=1)
        angle_ent  = (angle_dist.entropy() * sending_f).sum(dim=1)

        # ── Fraction log probs ───────────────────────────────────────────────
        frac_dist = Categorical(logits=frac_logits)            # over [B,48,6]
        frac_lp   = (frac_dist.log_prob(frac_a) * sending_f).sum(dim=1)
        frac_ent  = (frac_dist.entropy() * sending_f).sum(dim=1)

        log_probs = send_lp + angle_lp + frac_lp              # [B]
        entropies = send_ent + angle_ent + frac_ent            # [B]

        return log_probs, values, entropies
