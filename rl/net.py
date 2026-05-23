"""
rl/net.py  —  Neural network for Orbit Wars (actor-critic).

Architecture: shared trunk  30 → 128 (LayerNorm+ReLU+Dropout) → 64 (ReLU)
              value  head   64 → 1  (Sigmoid)   P(win)
              policy head   64 → 6  (raw logits) strategic intent
"""
import torch
import torch.nn as nn

STATE_DIM  = 30   # must match extract_features() in classifier/train_from_replays.py
HIDDEN1    = 128
HIDDEN2    = 64
N_ACTIONS  = 6    # strategic intents (see INTENT_NAMES in self_play_trainer.py)


class OrbitsNet(nn.Module):
    """
    Actor-critic network.

    forward(x)  → (value, policy_logits)
    value(x)    → value only   (used in replay pretraining, export)
    predict(fs) → float        (legacy interface for main.py rl_value_fn export)
    """
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN1),
            nn.LayerNorm(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
        )
        self.value_head  = nn.Sequential(nn.Linear(HIDDEN2, 1), nn.Sigmoid())
        self.policy_head = nn.Linear(HIDDEN2, N_ACTIONS)   # raw logits → Categorical outside

    def forward(self, x):
        """Returns (value, policy_logits) — both as tensors."""
        z = self.trunk(x)
        return self.value_head(z).squeeze(-1), self.policy_head(z)

    def value(self, x):
        """Value only — used during replay pretraining and export."""
        return self.value_head(self.trunk(x)).squeeze(-1)

    def predict(self, feats: list) -> float:
        """Single-sample value inference from a Python list (keeps old interface)."""
        self.eval()
        with torch.no_grad():
            return float(self.value(torch.FloatTensor(feats).unsqueeze(0)))
