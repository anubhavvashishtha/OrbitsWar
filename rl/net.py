"""
rl/net.py  —  Neural network for Orbit Wars value prediction.

Architecture: 30 → 128 (LayerNorm+ReLU) → 64 (ReLU) → 1 (Sigmoid)
Same 30 state features as the GBM classifier (classifier/train_from_replays.py).
"""
import torch
import torch.nn as nn

STATE_DIM  = 30   # must match _extract_features() in main.py
HIDDEN1    = 128
HIDDEN2    = 64


class OrbitsNet(nn.Module):
    """
    Value network: predicts P(win) from 30-dim state vector.
    Exported as numpy arrays for production use (export_weights.py).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN1),
            nn.LayerNorm(HIDDEN1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def predict(self, feats: list) -> float:
        """Single-sample inference from a Python list."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(feats).unsqueeze(0)
            return float(self.forward(x))
