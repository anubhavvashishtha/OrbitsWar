"""
model.py
--------
EdgeTransformerNet — joint edge scorer for Orbit Wars imitation learning.

All candidate edges for a game turn are embedded and processed TOGETHER
through a Transformer encoder.  Every edge (A→X) attends to every other
edge (B→Y) in the same turn, enabling coordination:
    "I'm already sending 60 % from planet A to X,
     so the (A→Y) score should drop."

Architecture
------------
Per-edge embedding  : split 81-dim into [global|src|tgt|edge] parts,
                      project each separately (64-d), fuse → d_model.
Transformer encoder : num_layers × multi-head self-attention + FFN.
                      Pre-LN (norm_first=True) for stable training.
Per-edge heads      : action logit (take this edge?) + fraction logit
                      (what fraction of ships to send?).

Training
--------
  python model.py --dataset imitation/dataset.npz --out_dir checkpoints/

Dataset format (produced by extract_dataset.py):
  X          (N, 81) float32
  y_action   (N,)    int8
  y_fraction (N,)    float32 — NaN for negative samples
  replay     (N,)    int32   — replay index (for grouped CV split)
  step       (N,)    int16   — game step (combined with replay → turn_id)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ════════════════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════════════════

class EdgeTransformerNet(nn.Module):
    """
    Joint edge scorer via self-attention.

    Forward:
        x        : (B, E, 81)  — B turns, E edges per turn (padded)
        pad_mask : (B, E) bool — True = padding token (ignored everywhere)
    Returns:
        action_logit   (B, E, 1)
        fraction_logit (B, E, 1)

    At inference pass a single turn (B=1) with pad_mask=None.
    """

    def __init__(self,
                 d_model:    int   = 128,
                 nhead:      int   = 4,
                 num_layers: int   = 3,
                 dropout:    float = 0.10):
        super().__init__()

        # Per-component projectors — split the 81-dim vector into parts
        # [global(31) | src_planet(18) | tgt_planet(18) | edge(14)]
        self.global_proj = nn.Sequential(
            nn.Linear(31, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.src_proj = nn.Sequential(
            nn.Linear(18, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.tgt_proj = nn.Sequential(
            nn.Linear(18, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(14, 64), nn.LayerNorm(64), nn.GELU()
        )

        # Fuse 4 x 64 = 256 → d_model
        self.fuse = nn.Sequential(
            nn.Linear(256, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Transformer encoder — all edges in a turn attend to each other.
        # norm_first=True (Pre-LN) gives more stable training.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Per-edge output heads
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.fraction_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x        : (B, E, 81)
        pad_mask : (B, E) bool — True = padding (skipped in attention + loss)
        Returns  : action_logit (B, E, 1), fraction_logit (B, E, 1)
        """
        g = self.global_proj(x[..., :31])     # (B, E, 64)
        s = self.src_proj(x[..., 31:49])      # (B, E, 64)
        t = self.tgt_proj(x[..., 49:67])      # (B, E, 64)
        e = self.edge_proj(x[..., 67:81])     # (B, E, 64)

        h = self.fuse(torch.cat([g, s, t, e], dim=-1))          # (B, E, d_model)
        h = self.transformer(h, src_key_padding_mask=pad_mask)   # (B, E, d_model)

        return self.action_head(h), self.fraction_head(h)        # (B, E, 1) each

    def predict_proba(self, x: torch.Tensor,
                      pad_mask: Optional[torch.Tensor] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_prob, fraction) in [0, 1], shape (B, E, 1)."""
        with torch.no_grad():
            al, fl = self.forward(x, pad_mask)
            return torch.sigmoid(al), torch.sigmoid(fl)


# ════════════════════════════════════════════════════════════════════════════
# DATASET
# ════════════════════════════════════════════════════════════════════════════

class TurnDataset(Dataset):
    """
    Groups flat edge samples into per-turn sets.

    Each item = all edges from ONE game turn (one replay + step pair).
    The Transformer needs the full edge set so attention runs across
    every move the winner could have taken in that turn.

    Memory layout: flat X / label arrays are kept as-is (shared with
    NumPy, no copy).  Only per-turn index lists are extra overhead.

    turn_id = replay_id * 10_000 + step   (unique per turn).
    """

    def __init__(self,
                 X:          np.ndarray,   # (N, 81) float32
                 y_action:   np.ndarray,   # (N,)    int8
                 y_fraction: np.ndarray,   # (N,)    float32, NaN for negatives
                 replay_ids: np.ndarray,   # (N,)    int32
                 step_ids:   np.ndarray):  # (N,)    int16

        # Flat tensors (shared memory)
        self._X          = torch.from_numpy(X).float()
        y_frac_clean     = np.where(np.isnan(y_fraction), 0.0, y_fraction)
        self._y_action   = torch.from_numpy(y_action.astype(np.float32))
        self._y_fraction = torch.from_numpy(y_frac_clean.astype(np.float32))
        self._has_frac   = torch.from_numpy((~np.isnan(y_fraction)).astype(np.float32))

        # Group sample indices by turn
        turn_ids = replay_ids.astype(np.int64) * 10_000 + step_ids.astype(np.int64)
        groups: dict = defaultdict(list)
        for i, tid in enumerate(turn_ids):
            groups[tid].append(i)
        self._turns = [np.array(v, dtype=np.int64) for v in groups.values()]

        e_per_turn = np.array([len(v) for v in self._turns])
        print(f"  TurnDataset: {len(self._turns):,} turns  "
              f"edges/turn  min={e_per_turn.min()} "
              f"mean={e_per_turn.mean():.1f} "
              f"max={e_per_turn.max()}")

    def __len__(self) -> int:
        return len(self._turns)

    def __getitem__(self, idx: int):
        idxs = self._turns[idx]
        return (
            self._X[idxs],            # (E, 81)
            self._y_action[idxs],     # (E,)
            self._y_fraction[idxs],   # (E,)
            self._has_frac[idxs],     # (E,)
        )


def collate_turns(batch):
    """
    Pad a list of variable-length turns to the same max edge count E_max.

    Returns:
        X_out    : (B, E_max, 81)   float32
        y_action : (B, E_max)       float32
        y_frac   : (B, E_max)       float32
        has_frac : (B, E_max)       float32
        pad_mask : (B, E_max)       bool — True = padding
    """
    B     = len(batch)
    E_max = max(t[0].shape[0] for t in batch)
    F     = batch[0][0].shape[1]   # 81

    X_out    = torch.zeros(B, E_max, F)
    y_act    = torch.zeros(B, E_max)
    y_frac   = torch.zeros(B, E_max)
    has_frac = torch.zeros(B, E_max)
    pad_mask = torch.ones(B, E_max, dtype=torch.bool)   # True = padding

    for i, (xi, ya, yf, hf) in enumerate(batch):
        E = xi.shape[0]
        X_out[i, :E]    = xi
        y_act[i, :E]    = ya
        y_frac[i, :E]   = yf
        has_frac[i, :E] = hf
        pad_mask[i, :E] = False   # real edge

    return X_out, y_act, y_frac, has_frac, pad_mask


# ════════════════════════════════════════════════════════════════════════════
# LOSS
# ════════════════════════════════════════════════════════════════════════════

def compute_loss(action_logit:   torch.Tensor,
                 fraction_logit: torch.Tensor,
                 y_action:       torch.Tensor,
                 y_fraction:     torch.Tensor,
                 has_frac:       torch.Tensor,
                 pad_mask:       torch.Tensor,
                 pos_weight_factor:    float = 2.0,
                 fraction_loss_weight: float = 0.5):
    """
    Two-head loss on (B, E) turn batches.  Padding positions are excluded.

    action_logit   : (B, E, 1)
    fraction_logit : (B, E, 1)
    y_action       : (B, E)  0/1
    y_fraction     : (B, E)  target fraction
    has_frac       : (B, E)  1.0 if positive sample
    pad_mask       : (B, E)  True = padding — excluded from loss
    """
    real = ~pad_mask
    al   = action_logit.squeeze(-1)[real]     # (N_real,)
    fl   = fraction_logit.squeeze(-1)[real]

    ya = y_action[real]
    yf = y_fraction[real]
    hf = has_frac[real]

    pos_w       = torch.tensor([pos_weight_factor], device=al.device)
    action_loss = F.binary_cross_entropy_with_logits(al, ya, pos_weight=pos_w)

    frac_pred = torch.sigmoid(fl)
    frac_mask = hf > 0.5
    if frac_mask.sum() > 0:
        frac_loss = F.mse_loss(frac_pred[frac_mask], yf[frac_mask])
    else:
        frac_loss = torch.tensor(0.0, device=al.device)

    total = action_loss + fraction_loss_weight * frac_loss
    return total, action_loss, frac_loss


# ════════════════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train(
    dataset_path:         str,
    out_dir:              str,
    d_model:              int   = 128,
    nhead:                int   = 4,
    num_layers:           int   = 3,
    dropout:              float = 0.10,
    batch_size:           int   = 32,       # TURNS per batch (not edges)
    lr:                   float = 3e-4,
    weight_decay:         float = 1e-4,
    epochs:               int   = 40,
    val_fraction:         float = 0.10,
    pos_weight_factor:    float = 2.0,
    fraction_loss_weight: float = 0.5,
    seed:                 int   = 42,
    device:               str   = 'auto',
):
    """
    Train EdgeTransformerNet on turn-grouped batches.

    batch_size = number of turns per batch.
    Effective edge count ~ batch_size x avg_edges_per_turn (typically 32 x 20 = 640).

    Requires 'step' and 'replay' arrays in the .npz (from extract_dataset.py).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── load data ────────────────────────────────────────────────────────────
    print(f"Loading {dataset_path} ...")
    d          = np.load(dataset_path)
    X          = d['X']
    y_action   = d['y_action']
    y_fraction = d['y_fraction']
    replay_ids = d['replay'] if 'replay' in d else np.zeros(len(X), dtype=np.int32)
    step_ids   = d['step']   if 'step'   in d else np.zeros(len(X), dtype=np.int16)

    print(f"  Samples: {len(X):,}  "
          f"Positives: {y_action.sum():,} ({y_action.mean()*100:.1f}%)")

    # ── train/val split by replay ─────────────────────────────────────────────
    unique_replays = np.unique(replay_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_replays)
    n_val     = max(1, int(len(unique_replays) * val_fraction))
    val_set   = set(map(int, unique_replays[:n_val]))
    train_set = set(map(int, unique_replays[n_val:]))

    tr_mask  = np.array([int(r) in train_set for r in replay_ids])
    val_mask = ~tr_mask

    print("Building TurnDataset (train) ...")
    tr_ds  = TurnDataset(X[tr_mask],  y_action[tr_mask],  y_fraction[tr_mask],
                         replay_ids[tr_mask],  step_ids[tr_mask])
    print("Building TurnDataset (val) ...")
    val_ds = TurnDataset(X[val_mask], y_action[val_mask], y_fraction[val_mask],
                         replay_ids[val_mask], step_ids[val_mask])

    print(f"  Turns train={len(tr_ds):,}  val={len(val_ds):,}  ({n_val} val replays)")

    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size,  shuffle=True,
                            collate_fn=collate_turns, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,  shuffle=False,
                            collate_fn=collate_turns, num_workers=0)

    # ── model ─────────────────────────────────────────────────────────────────
    model = EdgeTransformerNet(
        d_model=d_model, nhead=nhead,
        num_layers=num_layers, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    history = []

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, epochs + 1):

        # train
        model.train()
        tr_total = 0.0; n_batches = 0
        for X_b, y_act_b, y_frac_b, hf_b, pmask_b in tr_loader:
            X_b, y_act_b, y_frac_b, hf_b, pmask_b = (
                X_b.to(device), y_act_b.to(device),
                y_frac_b.to(device), hf_b.to(device), pmask_b.to(device)
            )
            al, fl = model(X_b, pad_mask=pmask_b)
            loss, _, _ = compute_loss(
                al, fl, y_act_b, y_frac_b, hf_b, pmask_b,
                pos_weight_factor, fraction_loss_weight
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_total += loss.item(); n_batches += 1

        scheduler.step()

        # val
        model.eval()
        val_total = 0.0; val_batches = 0
        ent_sum   = 0.0   # Bernoulli entropy of action probabilities
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for X_b, y_act_b, y_frac_b, hf_b, pmask_b in val_loader:
                X_b, y_act_b, y_frac_b, hf_b, pmask_b = (
                    X_b.to(device), y_act_b.to(device),
                    y_frac_b.to(device), hf_b.to(device), pmask_b.to(device)
                )
                al, fl = model(X_b, pad_mask=pmask_b)
                loss, _, _ = compute_loss(
                    al, fl, y_act_b, y_frac_b, hf_b, pmask_b,
                    pos_weight_factor, fraction_loss_weight
                )
                val_total += loss.item(); val_batches += 1

                real  = ~pmask_b
                probs = torch.sigmoid(al.squeeze(-1)[real])
                pred  = (probs > 0.5).long()
                gt    = y_act_b[real].long()
                tp  += ((pred == 1) & (gt == 1)).sum().item()
                fp  += ((pred == 1) & (gt == 0)).sum().item()
                tn  += ((pred == 0) & (gt == 0)).sum().item()
                fn  += ((pred == 0) & (gt == 1)).sum().item()

                # Bernoulli entropy H(p) = -p log p - (1-p) log(1-p)
                # High (~0.693) = uncertain; Low (~0) = confident
                ent = -(probs * (probs + 1e-9).log()
                        + (1 - probs) * (1 - probs + 1e-9).log())
                ent_sum += ent.mean().item()

        prec    = tp / max(1, tp + fp)
        rec     = tp / max(1, tp + fn)
        f1      = 2 * prec * rec / max(1e-9, prec + rec)
        tr_avg  = tr_total  / max(1, n_batches)
        val_avg = val_total / max(1, val_batches)
        val_ent = ent_sum   / max(1, val_batches)

        history.append({'epoch': epoch, 'tr_loss': tr_avg, 'val_loss': val_avg,
                        'entropy': val_ent,
                        'precision': prec, 'recall': rec, 'f1': f1})
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"tr={tr_avg:.4f}  val={val_avg:.4f}  ent={val_ent:.4f}  "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_loss': val_avg,
            'config': {
                'd_model': d_model, 'nhead': nhead,
                'num_layers': num_layers, 'dropout': dropout,
            },
        }
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt, out_dir / 'best_model.pt')
            print(f"  saved best (val={val_avg:.4f})")
        torch.save(ckpt, out_dir / 'latest_model.pt')

    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val loss: {best_val_loss:.4f}  checkpoints -> {out_dir}")
    return model, history


# ════════════════════════════════════════════════════════════════════════════
# LOADER
# ════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str,
               device: str = 'cpu') -> EdgeTransformerNet:
    """Load an EdgeTransformerNet checkpoint."""
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = ckpt['config']
    model = EdgeTransformerNet(
        d_model=cfg.get('d_model', 128),
        nhead=cfg.get('nhead', 4),
        num_layers=cfg.get('num_layers', 3),
        dropout=cfg.get('dropout', 0.10),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    model.to(device)
    epoch    = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"Loaded EdgeTransformerNet  epoch={epoch}  val_loss={val_loss:.4f}")
    return model


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train EdgeTransformerNet')
    parser.add_argument('--dataset',    required=True, type=str)
    parser.add_argument('--out_dir',    default='checkpoints', type=str)
    parser.add_argument('--d_model',    default=128,  type=int)
    parser.add_argument('--nhead',      default=4,    type=int)
    parser.add_argument('--num_layers', default=3,    type=int)
    parser.add_argument('--dropout',    default=0.10, type=float)
    parser.add_argument('--batch_size', default=32,   type=int,
                        help='Turns per batch (not edges)')
    parser.add_argument('--lr',         default=3e-4, type=float)
    parser.add_argument('--epochs',     default=40,   type=int)
    parser.add_argument('--device',     default='auto', type=str)
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )
