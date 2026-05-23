"""
rl/replay_trainer.py
====================
Pre-train OrbitsNet on Kaggle replay data.

Loads the same balanced (50/50 win/loss) records used by the GBM classifier
but trains a neural network with BCE loss instead.  The trained model is
saved to rl/model.pt and is the starting point for self_play_trainer.py.

Usage
-----
    python rl/replay_trainer.py            # fresh train
    python rl/replay_trainer.py --resume   # continue from rl/model.pt
"""
import os, sys, random, argparse
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "classifier"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH   = ROOT / "model.pt"
N_EPOCHS     = 30
BATCH_SIZE   = 512
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_PER_CAT  = 3000


def load_data():
    from train_from_replays import get_balanced_replay_records, get_val_records
    print("Loading training records...")
    train_recs = get_balanced_replay_records()
    print("Loading validation records...")
    val_recs   = get_val_records(max_per_cat=VAL_PER_CAT)
    return train_recs, val_recs


def to_tensors(records):
    X = torch.FloatTensor([r['feats'] for r in records])
    y = torch.FloatTensor([float(r['won']) for r in records])
    return X, y


def train(resume: bool = False):
    if not TORCH_AVAILABLE:
        print("Install PyTorch first:  pip install torch"); return None

    from net import OrbitsNet

    train_recs, val_recs = load_data()
    if not train_recs:
        print("No replay records found. Run classifier/train_from_replays.py first."); return None

    X_train, y_train = to_tensors(train_recs)
    X_val,   y_val   = to_tensors(val_recs) if val_recs else (None, None)

    model = OrbitsNet()
    if resume and MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"Resumed from {MODEL_PATH}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    criterion = nn.BCELoss()
    loader    = DataLoader(TensorDataset(X_train, y_train),
                           batch_size=BATCH_SIZE, shuffle=True)

    n_val = len(X_val) if X_val is not None else 0
    print(f"\nTrain: {len(X_train)}  |  Val: {n_val}")
    print(f"Epochs: {N_EPOCHS}  LR: {LR}  Batch: {BATCH_SIZE}")
    print("-" * 55)

    best_val = float('inf')
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            tag = ""
            if X_val is not None:
                model.eval()
                with torch.no_grad():
                    vl   = criterion(model(X_val), y_val).item()
                    vacc = ((model(X_val) > 0.5) == y_val.bool()).float().mean().item()
                tag = f"  val_loss={vl:.4f}  val_acc={vacc:.3f}"
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), MODEL_PATH)
                    tag += " ✓"
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  train_loss={total_loss/len(X_train):.4f}{tag}")

    if X_val is None:
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nSaved -> {MODEL_PATH}")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true", help="Continue from existing model.pt")
    args = p.parse_args()
    train(resume=args.resume)
