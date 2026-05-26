"""
rl/replay_trainer.py
====================
Pre-train OrbitsNet on Kaggle replay data in two phases:

  Phase A — BCE pretraining
      Supervised learning on balanced (50/50 win/loss) snapshots.
      Gives the value head a calibrated starting point.

  Phase B — Offline TD on replay trajectories (TODO #2)
      Parses consecutive turn pairs from each replay, bootstraps
      TD targets: V(s_t) <- gamma·V(s_{t+1})   (mid-game)
                  V(s_T) <- won              (terminal)
      Adds temporal structure the BCE phase cannot see.

Usage
-----
    python rl/replay_trainer.py            # full Phase A + B
    python rl/replay_trainer.py --resume   # continue from rl/model.pt
    python rl/replay_trainer.py --td-only  # skip BCE, run Phase B only
"""
import os, sys, random, argparse
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classifier"))
sys.path.insert(0, str(PROJECT_ROOT))

# Load get_td_trajectories from regressor/ explicitly to avoid collision with
# classifier/train_from_replays.py (same filename, different module)
import importlib.util as _ilu
_reg_spec = _ilu.spec_from_file_location(
    "_reg_train_from_replays",
    str(PROJECT_ROOT / "regressor" / "train_from_replays.py"))
_reg_mod = _ilu.module_from_spec(_reg_spec)
_reg_spec.loader.exec_module(_reg_mod)
get_td_trajectories = _reg_mod.get_td_trajectories

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH   = ROOT / "model.pt"

# ── Phase A (BCE) hyper-parameters ───────────────────────────────────────────
N_EPOCHS     = 30
BATCH_SIZE   = 512
LR           = 1e-3
WEIGHT_DECAY = 1e-4
VAL_PER_CAT  = 3000

# ── Phase B (offline TD) hyper-parameters ────────────────────────────────────
TD_EPOCHS    = 15
TD_LR        = 5e-4
TD_BATCH     = 512
GAMMA        = 0.99


# ── Phase A helpers ───────────────────────────────────────────────────────────
def load_data():
    from train_from_replays import get_balanced_replay_records, get_val_records
    print("Loading training records...")
    train_recs = get_balanced_replay_records()
    print("Loading validation records...")
    val_recs   = get_val_records(max_per_cat=VAL_PER_CAT)
    return train_recs, val_recs


def to_tensors(records):
    X = torch.FloatTensor([r['feats'][:30] for r in records])
    y = torch.FloatTensor([float(r['won']) for r in records])
    return X, y


# ── Phase B: offline TD ───────────────────────────────────────────────────────
def train_replay_td(model, n_epochs: int = TD_EPOCHS, gamma: float = GAMMA):
    """
    Offline TD(0) on Kaggle replay trajectories.

    Algorithm (fitted value iteration):
      Each epoch:
        1. Freeze bootstrap targets:
             mid-game:  target_t = gamma · V(s_{t+1})   [no_grad]
             terminal:  target_T = won
        2. Mini-batch MSE updates over all (s_t, target_t) pairs.

    The value function slowly learns the temporal structure of the game —
    something BCE training on independent snapshots can never see.
    """
    print("\n" + "=" * 55)
    print("  Phase B — Offline TD on replay trajectories")
    print("=" * 55)
    print("Parsing trajectories (first run builds cache — slow once)...")
    trajectories = get_td_trajectories()
    if not trajectories:
        print("No TD trajectories found — skipping."); return model

    n_trajs = len(trajectories)
    n_steps = sum(len(t["traj"]) for t in trajectories)
    print(f"  {n_trajs} player-trajectories  |  {n_steps:,} total steps")

    # ── Pre-build flattened tensors once ─────────────────────────────────────
    # Mid-game pairs: s_t → s_{t+1}  (all non-terminal steps)
    mid_X_t   = []   # (s_t  features)
    mid_X_t1  = []   # (s_{t+1} features)
    mid_r     = []   # dense in-game reward at step t+1 (event-based)
    # Terminal steps: s_T → won
    term_X    = []
    term_won  = []

    for td in trajectories:
        traj   = td["traj"]            # list of feature vectors
        won    = td["won"]             # 0.0 or 1.0
        shaped = td.get("shaped")      # list of per-step dense rewards (may be None for legacy cache)
        T      = len(traj)
        if T < 2:
            continue
        for t in range(T - 1):
            mid_X_t.append(traj[t][:30])
            mid_X_t1.append(traj[t + 1][:30])
            if shaped is not None and len(shaped) == T:
                mid_r.append(shaped[t + 1])
            else:
                mid_r.append(None)
        term_X.append(traj[T - 1][:30])
        term_won.append(won)

    if not mid_X_t:
        print("No valid step pairs — skipping."); return model

    all_X_t    = torch.FloatTensor(mid_X_t)            # (N_mid, 30)
    all_X_t1   = torch.FloatTensor(mid_X_t1)           # (N_mid, 30)
    all_X_term = torch.FloatTensor(term_X)             # (N_term, 30)
    all_won    = torch.FloatTensor(term_won)           # (N_term,)

    # Pre-stored dense rewards if available (else NaN, then patched per-batch)
    if all(r is not None for r in mid_r):
        all_r = torch.FloatTensor(mid_r)
        _have_dense = True
    else:
        all_r = None
        _have_dense = False

    N_mid  = len(all_X_t)
    N_term = len(all_X_term)
    print(f"  Mid-game TD pairs: {N_mid:,}  |  Terminal episodes: {N_term:,}")
    print(f"  Epochs: {n_epochs}  LR: {TD_LR}  Batch: {TD_BATCH}  gamma: {gamma}")
    print("-" * 55)

    optimizer = optim.Adam(model.parameters(), lr=TD_LR)
    mid_idx   = list(range(N_mid))
    term_idx  = list(range(N_term))

    for epoch in range(1, n_epochs + 1):
        model.train()

        # ── Bootstrap targets with shaped reward (frozen for this epoch) ────
        # Dense reward source:
        #   - Preferred: per-step event reward stored on the trajectory
        #     (planet capture, planet loss, opponent elimination, momentum)
        #   - Fallback: legacy feature-delta shaping when cache lacks 'shaped'
        with torch.no_grad():
            model.eval()
            chunk = 4096
            V_t1_chunks = []
            for start in range(0, N_mid, chunk):
                V_t1_chunks.append(model.value(all_X_t1[start:start + chunk]))
            V_t1 = torch.cat(V_t1_chunks)                      # (N_mid,)

            if _have_dense:
                shaped_r = all_r
            else:
                d_ships   = all_X_t1[:, 4] - all_X_t[:, 4]
                d_prod    = all_X_t1[:, 6] - all_X_t[:, 6]
                d_planets = all_X_t1[:, 2] - all_X_t[:, 2]
                shaped_r  = 0.1 * (0.4 * d_ships + 0.4 * d_prod + 0.2 * d_planets)

            targets_mid  = (shaped_r + gamma * V_t1).clamp(0.0, 1.0)
            targets_term = all_won                              # V(s_T) <- won

        model.train()
        total_loss = 0.0
        n_updates  = 0

        # ── Mid-game updates ───────────────────────────────────────────────
        random.shuffle(mid_idx)
        for start in range(0, N_mid, TD_BATCH):
            b    = mid_idx[start:start + TD_BATCH]
            pred = model.value(all_X_t[b])
            loss = F.mse_loss(pred, targets_mid[b])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_updates  += 1

        # ── Terminal updates ───────────────────────────────────────────────
        random.shuffle(term_idx)
        for start in range(0, N_term, TD_BATCH):
            b    = term_idx[start:start + TD_BATCH]
            pred = model.value(all_X_term[b])
            loss = F.mse_loss(pred, targets_term[b])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_updates  += 1

        avg_loss = total_loss / max(1, n_updates)
        # Value calibration check: mean predicted value at terminal steps
        with torch.no_grad():
            model.eval()
            term_pred = model.value(all_X_term).mean().item()
        model.train()

        print(f"  TD Epoch {epoch:2d}/{n_epochs}  "
              f"avg_td_loss={avg_loss:.5f}  "
              f"mean_V(terminal)={term_pred:.3f}  "
              f"(target={all_won.mean().item():.3f})")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nPhase B complete — model saved -> {MODEL_PATH}")
    return model


# ── Phase A: BCE pretraining ──────────────────────────────────────────────────
def train(resume: bool = False, skip_td: bool = False):
    if not TORCH_AVAILABLE:
        print("Install PyTorch first:  pip install torch"); return None

    from net import OrbitsNet

    # ── Phase A: BCE ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Phase A — BCE pretraining on replay snapshots")
    print("=" * 55)

    train_recs, val_recs = load_data()
    if not train_recs:
        print("No replay records found. Run classifier/train_from_replays.py first.")
        return None

    X_train, y_train = to_tensors(train_recs)
    X_val,   y_val   = to_tensors(val_recs) if val_recs else (None, None)

    model = OrbitsNet()
    if resume and MODEL_PATH.exists():
        try:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            print(f"Resumed from {MODEL_PATH}")
        except Exception as e:
            print(f"Could not load {MODEL_PATH} ({e}) — starting fresh.")

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
            loss = criterion(model.value(xb), yb)
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
                    v_pred = model.value(X_val)
                    vl     = criterion(v_pred, y_val).item()
                    vacc   = ((v_pred > 0.5) == y_val.bool()).float().mean().item()
                tag = f"  val_loss={vl:.4f}  val_acc={vacc:.3f}"
                if vl < best_val:
                    best_val = vl
                    torch.save(model.state_dict(), MODEL_PATH)
                    tag += " *"
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  "
                  f"train_loss={total_loss/len(X_train):.4f}{tag}")

    if X_val is None:
        torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nPhase A complete — best val_loss={best_val:.4f}")

    # ── Phase B: offline TD ───────────────────────────────────────────────────
    if not skip_td:
        model = train_replay_td(model)

    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume",  action="store_true", help="Continue from existing model.pt")
    p.add_argument("--td-only", action="store_true", help="Skip Phase A, run offline TD only")
    args = p.parse_args()

    if args.td_only:
        if not TORCH_AVAILABLE:
            print("Install PyTorch first."); raise SystemExit(1)
        from net import OrbitsNet
        m = OrbitsNet()
        if MODEL_PATH.exists():
            try:
                m.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
                print(f"Loaded {MODEL_PATH}")
            except Exception as e:
                print(f"Could not load model ({e}) — starting fresh.")
        train_replay_td(m)
    else:
        train(resume=args.resume)
