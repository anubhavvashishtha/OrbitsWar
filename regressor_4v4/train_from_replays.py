"""
regressor_4v4/train_from_replays.py
=====================================
WHAT IT DOES
------------
Same pipeline as regressor/train_from_replays.py but for 4-player FFA only.
Uses 44-dim features from extract_features_4v4() instead of 30-dim 1v1 features.

LABEL LOGIC (continuous: V in [0, 1])
--------------------------------------
Backward TD pass through each player's trajectory:

    V(s_T)   = won          (1.0 if you won, 0.0 if you lost/tied)
    V(s_t)   = shaped_r + gamma * V(s_{t+1})

In FFA (4 players), only ONE player wins so won=1.0 is rare.
The shaped reward Δships/Δprod/Δplanet nudges mid-game values:

    dominating at turn 480  →  V ≈ 0.97
    neutral at turn 100     →  V ≈ 0.25   (4-player baseline is ~0.25, not 0.5)
    losing at turn 480      →  V ≈ 0.03

Targets are logit-transformed before fitting — same _sig() inference as classifier.

OUTPUT
------
Writes regressor_4v4/value_fn_4v4.py  (GLOBAL trees only)
Returns True on success, False if not enough data.

DATA FLOW
---------
records/*.json  →  _parse_replay_td()  →  extract_features_4v4() every step
    →  _compute_td_values() backward pass  →  fit GBR on logit(V)
    →  value_fn_4v4.py

No bot games are played here — purely offline on saved replay files.
Fine-tune vs bots: python regressor_4v4/train_vs_bots.py

KEY FUNCTIONS (importable by train_vs_bots.py)
----------------------------------------------
get_td_trajectories()              → list of {traj, won, n_agents}
get_val_trajectories()             → val split of same
train_and_export(trajectories, val_trajectories, gamma, reward_scale)
                                   → True/False, writes value_fn_4v4.py

Usage
-----
    python regressor_4v4/train_from_replays.py
    python regressor_4v4/train_from_replays.py --gamma 0.99 --reward-scale 0.1
"""
import os, sys, json, random, pickle
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

# Feature extraction lives in heuristic_4v4 (shared, no duplication)
_h4_dir = str(PROJECT_ROOT / "heuristic_4v4")
if _h4_dir not in sys.path:
    sys.path.insert(0, _h4_dir)
from train_4v4 import extract_features_4v4, snap, split_replay_files

VALUEFN       = str(ROOT / "value_fn_4v4.py")
TD_CACHE_FILE = str(ROOT / "replay_td_cache_4v4.pkl")
MODEL_PKL     = str(ROOT / "model.pkl")

TD_SNAPSHOT_EVERY = 1
N_ESTIMATORS      = 600
MAX_DEPTH         = 5
LEARNING_RATE     = 0.003


def _parse_replay_td(rf):
    """Parse one replay JSON into 4-player per-player feature trajectories."""
    with open(rf, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    if not steps:
        return []

    n_agents  = len(steps[0])
    if n_agents != 4:           # regressor_4v4/ is 4-player only
        return []

    last_step = steps[-1]
    rewards   = [a.get("reward", 0) or 0 for a in last_step]
    max_r     = max(rewards)
    winner    = -1 if (rewards.count(max_r) > 1 or max_r <= 0) else rewards.index(max_r)

    trajectories = [[] for _ in range(n_agents)]
    for step_idx, step_data in enumerate(steps):
        if step_idx % TD_SNAPSHOT_EVERY != 0:
            continue
        obs       = step_data[0].get("observation", {})
        rp        = obs.get("planets", [])
        ci        = set(obs.get("comet_planet_ids", []) or [])
        rf_fleets = obs.get("fleets", []) or []
        if not isinstance(rp, list) or not rp:
            continue
        sd = snap(rp, ci)
        if not sd:
            continue
        for i in range(n_agents):
            trajectories[i].append(extract_features_4v4(sd, i, step_idx, n_agents, fleets=rf_fleets))

    result = []
    for i, traj in enumerate(trajectories):
        if len(traj) >= 2:
            result.append({"traj": traj, "won": 1.0 if i == winner else 0.0, "n_agents": 4})
    return result


def get_td_trajectories(replay_files=None):
    """Load 4-player per-player feature trajectories from replay files (train split)."""
    if replay_files is None:
        train_files, _ = split_replay_files()
        replay_files = train_files

    cache = {}
    if os.path.exists(TD_CACHE_FILE):
        try:
            with open(TD_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    all_trajectories = []
    cache_updated    = False
    n_cached = n_fresh = 0

    for i, rf in enumerate(replay_files):
        cache_key = f"td4v4|{rf}|{os.path.getmtime(rf):.0f}"
        if cache_key in cache:
            all_trajectories.extend(cache[cache_key])
            n_cached += 1
        else:
            try:
                trajs = _parse_replay_td(rf)
                cache[cache_key] = trajs
                all_trajectories.extend(trajs)
                cache_updated = True
                n_fresh += 1
            except Exception as e:
                print(f"  Error parsing {os.path.basename(rf)}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == len(replay_files):
            sys.stdout.write(
                f"\r  TD parse {i+1}/{len(replay_files)}"
                f"  (cached:{n_cached}  fresh:{n_fresh})   "
            )
            sys.stdout.flush()

    print()

    if cache_updated:
        try:
            with open(TD_CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)
            sz = os.path.getsize(TD_CACHE_FILE) / 1024 / 1024
            print(f"  TD cache saved -> {TD_CACHE_FILE}  ({sz:.1f} MB)")
        except Exception as e:
            print(f"  WARN: Could not save TD cache: {e}")

    return all_trajectories


def get_val_trajectories():
    """Load held-out validation trajectories (20% file split)."""
    _, val_files = split_replay_files()
    return get_td_trajectories(replay_files=val_files)


def _compute_td_values(trajectories, gamma=0.99, reward_scale=0.1):
    """
    Backward-pass TD returns for every step of every trajectory.
        V(s_T) = won
        V(s_t) = shaped_r(t,t+1) + gamma * V(s_{t+1})
    Returns list of (feats, V) pairs, V in [0, 1].
    Feature indices: 2=planet_share, 4=ship_share, 6=prod_share (same as 1v1).
    """
    pairs = []
    for td in trajectories:
        traj = td["traj"]
        won  = td["won"]
        T    = len(traj)

        shaped = [0.0] * T
        for t in range(1, T):
            shaped[t] = reward_scale * (
                0.4 * (traj[t][4] - traj[t-1][4]) +
                0.4 * (traj[t][6] - traj[t-1][6]) +
                0.2 * (traj[t][2] - traj[t-1][2])
            )

        V = [0.0] * T
        V[T - 1] = won
        for t in range(T - 2, -1, -1):
            V[t] = shaped[t + 1] + gamma * V[t + 1]

        for t in range(T):
            pairs.append((traj[t], V[t]))

    return pairs


def train_and_export(trajectories, val_trajectories=None,
                     gamma=0.99, reward_scale=0.1,
                     n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                     learning_rate=LEARNING_RATE,
                     base_model=None, n_extra=200):
    """
    Train GBR on TD targets and export trees to value_fn_4v4.py.
    `trajectories`: list of {'traj': [feats...], 'won': float, 'n_agents': int}
    Pass `base_model` to warm-start from an existing GBR instead of training from scratch.
    """
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error
        import joblib
    except ImportError:
        print("pip install scikit-learn numpy joblib"); return False

    if not trajectories:
        print("No trajectories to train on."); return False

    warm = base_model is not None

    print("Computing TD regression targets ...")
    train_pairs = _compute_td_values(trajectories, gamma=gamma, reward_scale=reward_scale)
    random.shuffle(train_pairs)

    X_train = np.array([p[0] for p in train_pairs], dtype=np.float32)
    y_train = np.array([p[1] for p in train_pairs], dtype=np.float32)

    eps = 1e-6
    logit_train = np.log(np.clip(y_train, eps, 1-eps) / (1 - np.clip(y_train, eps, 1-eps)))

    X_val = y_val = None
    if val_trajectories:
        val_pairs = _compute_td_values(val_trajectories, gamma=gamma, reward_scale=reward_scale)
        X_val     = np.array([p[0] for p in val_pairs], dtype=np.float32)
        y_val     = np.array([p[1] for p in val_pairs], dtype=np.float32)

    print(f"Train: {len(X_train):,} samples")
    print(f"V range: [{y_train.min():.3f}, {y_train.max():.3f}]  mean={y_train.mean():.3f}")

    if len(X_train) < 200:
        print(f"Too few samples ({len(X_train)}). Skipping."); return False

    if warm:
        print(f"\n  GLOBAL  {len(X_train):,} train  (warm-start +{n_extra} trees) ...")
        reg = base_model
        reg.set_params(warm_start=True, n_estimators=reg.n_estimators + n_extra)
    else:
        print(f"\n  GLOBAL  {len(X_train):,} train  ...")
        reg = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8,
            min_samples_leaf=30, random_state=42,
            loss="squared_error", verbose=1,
        )
    reg.fit(X_train, logit_train)

    if X_val is not None and len(X_val) >= 10:
        pred_prob = 1.0 / (1.0 + np.exp(-reg.predict(X_val)))
        val_mse   = mean_squared_error(y_val, pred_prob)
        baseline  = mean_squared_error(y_val, np.full(len(y_val), y_train.mean()))
        print(f"val_MSE={val_mse:.4f}  baseline={baseline:.4f}  "
              f"improve={100*(1-val_mse/baseline):.1f}%")

    init_val = float(logit_train.mean())
    lines = ["# AUTO-GENERATED by regressor_4v4/train_from_replays.py -- do not edit", ""]
    lines.append(f"GLOBAL_INIT={init_val:.8f}")
    lines.append(f"GLOBAL_N_TREES={len(reg.estimators_)}")
    lines.append("GLOBAL_TREES=[")
    for ta in reg.estimators_:
        t  = ta[0].tree_
        fi = [int(v)                          for v in t.feature]
        th = [round(float(v), 6)              for v in t.threshold]
        va = [round(float(t.value[i,0,0]), 8) for i in range(t.node_count)]
        le = [int(v)                          for v in t.children_left]
        ri = [int(v)                          for v in t.children_right]
        lines.append(f"    ({fi},{th},{va},{le},{ri}),")
    lines.append("]")
    lines.append("")
    lines.append("VALUE_FN_MODELS=['GLOBAL']")
    lines.append("VALUE_FN_4V4_LOADED=True")
    lines.append("")

    with open(VALUEFN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sz = os.path.getsize(VALUEFN) / 1024
    print(f"\nExported -> {VALUEFN}  ({sz:.1f} KB)")

    try:
        joblib.dump(reg, MODEL_PKL)
        print(f"Saved    -> {MODEL_PKL}")
    except Exception as e:
        print(f"[warn] Could not save model.pkl: {e}")

    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train 4v4 TD regressor from replays")
    ap.add_argument("--gamma",        type=float, default=0.99)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--n-estimators", type=int,   default=N_ESTIMATORS)
    ap.add_argument("--max-depth",    type=int,   default=MAX_DEPTH)
    args = ap.parse_args()

    print("=" * 60)
    print("  regressor_4v4/train_from_replays: 4v4 TD regressor")
    print("=" * 60)

    print("\nLoading train trajectories ...")
    train_trajs = get_td_trajectories()
    print(f"  {len(train_trajs)} trajectories")

    print("\nLoading val trajectories ...")
    val_trajs = get_val_trajectories()
    print(f"  {len(val_trajs)} val trajectories")

    ok = train_and_export(
        train_trajs, val_trajectories=val_trajs,
        gamma=args.gamma, reward_scale=args.reward_scale,
        n_estimators=args.n_estimators, max_depth=args.max_depth,
    )
    print("\nvalue_fn_4v4.py updated." if ok else "\nTraining failed.")


if __name__ == "__main__":
    main()
