"""
regressor/train_from_replays.py
================================
WHAT IT DOES
------------
Reads saved Kaggle replay JSON files from records/*.json,
extracts a feature snapshot at EVERY turn (TD_SNAPSHOT_EVERY=1),
and trains a GradientBoostingRegressor (GBR) on TD-discounted returns.

LABEL LOGIC (continuous: V in [0, 1])
--------------------------------------
Unlike the classifier which labels the whole game 0 or 1,
the regressor computes a per-step value via backward TD pass:

    V(s_T)   = won          (1.0 if you won, 0.0 if you lost)
    V(s_t)   = shaped_r + gamma * V(s_{t+1})

    shaped_r = reward_scale * (
        0.4 * Δship_share  +
        0.4 * Δprod_share  +
        0.2 * Δplanet_share
    )

Result:
    dominating at turn 480  →  V ≈ 0.97   (nearly certain win)
    ahead at turn 200       →  V ≈ 0.60   (moderate advantage)
    neutral at turn 100     →  V ≈ 0.50   (coin flip)
    behind at turn 200      →  V ≈ 0.40   (moderate deficit)
    losing at turn 480      →  V ≈ 0.03   (nearly certain loss)

Targets are logit-transformed before fitting so _sig() in main.py
converts raw tree output back to [0,1] — same inference code as classifier.

OUTPUT
------
Writes regressor/value_fn_1v1.py  (GLOBAL trees only, no EARLY/LATE/MID split)
Returns True on success, False if not enough data.

DATA FLOW
---------
records/*.json  →  _parse_replay_td()  →  extract_features() every step
    →  _compute_td_values() backward pass  →  fit GBR on logit(V)
    →  value_fn_1v1.py

No bot games are played here — purely offline on saved replay files.
Fine-tune vs bots: python regressor/train_vs_bots.py

KEY FUNCTIONS (importable by train_vs_bots.py)
----------------------------------------------
get_td_trajectories()              → list of {traj, won, n_agents}
get_val_trajectories()             → val split of same
train_and_export(trajectories, val_trajectories, gamma, reward_scale)
                                   → True/False, writes value_fn_1v1.py

Usage
-----
    python regressor/train_from_replays.py
    python regressor/train_from_replays.py --gamma 0.99 --reward-scale 0.1
"""
import os, sys, json, random, pickle
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

# Feature extraction lives in the classifier module (shared, no duplication)
_clf_dir = str(PROJECT_ROOT / "classifier")
if _clf_dir not in sys.path:
    sys.path.insert(0, _clf_dir)
from train_from_replays import extract_features, snap, split_replay_files

VALUEFN       = str(ROOT / "value_fn_1v1.py")
TD_CACHE_FILE = str(ROOT / "replay_td_cache_v3.pkl")
MODEL_PKL     = str(ROOT / "model.pkl")

TD_SNAPSHOT_EVERY = 1       # every turn — dense transitions for TD
N_ESTIMATORS      = 1200
MAX_DEPTH         = 7
LEARNING_RATE     = 0.005


# ── Dense in-game reward constants (must match rl/self_play_trainer.py) ─────
# These reward the value function for the actual game-events it should care
# about, not just feature-delta noise. See self_play_trainer.py for full
# documentation of what each constant means.
CAPTURE_BASE        = 0.06
CAPTURE_FROM_ENEMY  = 0.04
PLANET_LOSS         = 0.06
ELIMINATION_BONUS   = 0.30
MOMENTUM_SCALE      = 0.02


def _alive_owners(sd, fleets, ignore_player):
    """Owners that still have a planet or a fleet (excluding `ignore_player` and -1)."""
    alive = set()
    for p in sd:
        ow = p.get("owner", -1)
        if ow != -1 and ow != ignore_player:
            alive.add(ow)
    for f in (fleets or []):
        fow = f[1] if isinstance(f, (list, tuple)) else f.get("owner", -1)
        if fow != -1 and fow != ignore_player:
            alive.add(fow)
    return alive


def _event_reward(prev_sd, curr_sd, prev_alive, curr_alive, player):
    """Per-turn shaped reward from planet captures/losses + opponent eliminations."""
    if prev_sd is None:
        return 0.0
    prev_owner = {p["id"]: p.get("owner", -1)            for p in prev_sd}
    prev_prod  = {p["id"]: float(p.get("production", 0)) for p in prev_sd}
    r = 0.0
    for cp in curr_sd:
        pid = cp["id"]
        if pid not in prev_owner:
            continue
        po = prev_owner[pid]
        co = cp.get("owner", -1)
        if co == po:
            continue
        prod = max(prev_prod.get(pid, 1.0), float(cp.get("production", 0)))
        if co == player and po != player:
            r += CAPTURE_BASE * prod
            if po != -1:
                r += CAPTURE_FROM_ENEMY * prod
        elif po == player and co != player:
            r -= PLANET_LOSS * prod
    r += ELIMINATION_BONUS * len(prev_alive - curr_alive)
    return r


def _parse_replay_td(rf):
    """Parse one replay JSON into 1v1 per-player feature trajectories + dense rewards."""
    with open(rf, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    if not steps:
        return []

    n_agents  = len(steps[0])


    last_step = steps[-1]
    rewards   = [a.get("reward", 0) or 0 for a in last_step]
    max_r     = max(rewards)
    winner    = -1 if (rewards.count(max_r) > 1 or max_r <= 0) else rewards.index(max_r)

    trajectories = [[] for _ in range(n_agents)]
    shaped_per_player = [[] for _ in range(n_agents)]
    prev_sd_player    = [None] * n_agents
    prev_alive_player = [set() for _ in range(n_agents)]
    prev_share_player = [None] * n_agents

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

        tot_s = sum(float(p.get("ships", 0))      for p in sd) or 1.0
        tot_p = sum(float(p.get("production", 0)) for p in sd) or 1.0
        tot_pl = float(len(sd)) or 1.0

        for i in range(n_agents):
            trajectories[i].append(extract_features(sd, i, step_idx, n_agents, fleets=rf_fleets))
            curr_alive = _alive_owners(sd, rf_fleets, ignore_player=i)
            r_event = _event_reward(prev_sd_player[i], sd, prev_alive_player[i], curr_alive, i)
            my_s = sum(float(p.get("ships", 0))      for p in sd if p.get("owner") == i)
            my_p = sum(float(p.get("production", 0)) for p in sd if p.get("owner") == i)
            my_pl = sum(1 for p in sd if p.get("owner") == i)
            curr_share = (my_s / tot_s, my_p / tot_p, my_pl / tot_pl)
            if prev_share_player[i] is not None:
                ds, dp, dpl = (a - b for a, b in zip(curr_share, prev_share_player[i]))
                r_event += MOMENTUM_SCALE * (0.4 * ds + 0.4 * dp + 0.2 * dpl)
            shaped_per_player[i].append(r_event)
            prev_sd_player[i]    = sd
            prev_alive_player[i] = curr_alive
            prev_share_player[i] = curr_share

    result = []
    for i, traj in enumerate(trajectories):
        if len(traj) >= 2:
            result.append({
                "traj":     traj,
                "won":      1.0 if i == winner else 0.0,
                "n_agents": n_agents,
                "shaped":   shaped_per_player[i],
            })
    return result


def get_td_trajectories(replay_files=None):
    """Load 1v1 per-player feature trajectories from replay files (train split)."""
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
        # v2: trajectory dicts now include a 'shaped' field (per-step event rewards).
        # Old v1 entries silently miss; they get re-parsed and overwritten.
        cache_key = f"td|v3|{rf}|{os.path.getmtime(rf):.0f}"
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

    Prefers the per-step dense rewards stored on the trajectory dict
    (planet capture / loss / opponent elimination events). Falls back to
    the legacy feature-delta shaping when missing (old cache entries).
    Returns list of (feats, V) pairs, V clipped to [0, 1].
    """
    pairs = []
    for td in trajectories:
        traj   = td["traj"]
        won    = td["won"]
        shaped = td.get("shaped")
        T      = len(traj)

        if shaped is None or len(shaped) != T:
            # Legacy path — feature-delta shaping (matches old cache entries).
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
            v = shaped[t + 1] + gamma * V[t + 1]
            V[t] = max(0.0, min(1.0, v))

        for t in range(T):
            pairs.append((traj[t], V[t]))

    return pairs


def train_and_export(trajectories, val_trajectories=None,
                     gamma=0.99, reward_scale=0.1,
                     n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                     learning_rate=LEARNING_RATE,
                     base_model=None, n_extra=200):
    """
    Train GBR on TD targets and export trees to value_fn_1v1.py.
    `trajectories`: list of {'traj': [feats...], 'won': float, 'n_agents': int}
    Pass `base_model` to warm-start from an existing GBR instead of training from scratch.
    """
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        import joblib
    except ImportError:
        print("pip install scikit-learn numpy joblib"); return False

    if not trajectories:
        print("No trajectories to train on."); return False

    warm = base_model is not None
    base_models = {}
    if warm:
        if isinstance(base_model, dict):
            base_models = base_model
        else:
            base_models = {"GLOBAL": base_model}

    print("Computing TD regression targets ...")
    train_pairs = _compute_td_values(trajectories, gamma=gamma, reward_scale=reward_scale)
    random.shuffle(train_pairs)

    X_train = np.array([p[0] for p in train_pairs], dtype=np.float32)
    y_train = np.array([p[1] for p in train_pairs], dtype=np.float32)
    turns_train = np.round(X_train[:, 0] * 500.0).astype(np.int32)

    eps = 1e-6
    logit_train = np.log(np.clip(y_train, eps, 1-eps) / (1 - np.clip(y_train, eps, 1-eps)))

    X_val = y_val = turns_val = None
    if val_trajectories:
        val_pairs = _compute_td_values(val_trajectories, gamma=gamma, reward_scale=reward_scale)
        X_val     = np.array([p[0] for p in val_pairs], dtype=np.float32)
        y_val     = np.array([p[1] for p in val_pairs], dtype=np.float32)
        turns_val = np.round(X_val[:, 0] * 500.0).astype(np.int32)

    print(f"Train: {len(X_train):,} samples")
    print(f"V range: [{y_train.min():.3f}, {y_train.max():.3f}]  mean={y_train.mean():.3f}")

    if len(X_train) < 200:
        print(f"Too few samples ({len(X_train)}). Skipping."); return False

    def _tr(mask, mask_val, ne, d, base_clf=None):
        Xs, ys = X_train[mask], logit_train[mask]
        ys_raw = y_train[mask]
        if len(Xs) < 80:
            return None, 0.0, 0.0, 0.0
        if base_clf is not None:
            clf = base_clf
            clf.set_params(warm_start=True, n_estimators=clf.n_estimators + n_extra)
            clf.fit(Xs, ys)
            pred_train = 1.0 / (1.0 + np.exp(-clf.predict(Xs)))
            train_mse = mean_squared_error(ys_raw, pred_train)
            if X_val is not None and mask_val is not None and mask_val.sum() >= 10:
                pred_val = 1.0 / (1.0 + np.exp(-clf.predict(X_val[mask_val])))
                val_mse = mean_squared_error(y_val[mask_val], pred_val)
                baseline = mean_squared_error(y_val[mask_val], np.full(len(y_val[mask_val]), ys_raw.mean()))
                imp = 100 * (1 - val_mse / baseline)
            else:
                val_mse = train_mse
                baseline = mean_squared_error(ys_raw, np.full(len(ys_raw), ys_raw.mean()))
                imp = 100 * (1 - train_mse / baseline)
        else:
            Xt,Xv,yt,yv = train_test_split(Xs, ys, test_size=0.2, random_state=42)
            clf = GradientBoostingRegressor(
                n_estimators=ne, max_depth=d,
                learning_rate=learning_rate, subsample=0.8,
                min_samples_leaf=15, random_state=42,
                loss="squared_error",
                validation_fraction=0.1, n_iter_no_change=20, tol=1e-4,
                verbose=1
            )
            clf.fit(Xt, yt)
            pred_train = 1.0 / (1.0 + np.exp(-clf.predict(Xs)))
            train_mse = mean_squared_error(ys_raw, pred_train)
            pred_val = 1.0 / (1.0 + np.exp(-clf.predict(Xv)))
            yv_raw = 1.0 / (1.0 + np.exp(-yv))
            val_mse = mean_squared_error(yv_raw, pred_val)
            baseline = mean_squared_error(yv_raw, np.full(len(yv_raw), yv_raw.mean()))
            imp = 100 * (1 - val_mse / baseline)
        return clf, train_mse, val_mse, imp

    specs = [
        ("GLOBAL", np.ones(len(X_train), bool),  np.ones(len(X_val), bool) if X_val is not None else None, N_ESTIMATORS, MAX_DEPTH),
        ("EARLY",  turns_train <= 100,           turns_val <= 100 if X_val is not None else None,          1000,         7),
        ("LATE",   turns_train >= 300,           turns_val >= 300 if X_val is not None else None,          300,          5),
        ("MID",    (turns_train > 100) & (turns_train < 300), (turns_val > 100) & (turns_val < 300) if X_val is not None else None, 500, 5),
    ]

    lines = ["# AUTO-GENERATED by regressor/train_from_replays.py -- do not edit", ""]
    fitted = {}
    exported = []

    for name, mask, mask_val, ne, d in specs:
        print(f"\n  {name:<8} {'(warm-start)' if warm and base_models.get(name) else ''}...")
        base_clf = base_models.get(name) if warm else None
        clf, train_mse, val_mse, imp = _tr(mask, mask_val, ne, d, base_clf=base_clf)
        if clf is None:
            print("    Skipped (not enough data)")
            continue
        print(f"    Train MSE={train_mse:.4f}  Val MSE={val_mse:.4f}  Improve={imp:.1f}%  ({mask.sum()} samples)")
        
        init_val = float(logit_train[mask].mean())
        lines.append(f"{name}_INIT={init_val:.8f}")
        lines.append(f"{name}_N_TREES={len(clf.estimators_)}")
        lines.append(f"{name}_TREES=[")
        for ta in clf.estimators_:
            t  = ta[0].tree_
            fi = [int(v)                          for v in t.feature]
            th = [round(float(v), 6)              for v in t.threshold]
            va = [round(float(t.value[i,0,0]), 8) for i in range(t.node_count)]
            le = [int(v)                          for v in t.children_left]
            ri = [int(v)                          for v in t.children_right]
            lines.append(f"    ({fi},{th},{va},{le},{ri}),")
        lines.append("]")
        lines.append("")
        exported.append(name)
        fitted[name] = clf

    lines.append(f"VALUE_FN_MODELS={exported!r}")
    lines.append("VALUE_FN_LOADED=True")
    lines.append("")

    with open(VALUEFN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sz = os.path.getsize(VALUEFN) / 1024
    print(f"\nExported -> {VALUEFN}  ({sz:.1f} KB)")

    try:
        joblib.dump(fitted, MODEL_PKL)
        print(f"Saved    -> {MODEL_PKL}")
    except Exception as e:
        print(f"[warn] Could not save model.pkl: {e}")

    return True


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Train 1v1 TD regressor from replays")
    ap.add_argument("--gamma",        type=float, default=0.99)
    ap.add_argument("--reward-scale", type=float, default=0.1)
    ap.add_argument("--n-estimators", type=int,   default=N_ESTIMATORS)
    ap.add_argument("--max-depth",    type=int,   default=MAX_DEPTH)
    args = ap.parse_args()

    print("=" * 60)
    print("  regressor/train_from_replays: 1v1 TD regressor")
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
    print("\nvalue_fn_1v1.py updated." if ok else "\nTraining failed.")


if __name__ == "__main__":
    main()
