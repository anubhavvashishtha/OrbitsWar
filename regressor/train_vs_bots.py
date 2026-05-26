"""
regressor/train_vs_bots.py
===========================
Fine-tune the 1v1 TD regressor by playing live games vs enemy bots.

Pipeline
--------
  1. Load replay trajectories (seed) via get_td_trajectories()
  2. Play --games-per-enemy 1v1 games vs every enemy; collect full step trajectories
  3. Combine replay + bot trajectories → call train_and_export() → value_fn_1v1.py

Usage
-----
    python regressor/train_vs_bots.py
    python regressor/train_vs_bots.py --games-per-enemy 10 --gamma 0.99
"""
import os, sys, glob, importlib.util
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

MAIN_PY   = str(PROJECT_ROOT / "main.py")
ENEMY_DIR = PROJECT_ROOT / "enemy agents"

# ── Load classifier's feature extraction (snap / extract_features) ──────────
_clf_dir = str(PROJECT_ROOT / "classifier")
if _clf_dir not in sys.path:
    sys.path.insert(0, _clf_dir)
from train_from_replays import snap, extract_features   # classifier version

# ── Load regressor's train_from_replays via explicit path (avoids name clash) ─
def _load_reg_train():
    spec = importlib.util.spec_from_file_location(
        "_reg_train_from_replays", str(ROOT / "train_from_replays.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_rtfr                = _load_reg_train()
get_td_trajectories  = _rtfr.get_td_trajectories
get_val_trajectories = _rtfr.get_val_trajectories
train_and_export     = _rtfr.train_and_export
MODEL_PKL            = _rtfr.MODEL_PKL


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _reload_main():
    for k in list(sys.modules):
        if any(x in k for x in ("value_fn", "main", "agent_4v4")):
            del sys.modules[k]


def _label(path):
    p = Path(path)
    return p.stem if p.parent.name == "my" else p.parent.name


def collect_bot_trajectories(enemy_path, games):
    """
    Play `games` 1v1 matches vs the enemy.
    Returns list of {'traj': [feats...], 'won': float, 'n_agents': 2}.
    Collects every step (dense TD signal, not sampled).
    """
    from kaggle_environments import make

    trajectories = []
    for g in range(games):
        _reload_main()
        seed = 5001 + g * 17
        env  = make("orbit_wars",
                    configuration={"episodeSteps": 500, "seed": seed},
                    debug=False)
        try:
            steps = env.run([MAIN_PY, enemy_path])
        except Exception as e:
            print(f"    game {g+1} error: {e}")
            continue

        final = steps[-1]
        rs    = [s.reward if s.reward is not None else 0 for s in final]
        max_r = max(rs) if rs else 0
        won   = 1.0 if (rs[0] == max_r and rs.count(max_r) == 1) else 0.0
        tag   = "WIN" if won == 1.0 else ("TIE" if rs[0] == max_r else "LOSS")
        print(f"    game {g+1}/{games}  {tag}")

        traj = []
        for step_idx, step_states in enumerate(steps):
            obs = _read(step_states[0], "observation")
            if obs is None:
                continue
            rp = _read(obs, "planets", []) or []
            ci = set(_read(obs, "comet_planet_ids", []) or [])
            rf = _read(obs, "fleets", []) or []
            sd = snap(rp, ci)
            if not sd:
                continue
            traj.append(extract_features(sd, 0, step_idx, 2, fleets=rf))

        if len(traj) >= 2:
            trajectories.append({"traj": traj, "won": won, "n_agents": 2})

    return trajectories


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fine-tune 1v1 TD regressor vs enemy bots")
    ap.add_argument("--games-per-enemy", type=int,   default=10)
    ap.add_argument("--gamma",           type=float, default=0.99)
    ap.add_argument("--reward-scale",    type=float, default=0.1)
    args = ap.parse_args()

    import joblib

    print("=" * 60)
    print("  regressor/train_vs_bots: warm-start 1v1 TD fine-tune vs bots")
    print("=" * 60)

    val_trajs = get_val_trajectories()
    print(f"  {len(val_trajs)} val trajectories")

    base_model = None
    if os.path.exists(MODEL_PKL):
        base_model = joblib.load(MODEL_PKL)
        print(f"  Loaded base model from {MODEL_PKL}")
    else:
        print(f"  WARNING: {MODEL_PKL} not found — will train from scratch on bot data.")

    _std = [p for p in glob.glob(str(ENEMY_DIR / "*" / "main.py"))
            if Path(p).parent.name != "my"]
    _my  = [p for p in glob.glob(str(ENEMY_DIR / "my" / "*.py"))
            if not Path(p).name.startswith("__")]
    enemy_paths = sorted(_std + _my)
    if not enemy_paths:
        print("ERROR: No enemy agents found."); sys.exit(1)

    print(f"\nFound {len(enemy_paths)} enemies:")
    for ep in enemy_paths:
        print(f"  - {_label(ep)}")

    bot_trajs = []
    for ep in enemy_paths:
        name = _label(ep)
        print(f"\n[vs {name}]  {args.games_per_enemy} games ...")
        try:
            trajs = collect_bot_trajectories(ep, args.games_per_enemy)
            wins  = sum(t["won"] for t in trajs)
            print(f"  {wins:.0f}/{args.games_per_enemy} wins  |  {len(trajs)} trajectories")
            bot_trajs.extend(trajs)
        except Exception as e:
            print(f"  WARN skipping {name}: {e}")

    print(f"\nBot-game dataset: {len(bot_trajs)} trajectories")

    print("\nFine-tuning TD regressor (warm-start on bot data) ...")
    ok = train_and_export(
        bot_trajs, val_trajectories=val_trajs,
        gamma=args.gamma, reward_scale=args.reward_scale,
        base_model=base_model,
    )
    print("\nvalue_fn_1v1.py updated." if ok else "\nTraining failed.")


if __name__ == "__main__":
    main()
