"""
classifier/train_1v1_vs_bots.py
================================
Fine-tune the 1v1 value function by playing live games against enemy bots,
then retraining the GBC on combined replay + live-game data.

Pipeline
--------
  1. Load replay training records (seed data) via get_balanced_replay_records()
  2. Discover enemy agents in enemy agents/*/main.py  (skips the my/ folder)
  3. Play --games-per-enemy 1v1 games vs each enemy with the current value_fn_1v1 active
  4. Collect (features, outcome) records at every --snapshot-every steps
  5. Combine replay records + bot-game records, retrain GBC, export value_fn_1v1.py

Prerequisites
-------------
Run classifier/train_from_replays.py first so the replay cache and value_fn_1v1.py
already exist (the cache is the seed; the live bot uses the current value function).

Usage
-----
    python classifier/train_1v1_vs_bots.py
    python classifier/train_1v1_vs_bots.py --games-per-enemy 15
    python classifier/train_1v1_vs_bots.py --games-per-enemy 5 --snapshot-every 3
"""
import os, sys, glob, argparse
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

# train_from_replays is in the same directory; project root for main.py
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

MAIN_PY        = str(PROJECT_ROOT / "main.py")
ENEMY_DIR      = PROJECT_ROOT / "enemy agents"
SNAPSHOT_EVERY = 5


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _ensure_path():
    """Ensure classifier/ is on sys.path so main.py can import value_fn_1v1."""
    cdir = str(ROOT)
    if cdir not in sys.path:
        sys.path.insert(0, cdir)


def _reload_main():
    """Drop cached main / value_fn modules so the next game loads them fresh."""
    for k in list(sys.modules):
        if any(x in k for x in ("value_fn", "main", "agent_4v4")):
            del sys.modules[k]


def collect_1v1_records(enemy_path, games, snapshot_every):
    """
    Play `games` 1v1 matches: our main.py (player 0) vs the enemy.
    Returns a list of {'feats', 'won', 'turn', 'n_agents'} records.
    """
    from kaggle_environments import make
    from train_from_replays import extract_features, snap

    records = []
    for g in range(games):
        _ensure_path()
        _reload_main()

        seed  = 3001 + g * 13
        env   = make("orbit_wars",
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
        won   = 1 if (rs[0] == max_r and rs.count(max_r) == 1) else 0
        tag   = "WIN" if won else ("TIE" if rs[0] == max_r else "LOSS")
        print(f"    game {g+1}/{games}  {tag}")

        for step_idx, step_states in enumerate(steps):
            if step_idx % snapshot_every != 0:
                continue
            obs = _read(step_states[0], "observation")
            if obs is None:
                continue
            rp = _read(obs, "planets", []) or []
            ci = set(_read(obs, "comet_planet_ids", []) or [])
            rf = _read(obs, "fleets", []) or []
            sd = snap(rp, ci)
            if not sd:
                continue
            feats = extract_features(sd, 0, step_idx, 2, fleets=rf)
            records.append({"feats": feats, "won": won, "turn": step_idx, "n_agents": 2})

    return records


def main():
    ap = argparse.ArgumentParser(description="Fine-tune 1v1 value function vs enemy bots")
    ap.add_argument("--games-per-enemy", type=int, default=10,
                    help="Games to play against each enemy (default: 10)")
    ap.add_argument("--snapshot-every",  type=int, default=SNAPSHOT_EVERY,
                    help="Collect a record every N steps (default: 5)")
    args = ap.parse_args()

    import joblib
    from train_from_replays import get_val_records, train_and_export, MODEL_PKL

    print("=" * 60)
    print("  train_1v1_vs_bots: warm-start fine-tune 1v1 value function")
    print("=" * 60)

    val_records = get_val_records()
    print(f"  {len(val_records)} validation records loaded")

    base_models = None
    if os.path.exists(MODEL_PKL):
        base_models = joblib.load(MODEL_PKL)
        print(f"  Loaded base models from {MODEL_PKL}")
    else:
        print(f"  WARNING: {MODEL_PKL} not found — will train from scratch on bot data.")

    _std = [p for p in glob.glob(str(ENEMY_DIR / "*" / "main.py"))
            if Path(p).parent.name != "my"]
    _my  = [p for p in glob.glob(str(ENEMY_DIR / "my" / "*.py"))
            if not Path(p).name.startswith("__")]
    enemy_paths = sorted(_std + _my)
    if not enemy_paths:
        print("ERROR: No enemy agents found in 'enemy agents/'.")
        sys.exit(1)

    print(f"\nFound {len(enemy_paths)} enemy agents:")
    for ep in enemy_paths:
        p = Path(ep)
        print(f"  - {p.stem if p.parent.name == 'my' else p.parent.name}")

    bot_records = []
    for ep in enemy_paths:
        name = Path(ep).parent.name
        print(f"\n[vs {name}]  {args.games_per_enemy} games ...")
        try:
            recs = collect_1v1_records(ep, args.games_per_enemy, args.snapshot_every)
            wins = sum(r["won"] for r in recs)
            print(f"  {wins}/{args.games_per_enemy} wins  |  {len(recs)} steps collected")
            bot_records.extend(recs)
        except Exception as e:
            print(f"  WARN skipping {name}: {e}")

    print(f"\nBot-game dataset: {len(bot_records)} records")

    print("\nFine-tuning GBC (warm-start on bot data) ...")
    ok = train_and_export(bot_records, val_records=val_records, base_models=base_models)
    if ok:
        print("\nvalue_fn_1v1.py updated with bot-game fine-tuning.")
    else:
        print("\nTraining failed — value_fn_1v1.py unchanged.")


if __name__ == "__main__":
    main()
