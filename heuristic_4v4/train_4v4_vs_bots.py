"""
heuristic_4v4/train_4v4_vs_bots.py
=====================================
Fine-tune the 4v4 value function using exhaustive C(n,3) opponent coverage,
then mixing in replay records as an anchor before retraining the GBC.

Pipeline
--------
  1. Load 4-player replay records (seed) via get_balanced_replay_records()
  2. Discover all N enemy agents in enemy agents/*/main.py
  3. Generate every unique C(N,3) opponent trio — shuffled so early termination
     still gives diverse coverage
  4. Play one game per trio (our main.py as player 0); repeat for --passes cycles
  5. Collect (features, outcome) records using extract_features_4v4() every N steps
  6. Combine bot-game records + replay anchor, retrain GBC, export value_fn_4v4.py

Why exhaustive C(N,3)?
  Random sampling misses many trios and can repeat others.  Exhaustive coverage
  guarantees every opponent combination is seen exactly once per pass, giving the
  GBC a balanced view of all strategy matchups.

Prerequisites
-------------
Run heuristic_4v4/train_4v4.py first so the replay cache and value_fn_4v4.py exist.

Usage
-----
    python heuristic_4v4/train_4v4_vs_bots.py           # 1 full pass (all C(n,3) combos)
    python heuristic_4v4/train_4v4_vs_bots.py --passes 2
    python heuristic_4v4/train_4v4_vs_bots.py --snapshot-every 3
"""
import os, sys, glob, random, argparse
from itertools import combinations
from math import comb
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
CLASSIFIER_DIR = str(PROJECT_ROOT / "classifier")
if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)

MAIN_PY        = str(PROJECT_ROOT / "main.py")
ENEMY_DIR      = PROJECT_ROOT / "enemy agents"
SNAPSHOT_EVERY = 5


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


def collect_exhaustive_records(enemy_pool, passes, snapshot_every):
    """
    Build the full C(n,3) combo list, shuffle it, then play one game per combo.
    Repeat for `passes` full cycles.  Returns feature records for all games.
    """
    from kaggle_environments import make
    from train_4v4 import extract_features_4v4, snap

    all_combos = list(combinations(enemy_pool, 3))
    n_combos   = len(all_combos)
    total_games = n_combos * passes

    print(f"  {len(enemy_pool)} enemies  →  C({len(enemy_pool)},3) = {n_combos} unique trios"
          f"  ×  {passes} pass(es)  =  {total_games} games")

    records  = []
    game_num = 0

    for p in range(passes):
        random.shuffle(all_combos)
        for combo_idx, trio in enumerate(all_combos):
            _reload_main()
            game_num += 1
            e_names = [_label(e) for e in trio]

            seed = 7001 + game_num * 19
            env  = make("orbit_wars",
                        configuration={"episodeSteps": 500, "seed": seed},
                        debug=False)
            try:
                steps = env.run([MAIN_PY] + list(trio))
            except Exception as e:
                print(f"    game {game_num}/{total_games} error: {e}")
                continue

            final = steps[-1]
            rs    = [s.reward if s.reward is not None else 0 for s in final]
            max_r = max(rs) if rs else 0
            won   = 1 if (rs[0] == max_r and rs.count(max_r) == 1) else 0
            tag   = "WIN" if won else ("TIE" if rs[0] == max_r else "LOSS")
            print(f"    game {game_num}/{total_games}  pass {p+1}/{passes}"
                  f"  {tag}  vs [{', '.join(e_names)}]")

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
                feats = extract_features_4v4(sd, 0, step_idx, 4, fleets=rf)
                records.append({"feats": feats, "won": won, "turn": step_idx, "n_agents": 4})

    return records


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune 4v4 value function — exhaustive C(n,3) opponent coverage")
    ap.add_argument("--passes",         type=int, default=1,
                    help="Full cycles through all C(n,3) combos (default: 1)")
    ap.add_argument("--snapshot-every", type=int, default=SNAPSHOT_EVERY,
                    help="Collect a record every N steps (default: 5)")
    args = ap.parse_args()

    from train_4v4 import get_balanced_replay_records, get_val_records, train_and_export

    print("=" * 60)
    print("  train_4v4_vs_bots: exhaustive C(n,3) pretraining")
    print("=" * 60)

    print("\nLoading 4v4 replay anchor data ...")
    replay_records = get_balanced_replay_records()
    print(f"  {len(replay_records)} replay records loaded")
    val_records = get_val_records()
    print(f"  {len(val_records)} validation records loaded")

    _std = [p for p in glob.glob(str(ENEMY_DIR / "*" / "main.py"))
            if Path(p).parent.name != "my"]
    _my  = [p for p in glob.glob(str(ENEMY_DIR / "my" / "*.py"))
            if not Path(p).name.startswith("__")]
    enemy_paths = sorted(_std + _my)
    if not enemy_paths:
        print("ERROR: No enemy agents found in 'enemy agents/'.")
        sys.exit(1)

    n = len(enemy_paths)
    print(f"\nFound {n} enemy agents  (C({n},3) = {comb(n,3)} unique trios per pass):")
    for ep in enemy_paths:
        print(f"  - {_label(ep)}")

    print(f"\nCollecting bot-game data ...")
    bot_records = collect_exhaustive_records(enemy_paths, args.passes, args.snapshot_every)
    wins = sum(r["won"] for r in bot_records)
    print(f"\n  {len(bot_records)} steps collected  ({wins} winning-player steps)")

    # Replay records used as anchor — mixed in after bot-game data
    total = len(bot_records) + len(replay_records)
    print(f"\nFinal dataset: {len(bot_records)} bot-game + {len(replay_records)} replay anchor"
          f" = {total} records")

    print("\nRetraining GBC ...")
    ok = train_and_export(bot_records + replay_records, val_records=val_records)
    if ok:
        print("\nvalue_fn_4v4.py updated.")
    else:
        print("\nTraining failed — value_fn_4v4.py unchanged.")


if __name__ == "__main__":
    main()
