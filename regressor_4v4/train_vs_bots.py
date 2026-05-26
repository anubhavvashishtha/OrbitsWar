"""
regressor_4v4/train_vs_bots.py
================================
Fine-tune the 4v4 TD regressor using exhaustive C(N,3) opponent coverage.

Pipeline
--------
  1. Load replay trajectories (seed) via get_td_trajectories()
  2. Discover all N enemy agents in enemy agents/*/main.py
  3. Generate every unique C(N,3) opponent trio — shuffled for diverse coverage
  4. Play one game per trio (our main.py as player 0); collect full step trajectories
  5. Combine replay trajectories + bot trajectories → train_and_export() → value_fn_4v4.py

Usage
-----
    python regressor_4v4/train_vs_bots.py
    python regressor_4v4/train_vs_bots.py --passes 2 --gamma 0.99
"""
import os, sys, glob, random, importlib.util
from itertools import combinations
from math import comb
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent

MAIN_PY   = str(PROJECT_ROOT / "main.py")
ENEMY_DIR = PROJECT_ROOT / "enemy agents"

# ── Load 4v4 feature extraction (snap / extract_features_4v4) ───────────────
_h4_dir = str(PROJECT_ROOT / "heuristic_4v4")
if _h4_dir not in sys.path:
    sys.path.insert(0, _h4_dir)
from train_4v4 import extract_features_4v4, snap

# ── Load regressor_4v4's train_from_replays via explicit path ────────────────
def _load_reg_train():
    spec = importlib.util.spec_from_file_location(
        "_reg4_train_from_replays", str(ROOT / "train_from_replays.py"))
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


def collect_exhaustive_trajectories(enemy_pool, passes):
    """
    Generate all C(N,3) combos, shuffle, play one game per combo.
    Repeat for `passes` full cycles.
    Returns list of {'traj': [feats...], 'won': float, 'n_agents': 4} dicts.
    """
    from kaggle_environments import make

    all_combos  = list(combinations(enemy_pool, 3))
    n_combos    = len(all_combos)
    total_games = n_combos * passes

    print(f"  {len(enemy_pool)} enemies  ->  C({len(enemy_pool)},3) = {n_combos} unique trios"
          f"  x  {passes} pass(es)  =  {total_games} games")

    trajectories = []
    game_num     = 0

    for p in range(passes):
        random.shuffle(all_combos)
        for trio in all_combos:
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
            won   = 1.0 if (rs[0] == max_r and rs.count(max_r) == 1) else 0.0
            tag   = "WIN" if won == 1.0 else ("TIE" if rs[0] == max_r else "LOSS")
            print(f"    game {game_num}/{total_games}  pass {p+1}/{passes}"
                  f"  {tag}  vs [{', '.join(e_names)}]")

            traj = []
            for step_idx, step_states in enumerate(steps):
                if _read(step_states[0], "done", False):
                    break  # player 0 eliminated — stop collecting stale observations
                obs = _read(step_states[0], "observation")
                if obs is None:
                    continue
                rp = _read(obs, "planets", []) or []
                ci = set(_read(obs, "comet_planet_ids", []) or [])
                rf = _read(obs, "fleets", []) or []
                sd = snap(rp, ci)
                if not sd:
                    continue
                traj.append(extract_features_4v4(sd, 0, step_idx, 4, fleets=rf))

            if len(traj) >= 2:
                trajectories.append({"traj": traj, "won": won, "n_agents": 4})

    return trajectories


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Fine-tune 4v4 TD regressor — exhaustive C(n,3) opponent coverage")
    ap.add_argument("--passes",      type=int,   default=1,
                    help="Full cycles through all C(n,3) combos (default: 1)")
    ap.add_argument("--gamma",       type=float, default=0.99)
    ap.add_argument("--reward-scale",type=float, default=0.1)
    args = ap.parse_args()

    import joblib

    print("=" * 60)
    print("  regressor_4v4/train_vs_bots: warm-start fine-tune via C(n,3) bot coverage")
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

    n = len(enemy_paths)
    print(f"\nFound {n} enemies  (C({n},3) = {comb(n,3)} unique trios per pass):")
    for ep in enemy_paths:
        print(f"  - {_label(ep)}")

    print(f"\nCollecting bot-game trajectories ...")
    bot_trajs = collect_exhaustive_trajectories(enemy_paths, args.passes)
    wins = sum(t["won"] for t in bot_trajs)
    print(f"\n  {len(bot_trajs)} trajectories collected  ({wins:.0f} wins)")

    print(f"\nBot-game dataset: {len(bot_trajs)} trajectories")

    print("\nFine-tuning TD regressor (warm-start on bot data) ...")
    ok = train_and_export(
        bot_trajs, val_trajectories=val_trajs,
        gamma=args.gamma, reward_scale=args.reward_scale,
        base_model=base_model,
    )
    print("\nvalue_fn_4v4.py updated." if ok else "\nTraining failed.")


if __name__ == "__main__":
    main()
