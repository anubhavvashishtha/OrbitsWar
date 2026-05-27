"""
eval_full.py
============
Run a thorough evaluation of main.py against every enemy in 'enemy agents/'
(except the RL one), in both 1v1 and 4v4 formats. Prints a per-enemy table
and saves raw results to eval_results.json.

Config:
  N_1V1  matches per enemy in 1v1
  N_4V4  matches per enemy in 4v4 (+ 2 random fillers picked from the rest)
  EXCLUDE  bot folder/file names to skip
"""
import os
import sys
import glob
import json
import random
import time
from pathlib import Path

from kaggle_environments import make

ROOT          = Path(__file__).parent
ENEMY_DIR     = ROOT / "enemy agents"
MY_PATH       = str(ROOT / "main.py")          # may be overridden by set_agent_mode
NN_AGENT_PATH = str(ROOT / "imitation" / "agent_nn.py")

# ── Config ──────────────────────────────────────────────────────────────────
N_4V4   = 1
N_1V1   = 1
EXCLUDE = {"kashiwaba-rl"}   # 'the RL one' — add more names here to skip
CURRENT_MODE_LABEL = "main.py"


# ── Module-clearing helper (lifted from eval_bots.py) ───────────────────────
def _reload_main():
    """Clear cached main / agent_4v4 / value_fn modules so each match loads fresh."""
    drop = [m for m in sys.modules
            if any(k in m for k in ("value_fn", "main", "agent_4v4"))]
    for m in drop:
        try:
            del sys.modules[m]
        except KeyError:
            pass


def collect_enemies():
    """Return list of (label, path) for all enemy agents not in EXCLUDE."""
    pairs = []
    for p in sorted(glob.glob(str(ENEMY_DIR / "*" / "main.py"))):
        label = Path(p).parent.name
        if label == "my" or label in EXCLUDE:
            continue
        pairs.append((label, p))
    for p in sorted(glob.glob(str(ENEMY_DIR / "my" / "*.py"))):
        if Path(p).name.startswith("__"):
            continue
        label = Path(p).stem
        if label in EXCLUDE:
            continue
        pairs.append((label, p))
    return pairs


def get_our_bot_label():
    try:
        import sys
        if "main" in sys.modules:
            m = sys.modules["main"]
        else:
            import main as m
        
        v1v1 = getattr(m, "VALUE_FN_1V1_LOADED", False)
        v4v4 = getattr(m, "VALUE_FN_4V4_LOADED", False)
        
        label = CURRENT_MODE_LABEL
        if not v1v1 or not v4v4:
            missing = []
            if not v1v1: missing.append("no-1v1")
            if not v4v4: missing.append("no-4v4")
            return f"{label}({'+'.join(missing)})"
            
        return label
    except Exception:
        return CURRENT_MODE_LABEL


def get_bot_label(path):
    p = Path(path)
    if "enemy agents" in p.parts:
        if p.parent.name == "my":
            return p.stem
        if p.name == "main.py":
            return p.parent.name
        return p.stem
    return get_our_bot_label()


def run_match(opponent_paths, seed):
    """Play one match: main.py + opponent_paths, randomise seat.
    Returns (outcome, our_reward, max_reward, error_str_or_None)."""
    _reload_main()
    agents = [MY_PATH] + opponent_paths
    seats  = list(range(len(agents)))
    random.seed(seed)
    random.shuffle(seats)
    ordered  = [agents[s] for s in seats]
    our_seat = seats.index(0)

    ordered_labels = [get_bot_label(p) for p in ordered]
    match_type = "1v1" if len(ordered) == 2 else "4v4"
    print(f"    [{match_type}] Seed {seed}: " + " vs ".join(ordered_labels), end=" ... ", flush=True)

    env = make("orbit_wars",
               configuration={"episodeSteps": 500, "seed": seed, "randomSeed": seed})
    try:
        steps = env.run(ordered)
    except Exception as e:
        print(f"ERROR: {e}")
        return "ERR", 0, 0, str(e)

    rewards = [(s.reward if s.reward is not None else 0) for s in steps[-1]]
    our_r   = rewards[our_seat]
    max_r   = max(rewards)
    if our_r >= max_r:
        outcome = "TIE" if rewards.count(max_r) > 1 else "WIN"
    else:
        outcome = "LOSS"

    rewards_str = ", ".join(f"{ordered_labels[i]}:{rewards[i]:.0f}" for i in range(len(ordered)))
    print(f"{outcome} (rewards: {rewards_str})")
    return outcome, our_r, max_r, None


def set_agent_mode(mode):
    """
    Configure sys.path + MY_PATH so the right agent is evaluated.
    Modes:
      - "classifier": main.py + value_fn from classifier/, agent_4v4 from heuristic_4v4/
      - "regressor":  main.py + value_fn from regressor/, regressor_4v4/, heuristic_4v4/
      - "nn":         imitation/agent_nn.py (EdgeTransformerNet, no main.py dependency)
    """
    global MY_PATH

    # Remove any existing agent/value-fn directories from sys.path
    drop_keys = ("classifier", "regressor", "heuristic_4v4", "regressor_4v4", "imitation")
    for path in list(sys.path):
        if any(k in path for k in drop_keys):
            try:
                sys.path.remove(path)
            except ValueError:
                pass

    if mode == "classifier":
        MY_PATH = str(ROOT / "main.py")
        sys.path.insert(0, str(ROOT / "classifier"))
        sys.path.insert(0, str(ROOT / "heuristic_4v4"))
    elif mode == "regressor":
        MY_PATH = str(ROOT / "main.py")
        sys.path.insert(0, str(ROOT / "regressor"))
        sys.path.insert(0, str(ROOT / "regressor_4v4"))
        sys.path.insert(0, str(ROOT / "heuristic_4v4"))
    elif mode == "nn":
        # agent_nn.py is a self-contained drop-in: exports agent(obs, config)
        MY_PATH = NN_AGENT_PATH
        sys.path.insert(0, str(ROOT / "imitation"))

    _reload_main()


def run_1v1_all_enemies(enemies):
    w = t = l = err = 0
    t0 = time.time()
    for idx, (label, path) in enumerate(enemies, 1):
        seed = abs(hash((label, "1v1", 0))) % (2**31)
        outcome, *_ , error = run_match([path], seed)
        if error:
            err += 1
        elif outcome == "WIN":
            w += 1
        elif outcome == "TIE":
            t += 1
        else:
            l += 1
        print(f"    Match {idx}/{len(enemies)}: W:{w} T:{t} L:{l} E:{err}")
    elapsed = time.time() - t0
    return {"W": w, "T": t, "L": l, "err": err, "N": len(enemies), "elapsed_s": round(elapsed, 1)}


def run_4v4_twenty_matches(enemies, n: int = 20):
    w = t = l = err = 0
    t0 = time.time()
    for i in range(n):
        seed = abs(hash(("global_4v4", i))) % (2**31)
        random.seed(seed)
        selected_enemies = random.sample(enemies, k=min(3, len(enemies)))
        opp_paths = [e[1] for e in selected_enemies]
        outcome, *_ , error = run_match(opp_paths, seed)
        if error:
            err += 1
        elif outcome == "WIN":
            w += 1
        elif outcome == "TIE":
            t += 1
        else:
            l += 1
        print(f"    Match {i+1}/{n}: W:{w} T:{t} L:{l} E:{err}")
    elapsed = time.time() - t0
    return {"W": w, "T": t, "L": l, "err": err, "N": n, "elapsed_s": round(elapsed, 1)}


def print_summary_table(all_results):
    print(f"\n========================================================================")
    print(f"  EVALUATION SUMMARY")
    print(f"========================================================================")
    print(f"{'Agent Configuration':<25} | {'1v1 (vs all enemies)':<22} | {'4v4 (20 matches)':<18}")
    print(f"{'':<25} | {'W-T-L-E':<12} {'WR%':>8} | {'W-T-L-E':<12} {'WR%':>8}")
    print("-" * 76)
    for mode, res in all_results.items():
        res1 = res["1v1"]
        res4 = res["4v4"]
        
        wr1 = 100.0 * res1["W"] / max(1, res1["N"])
        wr4 = 100.0 * res4["W"] / max(1, res4["N"])
        
        str1 = f"{res1['W']}-{res1['T']}-{res1['L']}-{res1['err']}"
        str4 = f"{res4['W']}-{res4['T']}-{res4['L']}-{res4['err']}"
        
        print(f"{mode:<25} | {str1:<12} {wr1:>7.1f}% | {str4:<12} {wr4:>7.1f}%")
    print("-" * 76)


def main():
    global CURRENT_MODE_LABEL

    import argparse as _ap
    parser = _ap.ArgumentParser(description="Full evaluation against all enemies.")
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "classifier", "regressor", "nn"],
        help="Which agent to evaluate: all | classifier | regressor | nn (NeuralNet/Transformer)")
    parser.add_argument(
        "--n1v1", type=int, default=N_1V1,
        help="Matches per enemy in 1v1 (default 1)")
    parser.add_argument(
        "--n4v4", type=int, default=20,
        help="Total 4v4 matches (default 20)")
    args = parser.parse_args()

    enemies = collect_enemies()
    if not enemies:
        print(f"No enemies found under {ENEMY_DIR}"); return

    print(f"Excluded: {sorted(EXCLUDE) or '(none)'}")
    print(f"Loaded {len(enemies)} enemies:")
    for label, _ in enemies:
        print(f"  - {label}")

    # ── Build mode list from --mode arg ──────────────────────────────────────
    all_modes = [
        ("Classifier", "classifier"),
        ("Regressor",  "regressor"),
        ("NeuralNet",  "nn"),
    ]
    if args.mode == "all":
        modes = all_modes
    else:
        modes = [(label, mid) for label, mid in all_modes if mid == args.mode]

    n_matches = len(modes) * (len(enemies) * args.n1v1 + args.n4v4)
    print(f"\nEvaluation Plan: {[m[0] for m in modes]}")
    print(f"  4v4: {args.n4v4} matches per agent")
    print(f"  1v1: {args.n1v1} match(es) vs each of {len(enemies)} enemies")
    print(f"  Total: ~{n_matches} matches\n")

    t_start = time.time()
    all_results = {label: {} for label, _ in modes}
    
    # ── 1. Run all 4v4 matches ──────────────────────────────────────────────
    for mode_name, mode_id in modes:
        CURRENT_MODE_LABEL = mode_name
        print("\n" + "=" * 72)
        print(f"  [4v4 Evaluation] Agent: {mode_name}")
        print("=" * 72)
        
        set_agent_mode(mode_id)
        
        print(f"\n  Running {args.n4v4} matches of 4v4...")
        res_4v4 = run_4v4_twenty_matches(enemies, n=args.n4v4)
        print(f"  4v4 Summary: W:{res_4v4['W']} T:{res_4v4['T']} L:{res_4v4['L']} E:{res_4v4['err']} "
              f"(WR: {100*res_4v4['W']/res_4v4['N']:.1f}%)")
              
        all_results[mode_name]["4v4"] = res_4v4
        
    # ── 2. Run all 1v1 matches ──────────────────────────────────────────────
    for mode_name, mode_id in modes:
        CURRENT_MODE_LABEL = mode_name
        print("\n" + "=" * 72)
        print(f"  [1v1 Evaluation] Agent: {mode_name}")
        print("=" * 72)
        
        set_agent_mode(mode_id)
        
        print(f"\n  Running 1v1 matches vs all {len(enemies)} enemies...")
        res_1v1 = run_1v1_all_enemies(enemies)
        print(f"  1v1 Summary: W:{res_1v1['W']} T:{res_1v1['T']} L:{res_1v1['L']} E:{res_1v1['err']} "
              f"(WR: {100*res_1v1['W']/res_1v1['N']:.1f}%)")
              
        all_results[mode_name]["1v1"] = res_1v1

    print_summary_table(all_results)

    total_elapsed = time.time() - t_start
    print(f"\nTotal eval time: {total_elapsed/60:.1f} min")

    out = {
        "results": all_results,
        "excluded": sorted(EXCLUDE),
        "elapsed_s": round(total_elapsed, 1),
    }
    out_path = ROOT / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved raw results -> {out_path}")


if __name__ == "__main__":
    main()
