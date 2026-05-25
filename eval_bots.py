import os
import sys
import glob
import random
from kaggle_environments import make

ROOT            = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_DIR  = os.path.join(ROOT, "classifier")
HEURISTIC_DIR   = os.path.join(ROOT, "heuristic_4v4")
ENEMY_DIR       = os.path.join(ROOT, "enemy agents")

# classifier/ is always needed (value_fn_1v1)
if CLASSIFIER_DIR not in sys.path:
    sys.path.insert(0, CLASSIFIER_DIR)


def _set_mode(mode):
    """
    "classifier" → only value_fn_1v1 visible; agent_4v4 + value_fn_4v4 hidden.
    "hybrid"     → value_fn_1v1 + value_fn_4v4 + agent_4v4 all visible.
    """
    if mode == "hybrid":
        if HEURISTIC_DIR not in sys.path:
            sys.path.insert(0, HEURISTIC_DIR)
    else:
        if HEURISTIC_DIR in sys.path:
            sys.path.remove(HEURISTIC_DIR)


def _reload():
    """Clear cached main / value_fn / agent modules so the next run loads fresh."""
    drop = [m for m in sys.modules
            if any(k in m for k in ("value_fn", "main", "agent_4v4"))]
    for m in drop:
        del sys.modules[m]


def _run_matches(mode, game_type, num_matches, enemy_pool):
    """
    Core match loop.
    game_type: "4v4" (3 enemies) | "1v1" (1 enemy)
    Returns dict with wins / ties / losses / total.
    """
    n_enemies   = 3 if game_type == "4v4" else 1
    wins = ties = losses = 0

    for i in range(num_matches):
        _set_mode(mode)
        _reload()

        opponents   = [random.choice(enemy_pool) for _ in range(n_enemies)]
        all_agents  = ["main.py"] + opponents

        # Randomise seats so we are not always player-0
        seats = list(range(len(all_agents)))
        random.shuffle(seats)
        shuffled  = [all_agents[s] for s in seats]
        our_seat  = seats.index(0)

        names = [os.path.basename(o).replace(".py", "") for o in opponents]
        print(f"  {i+1:2}/{num_matches} seat={our_seat} vs {names} ...",
              end=" ", flush=True)

        env = make("orbit_wars", configuration={"episodeSteps": 500})
        try:
            steps   = env.run(shuffled)
            rewards = [s.reward if s.reward is not None else 0 for s in steps[-1]]
            our_r   = rewards[our_seat]
            max_r   = max(rewards)

            if our_r == max_r:
                if rewards.count(max_r) > 1:
                    print(f"TIE  (r={our_r:.0f})")
                    ties += 1
                else:
                    print(f"WIN  (r={our_r:.0f})")
                    wins += 1
            else:
                print(f"LOSS (ours={our_r:.0f} best={max_r:.0f})")
                losses += 1
        except Exception as e:
            print(f"ERROR: {e}")
            losses += 1

    return {"wins": wins, "ties": ties, "losses": losses, "total": num_matches}


def run_eval(label, mode, game_type, num_matches=20):
    """Run one evaluation block and print per-match results."""
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"  Mode: {mode.upper()}   Game: {game_type}")
    print(f"{'='*58}")

    _std = [p for p in glob.glob(os.path.join(ENEMY_DIR, "*", "main.py"))
            if os.path.basename(os.path.dirname(p)) != "my"]
    _my  = [p for p in glob.glob(os.path.join(ENEMY_DIR, "my", "*.py"))
            if not os.path.basename(p).startswith("__")]
    enemy_pool = sorted(_std + _my)
    if not enemy_pool:
        print("  ERROR: no enemies found in 'enemy agents/'")
        return None

    res = _run_matches(mode, game_type, num_matches, enemy_pool)
    t   = res["total"]
    print(f"\n  Wins:    {res['wins']}/{t}  ({res['wins']/t*100:.1f}%)")
    print(f"  Ties:    {res['ties']}/{t}  ({res['ties']/t*100:.1f}%)")
    print(f"  Losses:  {res['losses']}/{t}  ({res['losses']/t*100:.1f}%)")
    print(f"  Win+Tie: {(res['wins']+res['ties'])/t*100:.1f}%")
    return res


def comparison_table(results):
    """Print a side-by-side comparison of all evaluation runs."""
    print(f"\n{'='*58}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*58}")
    print(f"  {'Label':<35} {'Win%':>6}  {'Win+Tie%':>9}")
    print(f"  {'-'*53}")
    for label, res in results:
        if res is None:
            continue
        t   = res["total"]
        wp  = res["wins"] / t * 100
        wtp = (res["wins"] + res["ties"]) / t * 100
        print(f"  {label:<35} {wp:>5.1f}%  {wtp:>8.1f}%")
    print()


if __name__ == "__main__":
    N = 20   # matches per evaluation block

    # ── 4v4 battles ───────────────────────────────────────────────
    # Both modes face 3 random enemies in a 4-player FFA.
    # classifier : plan_moves() + value_fn_1v1 only (no agent_4v4)
    # hybrid     : agent_4v4() + value_fn_4v4 (the real 4v4 brain)
    r_clf_4v4 = run_eval("Classifier only  (4v4)", "classifier", "4v4", N)
    r_hyb_4v4 = run_eval("Hybrid 4v4 agent (4v4)", "hybrid",     "4v4", N)

    # ── 1v1 battles ───────────────────────────────────────────────
    # In 1v1, agent_4v4 is never called regardless of mode —
    # both use plan_moves() + value_fn_1v1.
    # This confirms the 4v4 additions have no 1v1 regression.
    r_clf_1v1 = run_eval("Classifier only  (1v1)", "classifier", "1v1", N)
    r_hyb_1v1 = run_eval("Hybrid 4v4 agent (1v1)", "hybrid",     "1v1", N)

    # ── Summary ───────────────────────────────────────────────────
    comparison_table([
        ("Classifier only  — 4v4", r_clf_4v4),
        ("Hybrid 4v4 agent — 4v4", r_hyb_4v4),
        ("Classifier only  — 1v1", r_clf_1v1),
        ("Hybrid 4v4 agent — 1v1", r_hyb_1v1),
    ])

    print("Evaluation complete!")
