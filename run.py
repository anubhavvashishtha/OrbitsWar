"""
run.py
======
Runs the trained RL agent (main.py) against each enemy agent
and saves one HTML simulation replay per match into the simulation/ directory.

Usage
-----
    python run.py                          # uses rl/checkpoints/best_policy.pt
    python run.py --checkpoint <path.pt>   # use a specific checkpoint
"""

import os
import sys
import glob
import argparse
import importlib.util

# ── Encode stdout/stderr as UTF-8 so emojis don't crash on Windows ──────────
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8")
    except Exception: pass
if hasattr(sys.stderr, "reconfigure"):
    try: sys.stderr.reconfigure(encoding="utf-8")
    except Exception: pass

from kaggle_environments import make

DEFAULT_CKPT = os.path.join("rl", "checkpoints", "best_policy.pt")


# ─────────────────────────────────────────────────────────────────────────────
def load_agent_file(filepath):
    """Dynamically import an agent() function from any .py file."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    dir_path    = os.path.dirname(os.path.abspath(filepath))
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)
    spec   = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = os.path.abspath(filepath)
    spec.loader.exec_module(module)
    return module.agent


def patch_rl_checkpoint(ckpt_path):
    """
    Point main.py at a specific checkpoint before importing it.
    This lets --checkpoint override the default best_policy.pt.
    """
    import main as main_rl
    main_rl._CKPT_PATH = os.path.abspath(ckpt_path)
    main_rl._MODEL     = None   # force reload
    return main_rl.agent


def run_match(env, agents, name_map, out_file):
    """
    Run one full game and save the HTML replay.

    Parameters
    ----------
    env      : freshly reset kaggle environment
    agents   : list of agent specs (file paths or callables)
    name_map : dict {player_index: display_name}
    out_file : path to write the HTML replay
    """
    runner = env._Environment__agent_runner(agents)
    step   = 0

    while not env.done:
        actions, logs = runner.act()
        env.step(actions, logs)
        step += 1
        if step % 50 == 0:
            sys.stdout.write(f"\r    Simulating step {step:>3} / 500 ...")
            sys.stdout.flush()

    sys.stdout.write(f"\r    Completed in {step} steps.          \n")
    sys.stdout.flush()

    final = env.steps[-1]

    # ── Print per-player results ───────────────────────────────────────────
    results = []
    for i, name in name_map.items():
        r = final[i].reward
        s = final[i].status
        results.append((i, name, r, s))
        print(f"    Player {i} ({name}): reward={r}  status={s}")

    # ── Determine winner ───────────────────────────────────────────────────
    p0_r = final[0].reward
    p1_r = final[1].reward
    if p0_r > p1_r:
        outcome = "WIN"
    elif p0_r < p1_r:
        outcome = "LOSS"
    else:
        outcome = "DRAW"
    print(f"    Result: {outcome}")

    # ── Render HTML ────────────────────────────────────────────────────────
    html = env.render(mode="html", width=900, height=700)

    header_text = " | ".join(
        f"P{i}: {name}" for i, name in name_map.items()
    )
    outcome_color = {"WIN": "#2ecc71", "LOSS": "#e74c3c", "DRAW": "#f39c12"}[outcome]
    header = (
        f"<div style='text-align:center;font-family:sans-serif;"
        f"padding:12px;background:{outcome_color};color:white;font-size:18px;font-weight:bold;'>"
        f"{outcome} &mdash; {header_text}</div>"
    )
    html = html.replace("<body>", f"<body>\n{header}") if "<body>" in html else header + html

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"    Saved --> {out_file}\n")

    return outcome


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Orbit Wars RL Simulation Runner")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CKPT,
        help=f"Path to trained .pt checkpoint (default: {DEFAULT_CKPT})"
    )
    args = parser.parse_args()

    # ── Validate checkpoint ────────────────────────────────────────────────
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found: {ckpt_path}")
        print("  Train first:  python -m rl.train_rl --steps 200000")
        sys.exit(1)

    # ── Discover enemy agents ──────────────────────────────────────────────
    enemy_dir   = "enemy agents"
    enemy_files = sorted(glob.glob(os.path.join(enemy_dir, "*.py")))
    enemy_files = [f for f in enemy_files if not os.path.basename(f).startswith("__")]

    if not enemy_files:
        print(f"[ERROR] No enemy agents found in '{enemy_dir}/'")
        sys.exit(1)

    sim_dir = "simulation"
    os.makedirs(sim_dir, exist_ok=True)

    # ── Load RL agent ──────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  Orbit Wars — RL Agent vs Enemies")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Enemies    : {len(enemy_files)}")
    print("=" * 65 + "\n")

    rl_agent = patch_rl_checkpoint(ckpt_path)

    wins = losses = draws = 0

    # ── Run one match per enemy ────────────────────────────────────────────
    for idx, enemy_file in enumerate(enemy_files, 1):
        enemy_name = os.path.splitext(os.path.basename(enemy_file))[0]

        print(f"[{idx}/{len(enemy_files)}] RL agent  vs  {enemy_name}")

        try:
            enemy_fn = load_agent_file(enemy_file)

            env = make("orbit_wars", debug=True)
            env.reset(2)

            name_map = {0: "RL_agent", 1: enemy_name}
            out_file = os.path.join(sim_dir, f"rl_vs_{enemy_name}.html")

            outcome = run_match(env, [rl_agent, enemy_fn], name_map, out_file)

            if outcome == "WIN":    wins   += 1
            elif outcome == "LOSS": losses += 1
            else:                   draws  += 1

        except Exception as exc:
            print(f"    [ERROR] {exc}")
            import traceback; traceback.print_exc()

    # ── Final scoreboard ───────────────────────────────────────────────────
    total = wins + losses + draws
    print("=" * 65)
    print(f"  Final record:  {wins}W / {losses}L / {draws}D  out of {total} matches")
    print(f"  Win rate    :  {wins/max(total,1):.0%}")
    print(f"  Replays saved in '{sim_dir}/'")
    print("=" * 65)


if __name__ == "__main__":
    main()
