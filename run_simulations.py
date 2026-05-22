# ===================================================================
# run_simulations.py
#
# Runs standalone 1v1 matches between main.py and all enemy agents.
# Does NOT save any training data or snapshots.
# Saves beautiful HTML replays to the simulation/ folder.
# ===================================================================

import os, sys, glob, importlib.util, time

# ── Paths ──────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
MAIN_PY   = os.path.join(ROOT, "main.py")
ENEMY_DIR = os.path.join(ROOT, "enemy agents")
SIM_DIR   = os.path.join(ROOT, "simulation")

os.makedirs(SIM_DIR, exist_ok=True)

# ── Agent loader ──────────────────────────────────────────────────
def load_agent(path):
    name = os.path.splitext(os.path.basename(path))[0]
    d    = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod  = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent, name

# ── Game runner ───────────────────────────────────────────────────
def run_simulation(my_fn, my_name, enemy_fn, enemy_name):
    from kaggle_environments import make
    print(f"\nRunning 1v1: {my_name} vs {enemy_name}...")
    
    env = make("orbit_wars", debug=False)
    env.reset(num_agents=2)
    html_path = os.path.join(SIM_DIR, f"1v1_vs_{enemy_name}.html")
    
    # Run the full match
    states = env.run([my_fn, enemy_fn])
    
    # Determine winner
    last_state = states[-1]
    rewards = [float((s.reward if hasattr(s,'reward') else s.get('reward',0)) or 0) for s in last_state]
    max_r   = max(rewards)
    winner  = -1 if max_r <= 0 else rewards.index(max_r)
    outcome = "DRAW" if winner < 0 else "WIN" if winner == 0 else "LOSS"
    
    print(f"  Result: {outcome} ({len(states)} steps) - rendering HTML...")
    
    # Render with custom header
    try:
        html = env.render(mode="html", width=960, height=720)
        names = [my_name, enemy_name]
        cols = ["#4a90e8","#e87e4a"]
        spans = "".join(
            f"<span style='background:{cols[i]};color:#fff;padding:3px 10px;"
            f"border-radius:12px;font-size:.9em;margin:2px'>P{i}: {nm}</span>"
            for i,nm in enumerate(names))
        hdr = (f"<div style='font-family:sans-serif;background:#1a1a2e;"
               f"color:#eee;padding:10px 18px;display:flex;gap:12px;"
               f"flex-wrap:wrap;'>{spans}</div>")
        html = html.replace("<body>", f"<body>\n{hdr}") if "<body>" in html else hdr+html
        
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Saved -> {html_path}")
    except Exception as e:
        print(f"  Failed to render HTML: {e}")

# ── Main loop ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Orbit Wars -- Standalone Simulation Runner")
    print("=" * 65)
    
    if not os.path.exists(MAIN_PY):
        print(f"Error: {MAIN_PY} not found.")
        return
        
    print("Loading your agent (main.py)...")
    my_fn, my_name = load_agent(MAIN_PY)
    
    enemy_files = sorted(glob.glob(os.path.join(ENEMY_DIR, "*.py")))
    if not enemy_files:
        print(f"No enemies found in {ENEMY_DIR}.")
        return
        
    for ef in enemy_files:
        if os.path.basename(ef).startswith("__"): 
            continue
        enemy_fn, enemy_name = load_agent(ef)
        run_simulation(my_fn, "main", enemy_fn, enemy_name)
        
    print("\nAll simulations complete! Check the 'simulation' folder.")

if __name__ == "__main__":
    main()
