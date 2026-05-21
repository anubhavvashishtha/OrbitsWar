# ===================================================================
# run_simulations.py
#
# Runs ALL match combinations and saves results + HTML replays.
#
# Match types run:
#   1v1  — main.py vs each enemy agent (3 matches)
#   1v1  — main.py vs random enemy (1 match, enemy chosen randomly)
#   FFA  — main.py vs all 3 enemies simultaneously (1v1v1v1)
#
# Output:
#   simulation/1v1_vs_ENEMY.html          replay for each 1v1
#   simulation/1v1_vs_random_ENEMY.html   replay for random-pick 1v1
#   simulation/ffa_all_agents.html        replay for 4-player FFA
#   simulation/results.json               all scores and outcomes
#
# Usage:
#   python run_simulations.py
# ===================================================================

import os, sys, glob, json, random, importlib.util, time
from collections import namedtuple

if hasattr(sys.stdout,'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ── Paths ─────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
MAIN_PY   = os.path.join(ROOT, "main.py")
ENEMY_DIR = os.path.join(ROOT, "enemy agents")
SIM_DIR   = os.path.join(ROOT, "simulation")
os.makedirs(SIM_DIR, exist_ok=True)

SEED = 42   # change for different board layouts

# ── Agent loader ──────────────────────────────────────────────────
def load_agent(path, force=True):
    name=os.path.splitext(os.path.basename(path))[0]
    d=os.path.dirname(os.path.abspath(path))
    if d not in sys.path: sys.path.insert(0,d)
    if force:
        for k in [name,"value_fn","main"]:
            if k in sys.modules: del sys.modules[k]
    spec=importlib.util.spec_from_file_location(name,os.path.abspath(path))
    mod=importlib.util.module_from_spec(spec)
    mod.__file__=os.path.abspath(path)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod.agent, name

# ── HTML builder ──────────────────────────────────────────────────
def _render_html(env, player_labels):
    """Render env to HTML and inject a header showing player colours."""
    html=env.render(mode="html",width=960,height=720)
    colours=["#4a90e8","#e87e4a","#4ae87e","#e84a90"]
    parts=[]
    parts.append("<div style='font-family:sans-serif;background:#1a1a2e;"
                 "color:#eee;padding:12px 20px;display:flex;gap:24px;"
                 "align-items:center;flex-wrap:wrap;'>")
    parts.append("<span style='font-size:1.1em;font-weight:bold;margin-right:8px;'>"
                 "Orbit Wars</span>")
    for i,(label,col) in enumerate(zip(player_labels,colours)):
        parts.append(f"<span style='background:{col};color:#fff;padding:3px 10px;"
                     f"border-radius:12px;font-size:0.9em;'>P{i}: {label}</span>")
    parts.append("</div>")
    header="".join(parts)
    if "<body>" in html: html=html.replace("<body>",f"<body>\n{header}")
    elif "<body" in html:
        idx=html.index("<body")
        end=html.index(">",idx)
        html=html[:end+1]+"\n"+header+html[end+1:]
    else: html=header+html
    return html

# ── Single game runner ────────────────────────────────────────────
def run_match(agent_fns, agent_names, seed=SEED, html_path=None):
    """
    agent_fns   : list of callables [p0_fn, p1_fn, ...]  (2 or 4 entries)
    agent_names : list of str labels
    Returns dict with all result info.
    """
    from kaggle_environments import make
    n=len(agent_fns)
    assert n in (2,4), "Only 2 or 4 agents supported"

    print(f"  Match: {' vs '.join(agent_names)}")
    env=make("orbit_wars",configuration={"seed":seed,"randomSeed":seed},debug=False)
    env.reset(num_agents=n)

    t0=time.time()
    states=env.step([[]]*n)
    step=0

    while True:
        statuses=[str(s.status if hasattr(s,'status') else s.get('status','ACTIVE'))
                  for s in states]
        if statuses[0]!='ACTIVE': break
        obss=[(s.observation if hasattr(s,'observation') else s.get('observation'))
              for s in states]
        actions=[]
        for i,obs in enumerate(obss):
            try: a=agent_fns[i](obs)
            except Exception as ex:
                print(f"\n    WARN: agent {agent_names[i]} threw: {ex}")
                a=[]
            actions.append(a or [])
        states=env.step(actions); step+=1
        if step%50==0:
            sys.stdout.write(f"\r    Step {step:3d}/500"); sys.stdout.flush()

    elapsed=time.time()-t0
    sys.stdout.write(f"\r    Done in {step} steps ({elapsed:.1f}s)       \n")
    sys.stdout.flush()

    rewards=[float((s.reward if hasattr(s,'reward') else s.get('reward',0)) or 0)
             for s in states]
    max_r=max(rewards)
    if max_r<=0:
        outcome="DRAW"
        winner_name="(draw)"
    else:
        wi=rewards.index(max_r)
        winner_name=agent_names[wi]
        outcome="WIN" if wi==0 else "LOSS" if n==2 else f"P{wi}_WINS"

    print(f"    Result: {outcome}  |  Winner: {winner_name}")
    for i,(name,r) in enumerate(zip(agent_names,rewards)):
        print(f"    P{i} {name:<22} reward={r:+.1f}")

    if html_path:
        html=_render_html(env,agent_names)
        with open(html_path,"w",encoding="utf-8") as f: f.write(html)
        print(f"    Saved -> {html_path}")

    return {
        "match_type": f"{n}p",
        "agents": agent_names,
        "seed": seed,
        "rewards": rewards,
        "outcome": outcome,
        "winner": winner_name,
        "steps": step,
        "elapsed_s": round(elapsed,1),
        "html": html_path,
    }

# ── Main ──────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("  Orbit Wars Simulation Runner")
    print("="*65)

    # ── Find enemy agents ──────────────────────────────────────────
    enemy_files=sorted([f for f in glob.glob(os.path.join(ENEMY_DIR,"*.py"))
                        if not os.path.basename(f).startswith("__")])
    if not enemy_files:
        print(f"ERROR: No .py files in '{ENEMY_DIR}'"); sys.exit(1)

    print(f"\nFound {len(enemy_files)} enemy agent(s):")
    for f in enemy_files: print(f"  - {os.path.basename(f)}")

    # ── Load main agent (fresh, picks up latest value_fn.py) ───────
    print("\nLoading agents...")
    my_fn, my_name = load_agent(MAIN_PY, force=True)
    vf_path=os.path.join(ROOT,"value_fn.py")
    vf_status="WITH value_fn.py" if os.path.exists(vf_path) else "NO value_fn.py (pure heuristic)"
    print(f"  main.py loaded ({vf_status})")

    enemies=[]
    for ef in enemy_files:
        try:
            fn,name=load_agent(ef,force=False)
            enemies.append((name,fn))
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  WARN: {ef}: {e}")

    if not enemies:
        print("No enemy agents loaded."); sys.exit(1)
    if len(enemies)<3:
        print(f"WARNING: expected 3 enemy agents, found {len(enemies)}.")

    all_results=[]

    # ══════════════════════════════════════════════════════════════
    # MATCH TYPE 1: main.py vs EACH enemy  (1v1, separate matches)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  BLOCK 1 — 1v1: main.py vs each enemy individually")
    print(f"{'─'*65}")

    for i,(ename,efn) in enumerate(enemies):
        seed=SEED+i
        html_path=os.path.join(SIM_DIR,f"1v1_vs_{ename}.html")
        result=run_match(
            agent_fns=[my_fn, efn],
            agent_names=[my_name, ename],
            seed=seed,
            html_path=html_path,
        )
        result["match_label"]=f"1v1_vs_{ename}"
        all_results.append(result)

    # ══════════════════════════════════════════════════════════════
    # MATCH TYPE 2: main.py vs ONE randomly chosen enemy  (1v1)
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  BLOCK 2 — 1v1: main.py vs randomly chosen enemy")
    print(f"{'─'*65}")

    random.seed(SEED)
    rand_name, rand_fn = random.choice(enemies)
    print(f"  Randomly selected: {rand_name}")
    html_path=None
    result=run_match(
        agent_fns=[my_fn, rand_fn],
        agent_names=[my_name, f"{rand_name}(random)"],
        seed=SEED+99,
        html_path=html_path,
    )
    result["match_label"]="1v1_random_enemy"
    result["randomly_chosen_enemy"]=rand_name
    all_results.append(result)

    # ══════════════════════════════════════════════════════════════
    # MATCH TYPE 3: FFA — main.py vs ALL 3 enemies  (1v1v1v1)
    # Uses exactly 4 agents. If fewer than 3 enemies, pads with
    # the first enemy repeated.
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'─'*65}")
    print("  BLOCK 3 — FFA 4-player: main.py vs all 3 enemies")
    print(f"{'─'*65}")

    ffa_agents=[(my_name,my_fn)]+list(enemies)
    # Pad to exactly 4 if fewer enemies
    while len(ffa_agents)<4:
        ffa_agents.append((f"{enemies[0][0]}_copy",enemies[0][1]))
    ffa_agents=ffa_agents[:4]  # cap at 4

    html_path=None
    result=run_match(
        agent_fns=[fn for _,fn in ffa_agents],
        agent_names=[nm for nm,_ in ffa_agents],
        seed=SEED+200,
        html_path=html_path,
    )
    result["match_label"]="ffa_4player"
    all_results.append(result)

    # ══════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ══════════════════════════════════════════════════════════════
    results_path=os.path.join(SIM_DIR,"results.json")
    with open(results_path,"w",encoding="utf-8") as f:
        json.dump(all_results,f,indent=2)
    print(f"\nAll results saved -> {results_path}")

    # ── Summary table ──────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Match':<32} {'Outcome':<10} {'Winner':<22} {'Steps'}")
    print(f"  {'-'*65}")
    for r in all_results:
        label=r.get("match_label","?")
        outcome=r.get("outcome","?")
        winner=r.get("winner","?")[:20]
        steps=r.get("steps",0)
        # colour-code outcome in terminal
        tag="[WIN] " if outcome=="WIN" else "[LOSS]" if outcome=="LOSS" else "[DRAW]" if outcome=="DRAW" else "[FFA] "
        print(f"  {tag} {label:<28} {outcome:<10} {winner:<22} {steps}")

    wins=sum(1 for r in all_results if r.get("outcome")=="WIN")
    losses=sum(1 for r in all_results if r.get("outcome")=="LOSS")
    draws=sum(1 for r in all_results if r.get("outcome")=="DRAW")
    print(f"\n  1v1 record: {wins}W / {losses}L / {draws}D "
          f"(excluding FFA)")
    print(f"\n  HTML replays saved to: {SIM_DIR}/")
    print(f"{'='*65}")

if __name__=="__main__":
    main()
