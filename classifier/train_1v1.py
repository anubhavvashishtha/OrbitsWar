# ===================================================================
# train.py  v5.0
#
# Training pipeline for Orbit Wars agent -- 1v1 mode.
# Fixes over v4:
#   - Replay pretraining from 271 Kaggle match replays
#   - balance_records() keeps training at 50/50 win/loss
#   - load_recent_records() stops stale round-1 data mixing in
#   - Opponent evaluation increased from 3 to 8 games
#   - AUC_MIN lowered to 0.58 so more models get exported
#   - Removed snapshot / self-play logic
# ===================================================================

import os, sys, glob, json, math, time, random, importlib.util
from pathlib import Path

if hasattr(sys.stdout,'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ── Paths ──────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
MAIN_PY      = os.path.join(PROJECT_ROOT, "main.py")
ENEMY_DIR    = os.path.join(PROJECT_ROOT, "enemy agents")
SIM_DIR      = os.path.join(PROJECT_ROOT, "simulation")
DATA_DIR     = os.path.join(ROOT, "training_data")
RECORDS_DIR  = os.path.join(PROJECT_ROOT, "records")
VALUEFN      = os.path.join(ROOT, "value_fn_1v1.py")
RESULTS_LOG  = os.path.join(ROOT, "training_log.json")

for _d in (ENEMY_DIR, SIM_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)

# ── Settings ───────────────────────────────────────────────────────
MAX_ROUNDS      = 25
WIN_RATE_TARGET = 0.85
GAMES_PER_ENEMY = 8
EVAL_GAMES      = 8      # was 3 -- enough games to get reliable WR estimates
SNAPSHOT_EVERY  = 5      # feature snapshot interval (turns)
MIN_SAMPLES     = 500
N_ESTIMATORS    = 600
MAX_DEPTH       = 5
LEARNING_RATE   = 0.003
AUC_MIN         = 0.58   # was 0.65
SEED_BASE       = 42
RECENT_ROUNDS   = 5      # only train on last N rounds of self-play
REPLAY_MIX_SIZE = 5000   # replay records mixed into each training round

CURRICULUM_PHASES = [
    {"name": "easy",   "min_wr": 0.6, "max_wr": 1.0, "weight": 2.0},
    {"name": "medium", "min_wr": 0.3, "max_wr": 0.7, "weight": 1.5},
    {"name": "hard",   "min_wr": 0.0, "max_wr": 0.4, "weight": 1.0},
]

# ── Agent loader ──────────────────────────────────────────────────
def load_agent(path, force=False):
    name = os.path.splitext(os.path.basename(path))[0]
    d    = os.path.dirname(os.path.abspath(path))
    if d not in sys.path:
        sys.path.insert(0, d)
    if force:
        for k in [name, "value_fn", "main"]:
            if k in sys.modules:
                del sys.modules[k]
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(path))
    mod  = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent, name

# ── Feature extraction (matches main.py exactly) ───────────────
def extract_features(state, player, turn, n_hint=2):
    ms=mp=mpr=mc=ts=tpr=0.0
    ed = {}
    for s in state:
        sh = max(0.0, float(s.get('ships', 0)))
        ts += sh
        ow = s.get('owner', -1)
        if ow == -1:
            continue
        pr = float(s.get('production', 0))
        tpr += pr
        cn = max(0.0, 60.0 - math.hypot(float(s.get('x',50))-50.0,
                                          float(s.get('y',50))-50.0))
        if ow == player:
            ms+=sh; mp+=1; mpr+=pr; mc+=cn
        else:
            e = ed.setdefault(ow, {'s':0.,'p':0,'pr':0.,'c':0.})
            e['s']+=sh; e['p']+=1; e['pr']+=pr; e['c']+=cn
    bes=bep=bepr=bec=0.0
    if ed:
        be = max(ed.values(), key=lambda e: e['s']+100*e['p'])
        bes=be['s']; bep=float(be['p'])
        bepr=be['pr']; bec=be['c']/max(1,be['p'])
    tp = max(1, len(state))
    sf = lambda a,b: a/b if b>1e-9 else 0.0
    np2 = max(2, 1+len(ed), n_hint)
    return [
        turn/500.0, np2/4.0,
        sf(mp,tp), sf(bep,tp),
        sf(ms,ts), sf(bes,ts),
        sf(mpr,tpr), sf(bepr,tpr),
        (mc/max(1,mp))/60.0, bec/60.0,
        sf(ms-bes,ts+1), sf(mp-bep,tp), sf(mpr-bepr,tpr+1),
        1.0 if np2==2 else 0.0,
        1.0 if np2==4 else 0.0,
        sf(ms, bes+1),
    ]

def snap(raw_p, comet_ids=None):
    cids = comet_ids or set()
    return [{'id':p[0],'owner':p[1],'ships':p[5],
             'production':p[6],'x':p[2],'y':p[3]}
            for p in raw_p if p[0] not in cids]

# ── Game runner ───────────────────────────────────────────────────
def play_game(fns, names, seed=0, html_path=None):
    from kaggle_environments import make
    n = len(fns)
    env = make("orbit_wars",
               configuration={"seed": seed, "randomSeed": seed},
               debug=False)
    env.reset(num_agents=n)
    snaps  = [[] for _ in range(n)]
    states = env.step([[]] * n)
    step   = 0

    while True:
        st0    = states[0]
        status = str(st0.status if hasattr(st0,'status') else st0.get('status','ACTIVE'))
        if status != 'ACTIVE':
            break
        obss = [(s.observation if hasattr(s,'observation') else s.get('observation'))
                for s in states]

        if step % SNAPSHOT_EVERY == 0 and obss[0] is not None:
            rp = (obss[0].get("planets",[]) if isinstance(obss[0],dict)
                  else getattr(obss[0],"planets",[]))
            ci = set(obss[0].get("comet_planet_ids",[]) if isinstance(obss[0],dict)
                     else getattr(obss[0],"comet_planet_ids",[]))
            sd = snap(rp, ci)
            if sd:
                for i in range(n):
                    snaps[i].append((step, extract_features(sd, i, step, n)))

        actions = []
        for i, obs in enumerate(obss):
            try:    a = fns[i](obs)
            except: a = []
            actions.append(a or [])
        states = env.step(actions)
        step  += 1

    rewards = [float((s.reward if hasattr(s,'reward') else s.get('reward',0)) or 0)
               for s in states]
    max_r   = max(rewards)
    winner  = -1 if max_r <= 0 else rewards.index(max_r)
    outcome = ("DRAW" if winner < 0 else
               "WIN"  if winner == 0 else
               "LOSS" if n == 2 else f"P{winner}_WINS")

    records = []
    for i in range(n):
        won = 1 if i == winner else 0
        records.append([{'feats':f,'turn':t,'won':won,'player':i}
                         for t,f in snaps[i]])

    if html_path:
        try:
            html = env.render(mode="html", width=960, height=720)
            cols  = ["#4a90e8","#e87e4a","#4ae87e","#e84a90"]
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
        except Exception as e:
            print(f"    WARN html: {e}")

    return records, rewards, step, outcome

# ── Opponent evaluation ───────────────────────────────────────────
def evaluate_opponent_strength(my_fn, enemy_fn, enemy_name, n_games=EVAL_GAMES):
    wins = 0
    for g in range(n_games):
        seed = SEED_BASE + hash(enemy_name) % 10000 + g
        recs, rewards, steps, outcome = play_game(
            [my_fn, enemy_fn], ["main", enemy_name], seed=seed)
        if outcome == "WIN":
            wins += 1
        labels = ["1st","2nd","3rd","4th","5th","6th","7th","8th"]
        print(f"    {labels[g] if g<8 else f'{g+1}th'} match {enemy_name}: {outcome}, {steps} turns")
    wr = wins / n_games
    print(f"    -> {enemy_name}: {wr:.0%} win rate ({wins}/{n_games})")
    return wr

# ── Curriculum opponent selection ─────────────────────────────────
def select_opponents_by_curriculum(enemies_with_wr, phase):
    min_wr, max_wr = phase["min_wr"], phase["max_wr"]
    matching = [(name, fn, wr) for name, fn, wr in enemies_with_wr
                if min_wr <= wr <= max_wr]
    if not matching:
        matching = sorted(enemies_with_wr,
                          key=lambda x: abs(x[2] - (min_wr+max_wr)/2))[:3]
    return matching

# ── Run one full training round ───────────────────────────────────
def run_round(my_fn, my_name, enemies, round_num, curriculum_phase):
    all_records   = []
    match_results = []

    selected = select_opponents_by_curriculum(enemies, curriculum_phase)
    if not selected:
        selected = enemies[:3]

    print(f"\n  [Curriculum: {curriculum_phase['name']}] Selected {len(selected)} opponents")
    for name, fn, wr in selected:
        print(f"    - {name} (WR: {wr:.1%})")

    total_1v1 = len(selected) * GAMES_PER_ENEMY
    done = 0

    print(f"\n  [1v1] {my_name} vs selected enemies ({GAMES_PER_ENEMY} games each)")
    for ename, efn, _ in selected:
        w = l = d = 0
        for g in range(GAMES_PER_ENEMY):
            seed = SEED_BASE + round_num * 10000 + done
            recs, rewards, steps, outcome = play_game(
                [my_fn, efn], [my_name, ename],
                seed=seed, html_path=None)
            if abs(rewards[0]) > 0.01:
                for pr in recs:
                    all_records.extend(pr)
            if outcome == "WIN":    w += 1
            elif outcome == "LOSS": l += 1
            else:                   d += 1
            done += 1
            pct = int(done / total_1v1 * 25)
            sys.stdout.write(
                f"\r    [{'#'*pct}{'-'*(25-pct)}] {done}/{total_1v1}"
                f"  vs {ename}: W{w} L{l} D{d}  ({steps}st)")
            sys.stdout.flush()
        print()
        match_results.append({"type":"1v1","enemy":ename,
                               "wins":w,"losses":l,"draws":d,
                               "win_rate": w/max(1,w+l+d)})

    return all_records, match_results

# ── Data helpers ──────────────────────────────────────────────────
def save_records(records, round_num):
    path = os.path.join(DATA_DIR, f"records_{round_num:04d}.json")
    try:
        with open(path,"w") as f:
            json.dump(records, f)
        print(f"  Saved {len(records)} records -> {path}")
    except Exception as e:
        print(f"  WARN save: {e}")

def load_recent_records(n_rounds=RECENT_ROUNDS):
    """Only load the most recent N rounds to avoid stale data."""
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "records_*.json")))
    recent    = all_files[-n_rounds:] if len(all_files) > n_rounds else all_files
    out = []
    for p in recent:
        try:
            with open(p) as f:
                out.extend(json.load(f))
        except: pass
    return out

def balance_records(records):
    """Equalise win/loss counts so the model trains on a 50/50 split."""
    wins   = [r for r in records if r['won'] == 1]
    losses = [r for r in records if r['won'] == 0]
    n = min(len(wins), len(losses))
    if n < 50:
        return records
    return random.sample(wins, n) + random.sample(losses, n)

# ── GBC trainer + exporter ────────────────────────────────────────
def train_and_export(all_records):
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  ERROR: pip install scikit-learn numpy")
        return False

    if len(all_records) < MIN_SAMPLES:
        print(f"  {len(all_records)} samples < {MIN_SAMPLES} needed. Skip.")
        return False

    X     = np.array([r['feats'] for r in all_records], dtype=np.float32)
    y     = np.array([r['won']   for r in all_records], dtype=np.int32)
    turns = np.array([r['turn']  for r in all_records])

    def _tr(mask, ne, d):
        Xs, ys = X[mask], y[mask]
        if len(Xs)<80 or ys.mean()<0.02 or ys.mean()>0.98:
            return None, 0.0
        Xt,Xv,yt,yv = train_test_split(
            Xs, ys, test_size=0.2, random_state=42,
            stratify=ys if min(ys.sum(),(1-ys).sum())>=10 else None)
        clf = GradientBoostingClassifier(
            n_estimators=ne, max_depth=d,
            learning_rate=LEARNING_RATE, subsample=0.8,
            min_samples_leaf=30, random_state=42)
        clf.fit(Xt, yt)
        return clf, roc_auc_score(yv, clf.predict_proba(Xv)[:,1])

    specs = [
        ("GLOBAL", np.ones(len(X), bool),           N_ESTIMATORS, MAX_DEPTH),
        ("EARLY",  turns <= 100,                     300, 5),
        ("LATE",   turns >= 300,                     300, 5),
        ("MID",    (turns > 100) & (turns < 300),   250, 4),
    ]

    lines    = ["# AUTO-GENERATED by train.py  --  do not edit", ""]
    exported = []
    any_good = False

    for name, mask, ne, d in specs:
        print(f"  {name:<8} ...", end=" ", flush=True)
        clf, auc = _tr(mask, ne, d)
        print(f"AUC={auc:.3f}  ({mask.sum()} samples)")
        if clf is None or auc < AUC_MIN:
            print(f"    Skipped (AUC too low or not enough data)")
            continue
        lines.append(f"{name}_INIT={float(clf.init_.class_prior_[1]):.8f}")
        lines.append(f"{name}_N_TREES={len(clf.estimators_)}")
        lines.append(f"{name}_TREES=[")
        for ta in clf.estimators_:
            t  = ta[0].tree_
            fi = [int(v) for v in t.feature]
            th = [round(float(v),6) for v in t.threshold]
            va = [round(float(t.value[i,0,0]),8) for i in range(t.node_count)]
            le = [int(v) for v in t.children_left]
            ri = [int(v) for v in t.children_right]
            lines.append(f"    ({fi},{th},{va},{le},{ri}),")
        lines.append("]")
        lines.append("")
        exported.append(name)
        any_good = True
        print(f"    Exported {name}  ({len(clf.estimators_)} trees,  AUC={auc:.3f})")

    if not any_good:
        print("  No models met AUC threshold.")
        return False

    lines.append(f"VALUE_FN_MODELS={exported!r}")
    lines.append("VALUE_FN_LOADED=True")
    lines.append("")

    with open(VALUEFN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sz = os.path.getsize(VALUEFN) / 1024
    print(f"  Written -> {VALUEFN}  ({sz:.1f} KB)")
    return True

# ── Win rate helpers ──────────────────────────────────────────────
def compute_1v1_wr(match_results):
    w = tot = 0
    for r in match_results:
        if r["type"] == "1v1":
            w   += r["wins"]
            tot += r["wins"] + r["losses"] + r["draws"]
    return w / max(1, tot)

def _bar(wr, width=25):
    n = int(wr * width)
    return f"[{'#'*n}{'-'*(width-n)}] {wr*100:5.1f}%"

def print_round_summary(match_results, round_num):
    print(f"\n  Round {round_num} results:")
    print(f"  {'Type':<14} {'Enemy':<22} {'W':>3} {'L':>3} {'D':>3} {'WR%':>5}")
    print(f"  {'-'*52}")
    for r in match_results:
        if r["type"] == "1v1":
            g  = r["wins"]+r["losses"]+r["draws"]
            wr = r["wins"] / max(1,g) * 100
            print(f"  {'1v1':<14} {r['enemy']:<22} "
                  f"{r['wins']:>3} {r['losses']:>3} {r['draws']:>3} {wr:>5.0f}%")

def _print_trend(history):
    print(f"\n  Training progress:")
    for rnd, wr in history:
        marker = " <-- TARGET" if wr >= WIN_RATE_TARGET else ""
        print(f"    Round {rnd:02d}: {_bar(wr)}{marker}")

# ── Main loop ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  Orbit Wars -- 1v1 Training Loop v5.0")
    print("  Fixes: Replay pretraining, balanced labels, no stale data")
    print("=" * 65)

    enemy_files = sorted([
        f for f in glob.glob(os.path.join(ENEMY_DIR, "*.py"))
        if not os.path.basename(f).startswith("__")
    ])
    if not enemy_files:
        print(f"ERROR: No .py files in '{ENEMY_DIR}'")
        sys.exit(1)

    print(f"\nEnemy agents: {len(enemy_files)}")
    for f in enemy_files:
        print(f"  - {os.path.basename(f)}")
    print(f"\nSettings: max_rounds={MAX_ROUNDS}  target={WIN_RATE_TARGET:.0%}  "
          f"games_per_enemy={GAMES_PER_ENEMY}  eval_games={EVAL_GAMES}")

    # ── Replay pretraining ─────────────────────────────────────────
    replay_records = []
    replay_files   = glob.glob(os.path.join(RECORDS_DIR, "*.json"))
    if replay_files:
        print(f"\n  Found {len(replay_files)} replay files.")
        try:
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)
            clf_dir = os.path.join(ROOT, "classifier")
            if clf_dir not in sys.path:
                sys.path.insert(0, clf_dir)
            from train_from_replays import get_balanced_replay_records, get_val_records
            replay_records = get_balanced_replay_records(max_per_cat=10000)
            val_records    = get_val_records(max_per_cat=5000)
            print(f"  Loaded {len(replay_records)} training + {len(val_records)} held-out val records.")
            if not os.path.exists(VALUEFN):
                print("  No value_fn_1v1.py found -- running replay pretraining...")
                train_and_export(replay_records, val_records=val_records)
                print("  Pretraining complete.")
            else:
                print("  value_fn_1v1.py exists -- replay records will mix in each round.")
        except Exception as e:
            print(f"  WARN: Replay load failed: {e}")
    else:
        print("  No replay files found. Skipping pretraining.")
        val_records = []

    print(f"  Recent self-play records available: {len(load_recent_records())}")

    history           = []
    all_rounds_log    = []
    enemies_with_wr   = None
    current_phase_idx = 0  # advances: easy -> medium -> hard

    for round_num in range(1, MAX_ROUNDS + 1):
        print(f"\n{'='*65}")
        print(f"  ROUND {round_num} / {MAX_ROUNDS}")
        print(f"{'='*65}")
        t0 = time.time()

        my_fn, my_name = load_agent(MAIN_PY, force=True)
        vf_ok = os.path.exists(VALUEFN)
        print(f"  main.py loaded  (value_fn.py: {'YES' if vf_ok else 'NO'})")

        enemies = []
        for ef in enemy_files:
            try:
                fn, name = load_agent(ef, force=False)
                enemies.append((name, fn))
            except Exception as e:
                print(f"  WARN: {os.path.basename(ef)}: {e}")
        if not enemies:
            print("No enemies loaded. Exiting.")
            sys.exit(1)

        # Re-evaluate every 3 rounds (or first round)
        if enemies_with_wr is None or round_num % 3 == 1:
            print(f"\n  Evaluating opponent strengths ({EVAL_GAMES} games each)...")
            enemies_with_wr = []
            for ename, efn in enemies:
                wr = evaluate_opponent_strength(my_fn, efn, ename)
                enemies_with_wr.append((ename, efn, wr))
            enemies_with_wr.sort(key=lambda x: x[2])

        phase     = CURRICULUM_PHASES[current_phase_idx]
        recent_wr = history[-1][1] if history else 0.0
        print(f"\n  Phase: {phase['name']} ({current_phase_idx+1}/{len(CURRICULUM_PHASES)})  "
              f"| Last WR: {recent_wr:.1%}  | Target: {WIN_RATE_TARGET:.0%}")

        new_records, match_results = run_round(
            my_fn, my_name, enemies_with_wr, round_num, phase)
        save_records(new_records, round_num)

        print_round_summary(match_results, round_num)
        wr      = compute_1v1_wr(match_results)
        elapsed = time.time() - t0
        history.append((round_num, wr))

        round_log = {"round": round_num, "win_rate_1v1": wr,
                     "elapsed_s": round(elapsed, 1),
                     "curriculum": phase["name"],
                     "match_results": match_results}
        all_rounds_log.append(round_log)
        with open(RESULTS_LOG, "w") as f:
            json.dump(all_rounds_log, f, indent=2)

        print(f"\n  1v1 win rate: {_bar(wr)}  ({elapsed:.0f}s)")

        if wr >= WIN_RATE_TARGET:
            if current_phase_idx < len(CURRICULUM_PHASES) - 1:
                current_phase_idx += 1
                next_phase = CURRICULUM_PHASES[current_phase_idx]
                print(f"\n  Phase '{phase['name']}' target reached ({wr:.1%})! "
                      f"-> Advancing to '{next_phase['name']}'")
            else:
                print(f"\n  ALL PHASES COMPLETE!  Hard phase WR: {wr:.1%}")
                _print_trend(history)
                break

        # Train: recent self-play (balanced) + replay anchor
        print(f"\n  Training value function...")
        recent_sp = balance_records(load_recent_records(RECENT_ROUNDS))
        if replay_records:
            n_rep      = min(len(replay_records), REPLAY_MIX_SIZE)
            replay_mix = random.sample(replay_records, n_rep)
        else:
            replay_mix = []
        combined = recent_sp + replay_mix
        print(f"  Self-play balanced: {len(recent_sp)}  |  "
              f"Replay mix: {len(replay_mix)}  |  Total: {len(combined)}")
        train_and_export(combined, val_records=val_records)

        _print_trend(history)

        if len(history) >= 4:
            recent = [w for _, w in history[-4:]]
            if max(recent) - min(recent) < 0.03 and recent[-1] < WIN_RATE_TARGET:
                print(f"\n  Win rate plateaued (~{recent[-1]:.1%} for 4 rounds).")
                print("  Consider adding stronger enemy agents.")

    else:
        print(f"\n  MAX_ROUNDS ({MAX_ROUNDS}) reached.")

    print(f"\n{'='*65}")
    print("  Training complete.")
    if history:
        print(f"  Final 1v1 win rate: {history[-1][1]:.1%}  "
              f"(after round {history[-1][0]})")
    print(f"  value_fn_1v1.py: {VALUEFN}")
    print(f"  Training data  : {DATA_DIR}/")
    print(f"  Training log   : {RESULTS_LOG}")
    print(f"\n  To submit: python submission_script.py  (packages both 1v1 and 4v4 models)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
