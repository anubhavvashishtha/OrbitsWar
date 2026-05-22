# ===================================================================
# train_from_replays.py
#
# Parses Kaggle JSON replays in the 'records' folder, extracts 
# features, and trains the value_fn.py model.
#
# Features:
#   - Learns from ANY agent in the replay (not just yours)
#   - Balances data perfectly: 1v1 wins, 1v1 losses, 4p wins, 4p losses
# ===================================================================

import os, sys, glob, json, math, random
from collections import defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
RECORDS_DIR = os.path.join(ROOT, "records")
VALUEFN   = os.path.join(ROOT, "value_fn.py")

# ── Settings ───────────────────────────────────────────────────────
SNAPSHOT_EVERY  = 5
MIN_SAMPLES     = 500
N_ESTIMATORS    = 600
MAX_DEPTH       = 5
LEARNING_RATE   = 0.003
AUC_MIN         = 0.65
MAX_RECORDS_PER_CAT = 15000  # Total up to 60,000 records across 4 categories

# ── Feature extraction (must match main.py exactly) ────────────
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

# ── Replay Parser ─────────────────────────────────────────────────
def parse_replays():
    replay_files = glob.glob(os.path.join(RECORDS_DIR, "*.json"))
    if not replay_files:
        print(f"No JSON replay files found in {RECORDS_DIR}")
        return []

    print(f"Found {len(replay_files)} replay files. Parsing...")
    
    parsed_records = []
    
    for rf in replay_files:
        try:
            with open(rf, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            steps = data.get("steps", [])
            if not steps:
                continue
                
            n_agents = len(steps[0])
            
            # Determine winner from last step
            last_step = steps[-1]
            rewards = [agent.get("reward", 0) for agent in last_step]
            # Handle None rewards
            rewards = [r if r is not None else 0 for r in rewards]
            max_r = max(rewards)
            
            if rewards.count(max_r) > 1 or max_r <= 0:
                winner = -1 # draw
            else:
                winner = rewards.index(max_r)
                
            # Iterate through steps
            for step_idx, step_data in enumerate(steps):
                if step_idx % SNAPSHOT_EVERY != 0:
                    continue
                    
                obs = step_data[0].get("observation", {})
                rp = obs.get("planets", [])
                ci = set(obs.get("comet_planet_ids", []))
                
                # Kaggle JSON might format planets slightly differently, ensure it's a list
                if not isinstance(rp, list):
                    continue
                    
                sd = snap(rp, ci)
                if not sd:
                    continue
                    
                for i in range(n_agents):
                    won = 1 if i == winner else 0
                    feats = extract_features(sd, i, step_idx, n_agents)
                    
                    parsed_records.append({
                        'feats': feats,
                        'turn': step_idx,
                        'won': won,
                        'n_agents': n_agents
                    })
                    
        except Exception as e:
            print(f"Error parsing {rf}: {e}")
            
    return parsed_records

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
        ("GLOBAL", np.ones(len(X), bool), N_ESTIMATORS, MAX_DEPTH),
        ("EARLY",  turns <= 100,          300, 5),
        ("LATE",   turns >= 300,          300, 5),
        ("MID",    (turns > 100) & (turns < 300), 250, 4),
    ]

    lines    = ["# AUTO-GENERATED by train_from_replays.py  --  do not edit", ""]
    exported = []
    any_good = False

    for name, mask, ne, d in specs:
        print(f"  {name:<8} ...", end=" ", flush=True)
        clf, auc = _tr(mask, ne, d)
        if clf is None:
            print(f"    Skipped (not enough variance in data)")
            continue
        print(f"AUC={auc:.3f}  ({mask.sum()} samples)")
        if auc < AUC_MIN:
            print(f"    Skipped (AUC too low)")
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

# ── Main ──────────────────────────────────────────────────────────
def main():
    print("=================================================================")
    print("  Orbit Wars -- Train from Kaggle Replays")
    print("=================================================================")
    
    raw_records = parse_replays()
    if not raw_records:
        print("No valid records extracted. Exiting.")
        return
        
    print(f"\nExtracted {len(raw_records)} total snapshots from replays.")
    
    # Categorize records to balance them
    cat_1v1_wins = [r for r in raw_records if r['n_agents'] == 2 and r['won'] == 1]
    cat_1v1_loss = [r for r in raw_records if r['n_agents'] == 2 and r['won'] == 0]
    cat_4p_wins  = [r for r in raw_records if r['n_agents'] == 4 and r['won'] == 1]
    cat_4p_loss  = [r for r in raw_records if r['n_agents'] == 4 and r['won'] == 0]
    
    print("\nData distribution before balancing:")
    print(f"  1v1 Wins   : {len(cat_1v1_wins)}")
    print(f"  1v1 Losses : {len(cat_1v1_loss)}")
    print(f"  4p Wins    : {len(cat_4p_wins)}")
    print(f"  4p Losses  : {len(cat_4p_loss)}")
    
    # Sample to balance
    balanced_records = []
    for cat, name in [
        (cat_1v1_wins, "1v1 Wins"), 
        (cat_1v1_loss, "1v1 Losses"), 
        (cat_4p_wins, "4p Wins"), 
        (cat_4p_loss, "4p Losses")
    ]:
        sample_size = min(len(cat), MAX_RECORDS_PER_CAT)
        if sample_size > 0:
            sampled = random.sample(cat, sample_size)
            balanced_records.extend(sampled)
            print(f"  Sampled {sample_size} records for {name}")
            
    print(f"\nTotal balanced dataset size: {len(balanced_records)}")
    
    print("\nTraining value function...")
    train_and_export(balanced_records)
    print("\nTraining complete! Your value_fn.py has been updated.")

if __name__ == "__main__":
    main()
