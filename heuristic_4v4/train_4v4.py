# ===================================================================
# train_from_replays.py
#
# Parses Kaggle JSON replays in the 'records' folder, extracts
# features, and trains the value_fn_regresoor.py model.
#
# Features:
#   - Learns from ANY agent in the replay (not just yours)
#   - Balances data perfectly: 1v1 wins, 1v1 losses, 4p wins, 4p losses
#   - get_balanced_replay_records() is importable by train.py / ffa_train.py
# ===================================================================

import os, sys, glob, json, math, random, pickle
from pathlib import Path

ROOT         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
RECORDS_DIR  = os.path.join(PROJECT_ROOT, "records")
VALUEFN      = os.path.join(ROOT, "value_fn_4v4.py")
CACHE_FILE   = os.path.join(ROOT, "replay_cache_4v4.pkl")

SNAPSHOT_EVERY      = 5
MIN_SAMPLES         = 500
N_ESTIMATORS        = 600
MAX_DEPTH           = 5
LEARNING_RATE       = 0.003
Bronze_AUC_MIN      = 0.58
AUC_MIN             = 0.58
MAX_RECORDS_PER_CAT = 9999999



def extract_features_4v4(state, player, turn, n_hint=2, fleets=None):
    ms=mp=mpr=mc=ts=tpr=0.0; ed={}
    msl=[]; mprl=[]; mpos=[]; epos=[]
    
    my_static_ships = 0.0
    enemy_static_ships = 0.0
    my_tanks = 0
    enemy_tanks = 0
    my_quad_prod = [0.0]*4
    my_quad_ships = [0.0]*4
    my_inner = 0.0
    my_outer = 0.0
    
    for s in state:
        sh=max(0.0,float(s.get('ships',0))); ts+=sh
        ow=s.get('owner',-1)
        if ow==-1: continue
        pr=float(s.get('production',0)); tpr+=pr
        px=float(s.get('x',50)); py=float(s.get('y',50))
        dist_c = ( (px-50)**2 + (py-50)**2 )**0.5
        cn=max(0.0,60.0-dist_c)
        
        qx=px-50; qy=py-50
        qidx = 0 if qx>=0 and qy>=0 else 1 if qx<0 and qy>=0 else 2 if qx<0 and qy<0 else 3
        is_stat = (dist_c >= 40.0)
        is_tank = (pr >= 3.0)
        
        if ow==player:
            ms+=sh; mp+=1; mpr+=pr; mc+=cn
            msl.append(sh); mprl.append(pr); mpos.append((px,py))
            if is_stat: my_static_ships += sh
            if is_tank: my_tanks += 1
            my_quad_prod[qidx] += pr; my_quad_ships[qidx] += sh
            if dist_c < 25: my_inner += sh
            else: my_outer += sh
        else:
            e=ed.setdefault(ow,{'s':0.,'p':0,'pr':0.,'c':0.,'sl':[],'prl':[]})
            e['s']+=sh; e['p']+=1; e['pr']+=pr; e['c']+=cn
            e['sl'].append(sh); e['prl'].append(pr); epos.append((px,py))
            if is_stat: enemy_static_ships += sh
            if is_tank: enemy_tanks += 1

    cx, cy = 50.0, 50.0
    if mpos: cx=sum(x for x,y in mpos)/len(mpos); cy=sum(y for x,y in mpos)/len(mpos)

    bes=bep=bepr=bec=0.0; best_ow=None
    if ed:
        best_ow,be=max(ed.items(),key=lambda kv:kv[1]['s']+100*kv[1]['p'])
        bes=be['s']; bep=float(be['p']); bepr=be['pr']; bec=be['c']/max(1,be['p'])
        
    tp=max(1,len(state)); sf=lambda a,b:a/b if b>1e-9 else 0.0
    np2=max(2,1+len(ed),n_hint)
    
    my_max_sh=max(msl) if msl else 0.0
    be_sl=ed[best_ow]['sl'] if best_ow else []
    enemy_max_sh=max(be_sl) if be_sl else 0.0
    neutral_cnt=sum(1 for s in state if s.get('owner',-1)==-1)
    my_top3_pr=sum(sorted(mprl,reverse=True)[:3]) if mprl else 0.0
    be_prl=ed[best_ow]['prl'] if best_ow else []
    enemy_top3_pr=sum(sorted(be_prl,reverse=True)[:3]) if be_prl else 0.0
    min_edist=1.0
    if mpos and epos:
        min_edist=min(((mx-ex)**2+(my-ey)**2)**0.5 for mx,my in mpos for ex,ey in epos)/100.0

    mfs=efs=0.0; mfc=efc=0
    in_att = {}; my_tgt = 0.0; my_off = 0.0; en_tgt = 0.0
    pid2ow = {s.get('id',-1): s.get('owner',-1) for s in state}
    pid2pos = {s.get('id',-1): (float(s.get('x',50)),float(s.get('y',50))) for s in state}
    att_dists = []
    
    for f in (fleets or []):
        fow=f[1] if isinstance(f,(list,tuple)) else f.get('owner',-1)
        fsh=float(f[6] if isinstance(f,(list,tuple)) else f.get('ships',0))
        tid = f[5] if isinstance(f,(list,tuple)) and len(f)>5 else f.get('target_id',-1)
        
        if fow==player: 
            mfs+=fsh; mfc+=1
            if tid!=-1 and pid2ow.get(tid,-1)!=player: my_off+=fsh
        elif fow!=-1:   
            efs+=fsh; efc+=1
            if tid!=-1:
                tow = pid2ow.get(tid,-1)
                if tow==player:
                    my_tgt+=fsh
                    px,py = pid2pos.get(tid,(50,50))
                    att_dists.append(((px-cx)**2+(py-cy)**2)**0.5)
                else:
                    if tow!=-1 and tow!=player: en_tgt+=fsh
                    if tid not in in_att: in_att[tid]=[]
                    in_att[tid].append((fow,fsh))
                    
    ta=ts+mfs+efs+1.0
    
    eve_conf = 0; eve_sh = 0.0
    for tid, atts in in_att.items():
        uniq = set(a[0] for a in atts)
        if len(uniq)>=2:
            eve_conf+=1; eve_sh+=sum(a[1] for a in atts)
            
    my_home_prod = max(my_quad_prod) if my_quad_prod else 0.0
    quad_dom = sf(max(my_quad_ships) if my_quad_ships else 0.0, ms)
    avg_att_dist = (sum(att_dists)/len(att_dists)/100.0) if att_dists else 0.0

    return [
        turn/500.0, np2/4.0, sf(mp,tp), sf(bep,tp), sf(ms,ts), sf(bes,ts),
        sf(mpr,tpr), sf(bepr,tpr), (mc/max(1,mp))/60.0, bec/60.0,
        sf(ms-bes,ts+1), sf(mp-bep,tp), sf(mpr-bepr,tpr+1),
        1.0 if np2==2 else 0.0, 1.0 if np2==4 else 0.0, sf(ms,bes+1),
        my_max_sh/max(1.0,ts), enemy_max_sh/max(1.0,ts), neutral_cnt/tp,
        sf(my_top3_pr,tpr), sf(enemy_top3_pr,tpr), min(1.0,min_edist),
        sf(mfs,ta), sf(efs,ta), min(1.0,mfc/8.0), min(1.0,efc/8.0),
        sf(mfs,ms+1.0), sf(efs,bes+1.0), sf(mpr,mp*5.0+1.0), min(1.0,(turn/500.0)*sf(ms,ts+1.0)),
        sf(my_static_ships, ts), sf(enemy_static_ships, ts), my_tanks/max(1.0,tp), enemy_tanks/max(1.0,tp),
        sf(my_tgt, ms+1.0), sf(my_off, ms+1.0), eve_conf/10.0, sf(eve_sh, ts),
        sf(my_home_prod, tpr), quad_dom, avg_att_dist, sf(my_inner, ms), sf(my_outer, ms), sf(en_tgt, ts)
    ]

def snap(raw_p, comet_ids=None):
    cids = comet_ids or set()
    return [{'id':p[0],'owner':p[1],'ships':p[5],
             'production':p[6],'x':p[2],'y':p[3]}
            for p in raw_p if p[0] not in cids]


def split_replay_files(records_dir=RECORDS_DIR):
    """Deterministic 80/20 train/val split at file level — prevents data leakage."""
    all_files   = sorted(glob.glob(os.path.join(records_dir, "*.json")))
    val_files   = [f for i, f in enumerate(all_files) if i % 5 == 0]
    train_files = [f for i, f in enumerate(all_files) if i % 5 != 0]
    return train_files, val_files


def _parse_single_replay(rf):
    """Parse one replay file into a list of feature records."""
    records = []
    with open(rf, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    if not steps:
        return records
    n_agents  = len(steps[0])
    last_step = steps[-1]
    rewards   = [agent.get("reward", 0) for agent in last_step]
    rewards   = [r if r is not None else 0 for r in rewards]
    max_r     = max(rewards)
    winner    = -1 if (rewards.count(max_r) > 1 or max_r <= 0) else rewards.index(max_r)
    for step_idx, step_data in enumerate(steps):
        if step_idx % SNAPSHOT_EVERY != 0:
            continue
        obs = step_data[0].get("observation", {})
        rp  = obs.get("planets", [])
        ci  = set(obs.get("comet_planet_ids", []))
        raw_fleets = obs.get("fleets", []) or []
        if not isinstance(rp, list):
            continue
        sd = snap(rp, ci)
        if not sd:
            continue
        for i in range(n_agents):
            records.append({
                'feats': extract_features_4v4(sd, i, step_idx, n_agents, fleets=raw_fleets),
                'turn':  step_idx,
                'won':   1 if i == winner else 0,
                'n_agents': n_agents
            })
    return records


def parse_replays(replay_files=None):
    """Parse replays with per-file caching — only new files are re-parsed."""
    if replay_files is None:
        replay_files = glob.glob(os.path.join(RECORDS_DIR, "*.json"))
    if not replay_files:
        print(f"No JSON replay files found in {RECORDS_DIR}")
        return []

    # Load existing cache
    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    all_records   = []
    cache_updated = False
    n_cached = n_fresh = 0

    for i, rf in enumerate(replay_files):
        cache_key = f"{rf}|{os.path.getmtime(rf):.0f}"
        if cache_key in cache:
            all_records.extend(cache[cache_key])
            n_cached += 1
        else:
            try:
                records = _parse_single_replay(rf)
                cache[cache_key] = records
                all_records.extend(records)
                cache_updated = True
                n_fresh += 1
            except Exception as e:
                print(f"  Error parsing {os.path.basename(rf)}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(replay_files):
            sys.stdout.write(f"\r  Parsed {i+1}/{len(replay_files)}  "
                             f"(cached: {n_cached}, fresh: {n_fresh})   ")
            sys.stdout.flush()

    print()

    if cache_updated:
        try:
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(cache, f)
            sz = os.path.getsize(CACHE_FILE) / 1024 / 1024
            print(f"  Cache saved -> {CACHE_FILE}  ({sz:.1f} MB)")
        except Exception as e:
            print(f"  WARN: Could not save cache: {e}")

    return all_records


def _strict_balance(raw, max_per_type=MAX_RECORDS_PER_CAT, label=""):
    """Balances by oversampling the minority class to match the majority class, keeping all data."""
    c1w = [r for r in raw if r['n_agents'] == 2 and r['won'] == 1]
    c1l = [r for r in raw if r['n_agents'] == 2 and r['won'] == 0]
    c4w = [r for r in raw if r['n_agents'] == 4 and r['won'] == 1]
    c4l = [r for r in raw if r['n_agents'] == 4 and r['won'] == 0]
    
    # Randomly duplicate the smaller class to match the larger class
    # 1v1
    max_1 = max(len(c1w), len(c1l))
    max_1 = min(max_1, max_per_type)
    if len(c1w) > 0 and len(c1w) < max_1: c1w += random.choices(c1w, k=max_1 - len(c1w))
    if len(c1l) > 0 and len(c1l) < max_1: c1l += random.choices(c1l, k=max_1 - len(c1l))
    c1w = c1w[:max_1]; c1l = c1l[:max_1]
    
    # 4p
    max_4 = max(len(c4w), len(c4l))
    max_4 = min(max_4, max_per_type)
    if len(c4w) > 0 and len(c4w) < max_4: c4w += random.choices(c4w, k=max_4 - len(c4w))
    if len(c4l) > 0 and len(c4l) < max_4: c4l += random.choices(c4l, k=max_4 - len(c4l))
    c4w = c4w[:max_4]; c4l = c4l[:max_4]

    out = c1w + c1l + c4w + c4l
    tag = f"  [{label}] " if label else "  "
    print(f"{tag}Oversampled balanced: {len(out)} records  "
          f"(1v1: {len(c1w)}W/{len(c1l)}L  |  4p: {len(c4w)}W/{len(c4l)}L)")
    return out


def get_val_records(max_per_cat=5000):
    """Load held-out validation records (20% of files, never used in training)."""
    _, val_files = split_replay_files()
    raw = parse_replays(val_files)
    if not raw:
        return []
    records = _strict_balance([r for r in raw if r['n_agents'] == 4], max_per_type=max_per_cat, label="val")
    return records


def get_balanced_replay_records(max_per_cat=MAX_RECORDS_PER_CAT):
    """Load and balance replay records from training files only (80%)."""
    train_files, _ = split_replay_files()
    raw = parse_replays(train_files)
    if not raw:
        return []
    return _strict_balance([r for r in raw if r['n_agents'] == 4], max_per_type=max_per_cat, label="train")


def train_and_export(all_records, val_records=None):
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  ERROR: pip install scikit-learn numpy")
        return False

    Xval = yval = None
    if val_records:
        Xval = np.array([r['feats'] for r in val_records], dtype=np.float32)
        yval = np.array([r['won']   for r in val_records], dtype=np.int32)

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
            min_samples_leaf=30, random_state=42,
            verbose=1)
        clf.fit(Xt, yt)
        return clf, roc_auc_score(yv, clf.predict_proba(Xv)[:,1])

    specs = [
        ("GLOBAL", np.ones(len(X), bool),             N_ESTIMATORS, MAX_DEPTH),
        ("EARLY",  turns <= 100,                      600,          5),
        ("LATE",   turns >= 300,                      200,          3),
        ("MID",    (turns > 100) & (turns < 300),     300,          3),
    ]

    lines    = ["# AUTO-GENERATED by train_from_replays.py  --  do not edit", ""]
    exported = []
    any_good = False

    for name, mask, ne, d in specs:
        print(f"\n  {name:<8} ...")
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
        val_str = ""
        if Xval is not None and yval is not None:
            val_auc = roc_auc_score(yval, clf.predict_proba(Xval)[:,1])
            val_str = f"  held-out AUC={val_auc:.3f}"
        print(f"    Exported {name}  ({len(clf.estimators_)} trees,  AUC={auc:.3f}{val_str})")

    if not any_good:
        print("  No models met AUC threshold.")
        return False

    lines.append(f"VALUE_FN_MODELS={exported!r}")
    lines.append("VALUE_FN_4V4_LOADED=True")
    lines.append("")

    with open(VALUEFN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sz = os.path.getsize(VALUEFN) / 1024
    print(f"  Written -> {VALUEFN}  ({sz:.1f} KB)")
    return True



def main():
    print("=================================================================")
    print("  Orbit Wars -- Train from Kaggle Replays")
    print("=================================================================")

    train_files, val_files = split_replay_files()
    print(f"\nReplay split: {len(train_files)} train files / {len(val_files)} val files (held-out)")

    raw_train = parse_replays(train_files)
    if not raw_train:
        print("No valid records extracted from training files. Exiting.")
        return

    print(f"\nExtracted {len(raw_train)} snapshots from training replays.")

    c1w = [r for r in raw_train if r['n_agents'] == 2 and r['won'] == 1]
    c1l = [r for r in raw_train if r['n_agents'] == 2 and r['won'] == 0]
    c4w = [r for r in raw_train if r['n_agents'] == 4 and r['won'] == 1]
    c4l = [r for r in raw_train if r['n_agents'] == 4 and r['won'] == 0]

    print("\nTraining data distribution (raw):")
    print(f"  1v1 Wins   : {len(c1w)}")
    print(f"  1v1 Losses : {len(c1l)}")
    print(f"  4p Wins    : {len(c4w)}")
    print(f"  4p Losses  : {len(c4l)}")

    balanced_records = _strict_balance([r for r in raw_train if r['n_agents'] == 4], max_per_type=MAX_RECORDS_PER_CAT, label="train")
    print(f"\nTotal balanced training size: {len(balanced_records)}")

    print("\nLoading held-out validation records...")
    val_records = get_val_records(max_per_cat=5000)

    print("\nTraining value function...")
    train_and_export(balanced_records, val_records=val_records)
    print("\nTraining complete! Your value_fn_4v4.py has been updated.")


if __name__ == "__main__":
    main()
