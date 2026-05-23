# ===================================================================
# train_from_replays.py
#
# Parses Kaggle JSON replays in the 'records' folder, extracts
# features, and trains the value_fn.py model.
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
VALUEFN      = os.path.join(PROJECT_ROOT, "value_fn.py")
CACHE_FILE   = os.path.join(ROOT, "replay_cache_v2.pkl")

SNAPSHOT_EVERY      = 5
MIN_SAMPLES         = 500
N_ESTIMATORS        = 600
MAX_DEPTH           = 5
LEARNING_RATE       = 0.003
AUC_MIN             = 0.58
MAX_RECORDS_PER_CAT = 9999999

# TD trajectory parsing (separate from BCE snapshot cache)
TD_SNAPSHOT_EVERY = 1   # every turn — dense transitions for offline TD
TD_CACHE_FILE     = os.path.join(ROOT, "replay_td_cache.pkl")


def extract_features(state, player, turn, n_hint=2, fleets=None):
    ms=mp=mpr=mc=ts=tpr=0.0; ed={}
    msl=[]; mprl=[]; mpos=[]; epos=[]
    for s in state:
        sh=max(0.0,float(s.get('ships',0))); ts+=sh
        ow=s.get('owner',-1)
        if ow==-1: continue
        pr=float(s.get('production',0)); tpr+=pr
        cn=max(0.0,60.0-math.hypot(float(s.get('x',50))-50.0,float(s.get('y',50))-50.0))
        if ow==player:
            ms+=sh; mp+=1; mpr+=pr; mc+=cn
            msl.append(sh); mprl.append(pr)
            mpos.append((float(s.get('x',50)),float(s.get('y',50))))
        else:
            e=ed.setdefault(ow,{'s':0.,'p':0,'pr':0.,'c':0.,'sl':[],'prl':[]})
            e['s']+=sh; e['p']+=1; e['pr']+=pr; e['c']+=cn
            e['sl'].append(sh); e['prl'].append(pr)
            epos.append((float(s.get('x',50)),float(s.get('y',50))))
    bes=bep=bepr=bec=0.0; best_ow=None
    if ed:
        best_ow,be=max(ed.items(),key=lambda kv:kv[1]['s']+100*kv[1]['p'])
        bes=be['s']; bep=float(be['p']); bepr=be['pr']; bec=be['c']/max(1,be['p'])
    tp=max(1,len(state)); sf=lambda a,b:a/b if b>1e-9 else 0.0
    np2=max(2,1+len(ed),n_hint)
    # per-planet features
    my_max_sh=max(msl) if msl else 0.0
    be_sl=ed[best_ow]['sl'] if best_ow else []
    enemy_max_sh=max(be_sl) if be_sl else 0.0
    neutral_cnt=sum(1 for s in state if s.get('owner',-1)==-1)
    my_top3_pr=sum(sorted(mprl,reverse=True)[:3]) if mprl else 0.0
    be_prl=ed[best_ow]['prl'] if best_ow else []
    enemy_top3_pr=sum(sorted(be_prl,reverse=True)[:3]) if be_prl else 0.0
    min_edist=1.0
    if mpos and epos:
        min_edist=min(math.hypot(mx-ex,my-ey)
                      for mx,my in mpos for ex,ey in epos)/100.0
    # fleet features (fleets: list of [id,owner,x,y,angle,from_id,ships])
    mfs=efs=0.0; mfc=efc=0
    for f in (fleets or []):
        fow=f[1] if isinstance(f,(list,tuple)) else f.get('owner',-1)
        fsh=float(f[6] if isinstance(f,(list,tuple)) else f.get('ships',0))
        if fow==player: mfs+=fsh; mfc+=1
        elif fow!=-1:   efs+=fsh; efc+=1
    ta=ts+mfs+efs+1.0
    return [
        turn/500.0, np2/4.0,
        sf(mp,tp), sf(bep,tp),
        sf(ms,ts), sf(bes,ts),
        sf(mpr,tpr), sf(bepr,tpr),
        (mc/max(1,mp))/60.0, bec/60.0,
        sf(ms-bes,ts+1), sf(mp-bep,tp), sf(mpr-bepr,tpr+1),
        1.0 if np2==2 else 0.0,
        1.0 if np2==4 else 0.0,
        sf(ms,bes+1),
        # per-planet & fleet features (features 16-29)
        my_max_sh/max(1.0,ts),    enemy_max_sh/max(1.0,ts),
        neutral_cnt/tp,
        sf(my_top3_pr,tpr),       sf(enemy_top3_pr,tpr),
        min(1.0,min_edist),
        sf(mfs,ta),               sf(efs,ta),
        min(1.0,mfc/8.0),         min(1.0,efc/8.0),
        sf(mfs,ms+1.0),           sf(efs,bes+1.0),
        sf(mpr,mp*5.0+1.0),
        min(1.0,(turn/500.0)*sf(ms,ts+1.0)),
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
                'feats': extract_features(sd, i, step_idx, n_agents, fleets=raw_fleets),
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
    records = _strict_balance(raw, max_per_type=max_per_cat, label="val")
    return records


def get_balanced_replay_records(max_per_cat=MAX_RECORDS_PER_CAT):
    """Load and balance replay records from training files only (80%)."""
    train_files, _ = split_replay_files()
    raw = parse_replays(train_files)
    if not raw:
        return []
    return _strict_balance(raw, max_per_type=max_per_cat, label="train")


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
        ("GLOBAL", np.ones(len(X), bool), N_ESTIMATORS, MAX_DEPTH),
        ("EARLY",  turns <= 100,          500, 5),
        ("LATE",   turns >= 300,          150, 4),
        ("MID",    (turns > 100) & (turns < 300), 200, 4),
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
    lines.append("VALUE_FN_LOADED=True")
    lines.append("")

    with open(VALUEFN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sz = os.path.getsize(VALUEFN) / 1024
    print(f"  Written -> {VALUEFN}  ({sz:.1f} KB)")
    return True


def _parse_replay_td(rf):
    """
    Parse one replay JSON into per-player feature trajectories for offline TD.

    Returns a list of dicts — one per player that survived long enough:
        {'traj': [feats_0, feats_1, ..., feats_T],   # one feats list per step
         'won':  0.0 | 1.0,
         'n_agents': int}

    Steps are sampled at TD_SNAPSHOT_EVERY (1 = every turn).
    """
    with open(rf, "r", encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    if not steps:
        return []

    n_agents  = len(steps[0])
    last_step = steps[-1]
    rewards   = [a.get("reward", 0) or 0 for a in last_step]
    max_r     = max(rewards)
    winner    = -1 if (rewards.count(max_r) > 1 or max_r <= 0) else rewards.index(max_r)

    trajectories = [[] for _ in range(n_agents)]

    for step_idx, step_data in enumerate(steps):
        if step_idx % TD_SNAPSHOT_EVERY != 0:
            continue
        obs = step_data[0].get("observation", {})
        rp  = obs.get("planets", [])
        ci  = set(obs.get("comet_planet_ids", []) or [])
        rf_fleets = obs.get("fleets", []) or []
        if not isinstance(rp, list) or not rp:
            continue
        sd = snap(rp, ci)
        if not sd:
            continue
        for i in range(n_agents):
            feats = extract_features(sd, i, step_idx, n_agents, fleets=rf_fleets)
            trajectories[i].append(feats)

    result = []
    for i, traj in enumerate(trajectories):
        if len(traj) < 2:
            continue
        won = 1.0 if i == winner else 0.0
        result.append({"traj": traj, "won": won, "n_agents": n_agents})
    return result


def get_td_trajectories(replay_files=None):
    """
    Load per-player feature trajectories from replay files for offline TD training.

    Uses a separate cache (TD_CACHE_FILE) so it doesn't interfere with the
    existing BCE snapshot cache (replay_cache_v2.pkl).
    Only training files (80% split) are used.

    Returns list of {'traj': [feats, ...], 'won': float, 'n_agents': int}.
    """
    if replay_files is None:
        train_files, _ = split_replay_files()
        replay_files = train_files

    cache = {}
    if os.path.exists(TD_CACHE_FILE):
        try:
            with open(TD_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
        except Exception:
            cache = {}

    all_trajectories = []
    cache_updated    = False
    n_cached = n_fresh = 0

    for i, rf in enumerate(replay_files):
        cache_key = f"td|{rf}|{os.path.getmtime(rf):.0f}"
        if cache_key in cache:
            all_trajectories.extend(cache[cache_key])
            n_cached += 1
        else:
            try:
                trajs = _parse_replay_td(rf)
                cache[cache_key] = trajs
                all_trajectories.extend(trajs)
                cache_updated = True
                n_fresh += 1
            except Exception as e:
                print(f"  Error parsing TD {os.path.basename(rf)}: {e}")

        if (i + 1) % 100 == 0 or (i + 1) == len(replay_files):
            sys.stdout.write(
                f"\r  TD parse {i+1}/{len(replay_files)}"
                f"  (cached:{n_cached}  fresh:{n_fresh})   "
            )
            sys.stdout.flush()

    print()

    if cache_updated:
        try:
            with open(TD_CACHE_FILE, "wb") as f:
                pickle.dump(cache, f)
            sz = os.path.getsize(TD_CACHE_FILE) / 1024 / 1024
            print(f"  TD cache saved -> {TD_CACHE_FILE}  ({sz:.1f} MB)")
        except Exception as e:
            print(f"  WARN: Could not save TD cache: {e}")

    return all_trajectories


def _compute_td_values(trajectories, gamma=0.99, reward_scale=0.1):
    """
    Compute discounted TD returns V(s_t) for every step of every trajectory.

    Algorithm (backward pass through each trajectory):
        V(s_T)   = won                               (terminal ground truth)
        V(s_t)   = shaped_r(s_t, s_{t+1}) + gamma * V(s_{t+1})

    Shaped reward uses feature indices already stored in each feats vector:
        index 2 = my_planet_share
        index 4 = my_ship_share
        index 6 = my_prod_share

    Returns list of (feats, V) pairs where V is in [0, 1].
    """
    pairs = []
    for td in trajectories:
        traj = td["traj"]   # list of 30-dim feature vectors
        won  = td["won"]    # 0.0 or 1.0
        T    = len(traj)

        # Per-step shaped rewards (index 0 is undefined, set to 0)
        shaped = [0.0] * T
        for t in range(1, T):
            d_ships   = traj[t][4] - traj[t-1][4]
            d_prod    = traj[t][6] - traj[t-1][6]
            d_planets = traj[t][2] - traj[t-1][2]
            shaped[t] = reward_scale * (0.4*d_ships + 0.4*d_prod + 0.2*d_planets)

        # Backward pass: V(s_T) = won, V(s_t) = r_{t+1} + gamma*V(s_{t+1})
        V = [0.0] * T
        V[T - 1] = won
        for t in range(T - 2, -1, -1):
            V[t] = shaped[t + 1] + gamma * V[t + 1]

        for t in range(T):
            pairs.append((traj[t], V[t]))

    return pairs


def train_and_export_regressor(gamma=0.99, reward_scale=0.1,
                                n_estimators=600, max_depth=5,
                                learning_rate=0.003):
    """
    Train a GradientBoostingRegressor on TD regression targets and export
    to value_fn.py in the same format as the classifier.

    Why regression improves small-signal learning
    ---------------------------------------------
    Binary labels collapse every winning snapshot to 1 and every losing one
    to 0.  A state where you're barely ahead at turn 200 and a state where
    you're dominating at turn 480 both get label 1.  The regressor instead
    gets V(s_t) = discounted return from that state, so:
        dominating win at t=480  -> V ~ 0.97
        barely ahead at t=200    -> V ~ 0.60
        neutral at t=100         -> V ~ 0.50
        barely losing at t=200   -> V ~ 0.40
        dominating loss at t=480 -> V ~ 0.03

    Targets are logit-transformed before training so _sig() in main.py
    correctly converts them back to probabilities at inference — no changes
    to main.py required.
    """
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error
    except ImportError:
        print("pip install scikit-learn numpy"); return False

    print("Loading TD trajectories (train split)...")
    train_files, val_files = split_replay_files()
    train_trajs = get_td_trajectories(replay_files=train_files)
    print("Loading TD trajectories (val split)...")
    val_trajs   = get_td_trajectories(replay_files=val_files)

    if not train_trajs:
        print("No trajectories found."); return False

    print("Computing TD regression targets...")
    train_pairs = _compute_td_values(train_trajs, gamma=gamma, reward_scale=reward_scale)
    val_pairs   = _compute_td_values(val_trajs,   gamma=gamma, reward_scale=reward_scale)
    random.shuffle(train_pairs)

    X_train = np.array([p[0] for p in train_pairs], dtype=np.float32)
    y_train = np.array([p[1] for p in train_pairs], dtype=np.float32)
    X_val   = np.array([p[0] for p in val_pairs],   dtype=np.float32)
    y_val   = np.array([p[1] for p in val_pairs],   dtype=np.float32)

    # Logit-transform targets so _sig() in main.py works correctly at inference
    eps = 1e-6
    logit_train = np.log(np.clip(y_train, eps, 1-eps) / (1 - np.clip(y_train, eps, 1-eps)))
    logit_val   = np.log(np.clip(y_val,   eps, 1-eps) / (1 - np.clip(y_val,   eps, 1-eps)))

    print(f"\nTrain: {len(X_train):,} samples  |  Val: {len(X_val):,} samples")
    print(f"V target range: [{y_train.min():.3f}, {y_train.max():.3f}]  "
          f"mean={y_train.mean():.3f}  std={y_train.std():.3f}")

    # Turn index: feature[0] = turn / 500
    train_turns = X_train[:, 0] * 500
    val_turns   = X_val[:,   0] * 500

    specs = [
        ("GLOBAL", np.ones(len(X_train), bool),  np.ones(len(X_val), bool),  n_estimators, max_depth),
    ]

    lines    = ["# AUTO-GENERATED by train_from_replays.py (TD regressor) -- do not edit", ""]
    exported = []

    for name, tr_mask, va_mask, ne, d in specs:
        n_tr = tr_mask.sum()
        n_va = va_mask.sum()
        if n_tr < 200:
            print(f"  {name:<8} Skipped (only {n_tr} samples)")
            continue

        print(f"\n  {name:<8} {n_tr:>7,} train  {n_va:>7,} val  ...")

        reg = GradientBoostingRegressor(
            n_estimators=ne, max_depth=d,
            learning_rate=learning_rate, subsample=0.8,
            min_samples_leaf=30, random_state=42,
            loss="squared_error",
            verbose=1,
        )
        reg.fit(X_train[tr_mask], logit_train[tr_mask])

        # Validation: convert logit predictions back to probability, measure MSE vs V
        if n_va >= 10:
            pred_logit = reg.predict(X_val[va_mask])
            pred_prob  = 1.0 / (1.0 + np.exp(-pred_logit))
            val_mse    = mean_squared_error(y_val[va_mask], pred_prob)
            # Also compare to naive baseline (predict mean V for everyone)
            baseline   = mean_squared_error(y_val[va_mask],
                                            np.full(n_va, y_train[tr_mask].mean()))
            print(f"val_MSE={val_mse:.4f}  baseline={baseline:.4f}  "
                  f"improve={100*(1-val_mse/baseline):.1f}%")
        else:
            print("(no val samples)")

        # Export in same format as classifier — INIT is mean(logit_train) for this subset,
        # which is what DummyRegressor(strategy='mean') predicts
        init_val = float(logit_train[tr_mask].mean())

        lines.append(f"{name}_INIT={init_val:.8f}")
        lines.append(f"{name}_N_TREES={len(reg.estimators_)}")
        lines.append(f"{name}_TREES=[")
        for ta in reg.estimators_:
            t  = ta[0].tree_
            fi = [int(v)                              for v in t.feature]
            th = [round(float(v), 6)                  for v in t.threshold]
            va_vals = [round(float(t.value[i,0,0]),8) for i in range(t.node_count)]
            le = [int(v)                              for v in t.children_left]
            ri = [int(v)                              for v in t.children_right]
            lines.append(f"    ({fi},{th},{va_vals},{le},{ri}),")
        lines.append("]")
        lines.append("")
        exported.append(name)

    if not exported:
        print("No models exported."); return False

    lines.append(f"VALUE_FN_MODELS={exported!r}")
    lines.append("VALUE_FN_LOADED=True")
    lines.append("")

    with open(VALUEFN, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sz = os.path.getsize(VALUEFN) / 1024
    print(f"\nExported (TD regressor) -> {VALUEFN}  ({sz:.1f} KB)")
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

    balanced_records = _strict_balance(raw_train, max_per_type=MAX_RECORDS_PER_CAT, label="train")
    print(f"\nTotal balanced training size: {len(balanced_records)}")

    print("\nLoading held-out validation records...")
    val_records = get_val_records(max_per_cat=5000)

    print("\nTraining value function...")
    train_and_export(balanced_records, val_records=val_records)
    print("\nTraining complete! Your value_fn.py has been updated.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Train Orbit Wars value function from replays")
    p.add_argument("--regressor", action="store_true",
                   help="Train TD GBM regressor (value_fn.py for Submission 2)")
    p.add_argument("--gamma",        type=float, default=0.99,  help="TD discount (default 0.99)")
    p.add_argument("--reward-scale", type=float, default=0.1,   help="Shaped reward scale (default 0.1)")
    args = p.parse_args()

    if args.regressor:
        train_and_export_regressor(gamma=args.gamma, reward_scale=args.reward_scale)
    else:
        main()
