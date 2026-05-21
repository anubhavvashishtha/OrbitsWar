# ===================================================================
# train_value_fn.py
#
# Collects game data by playing main.py vs the 3 enemy agents,
# trains a GBC value function, and writes value_fn.py.
#
# Run this AFTER run_simulations.py has played some games,
# OR run it standalone — it will play its own training games.
#
# Usage:
#   python train_value_fn.py
#
# Output:
#   value_fn.py   (baked-in weights, add to zip for submission)
#   training_data/records_round_NN.json  (raw game records)
# ===================================================================

import os, sys, math, json, glob, importlib.util, random
from collections import namedtuple

if hasattr(sys.stdout,'reconfigure'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

# ── Paths ────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
MAIN_PY    = os.path.join(ROOT, "main.py")
ENEMY_DIR  = os.path.join(ROOT, "enemy agents")
DATA_DIR   = os.path.join(ROOT, "training_data")
VALUEFN    = os.path.join(ROOT, "value_fn.py")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────
GAMES_PER_ENEMY   = 30    # games per enemy agent per training run
                           # increase for better models (50-100 ideal)
SNAPSHOT_EVERY    = 5     # record board state every N turns
MIN_SAMPLES       = 300   # don't export model below this
N_ESTIMATORS      = 500
MAX_DEPTH         = 4
LEARNING_RATE     = 0.005
AUC_MIN           = 0.62  # minimum AUC to export

Planet = namedtuple("Planet",["id","owner","x","y","radius","ships","production"])

# ── Utilities ─────────────────────────────────────────────────────
def load_agent(path, force=False):
    name = os.path.splitext(os.path.basename(path))[0]
    d    = os.path.dirname(os.path.abspath(path))
    if d not in sys.path: sys.path.insert(0,d)
    if force:
        for k in [name,"value_fn"]:
            if k in sys.modules: del sys.modules[k]
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    mod.__file__ = os.path.abspath(path)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent

def extract_features(state, player, turn, n_hint=2):
    ms=mp=mpr=mc=ts=tpr=0.0; ed={}
    for s in state:
        sh=max(0.0,float(s.get('ships',0))); ts+=sh
        ow=s.get('owner',-1)
        if ow==-1: continue
        pr=float(s.get('production',0)); tpr+=pr
        cn=max(0.0,60.0-math.hypot(float(s.get('x',50))-50.0,float(s.get('y',50))-50.0))
        if ow==player: ms+=sh; mp+=1; mpr+=pr; mc+=cn
        else:
            e=ed.setdefault(ow,{'s':0.,'p':0,'pr':0.,'c':0.})
            e['s']+=sh; e['p']+=1; e['pr']+=pr; e['c']+=cn
    bes=bep=bepr=bec=0.0
    if ed:
        be=max(ed.values(),key=lambda e:e['s']+100*e['p'])
        bes=be['s']; bep=float(be['p']); bepr=be['pr']; bec=be['c']/max(1,be['p'])
    tp=max(1,len(state)); sf=lambda a,b:a/b if b>1e-9 else 0.0
    np2=max(2,1+len(ed),n_hint)
    return [turn/500.0,np2/4.0,sf(mp,tp),sf(bep,tp),sf(ms,ts),sf(bes,ts),
            sf(mpr,tpr),sf(bepr,tpr),(mc/max(1,mp))/60.0,bec/60.0,
            sf(ms-bes,ts+1),sf(mp-bep,tp),sf(mpr-bepr,tpr+1),
            1.0 if np2==2 else 0.0,1.0 if np2==4 else 0.0,sf(ms,bes+1)]

def snap(raw_planets, comet_ids=None):
    comet_ids=comet_ids or set()
    return [{'id':p[0],'owner':p[1],'ships':p[5],'production':p[6],'x':p[2],'y':p[3]}
            for p in raw_planets if p[0] not in comet_ids]

# ── Game runner ───────────────────────────────────────────────────
def play_game(my_fn, enemy_fn, seed=0, n_agents=2):
    """Play one game. Returns (records_per_player_list, rewards_list, steps)."""
    from kaggle_environments import make
    env = make("orbit_wars",configuration={"seed":seed,"randomSeed":seed},debug=False)
    env.reset(num_agents=n_agents)
    snaps = [[] for _ in range(n_agents)]
    states = env.step([[]]*n_agents)
    step=0
    while True:
        statuses=[str(s.status if hasattr(s,'status') else s.get('status','ACTIVE')) for s in states]
        if statuses[0]!='ACTIVE': break
        obss=[(s.observation if hasattr(s,'observation') else s.get('observation')) for s in states]
        # snapshot
        if step%SNAPSHOT_EVERY==0 and obss[0] is not None:
            raw_p=(obss[0].get("planets",[]) if isinstance(obss[0],dict) else getattr(obss[0],"planets",[]))
            cids=set(obss[0].get("comet_planet_ids",[]) if isinstance(obss[0],dict) else getattr(obss[0],"comet_planet_ids",[]))
            sd=snap(raw_p,cids)
            if sd:
                for i in range(n_agents):
                    snaps[i].append((step,extract_features(sd,i,step,n_agents)))
        actions=[]
        for i,obs in enumerate(obss):
            try: a=(my_fn if i==0 else enemy_fn)(obs)
            except: a=[]
            actions.append(a if a else [])
        states=env.step(actions); step+=1
    rewards=[float((s.reward if hasattr(s,'reward') else s.get('reward',0)) or 0) for s in states]
    winner=rewards.index(max(rewards)) if max(rewards)>0 else -1
    records=[]
    for i in range(n_agents):
        won=1 if i==winner else 0
        records.append([{'feats':f,'turn':t,'won':won,'player':i} for t,f in snaps[i]])
    return records, rewards, step

# ── Collect data ──────────────────────────────────────────────────
def collect_data(my_fn, enemy_agents, round_id=0):
    """Play games vs each enemy, collect records. Returns flat record list."""
    all_records=[]
    total=len(enemy_agents)*GAMES_PER_ENEMY
    done=0
    for ename,efn in enemy_agents:
        w=l=d=0
        for g in range(GAMES_PER_ENEMY):
            seed=round_id*10000+done
            recs,rewards,steps=play_game(my_fn,efn,seed=seed)
            p0r=rewards[0]
            if p0r>0: w+=1
            elif p0r<0: l+=1
            else: d+=1
            # only keep non-draws
            if abs(p0r)>0.01:
                for player_recs in recs:
                    all_records.extend(player_recs)
            done+=1
            pct=int(done/total*30)
            sys.stdout.write(f"\r  [{'#'*pct}{'-'*(30-pct)}] {done}/{total}  "
                             f"vs {ename}: W{w} L{l} D{d}  ({steps}st)")
            sys.stdout.flush()
    print()
    return all_records

# ── Train GBC ─────────────────────────────────────────────────────
def train(records):
    try:
        import numpy as np
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("  ERROR: pip install scikit-learn numpy")
        return {}
    if len(records)<MIN_SAMPLES:
        print(f"  Only {len(records)} samples, need {MIN_SAMPLES}. Skipping.")
        return {}
    X=np.array([r['feats'] for r in records],dtype=np.float32)
    y=np.array([r['won']   for r in records],dtype=np.int32)
    turns=np.array([r['turn'] for r in records])
    models={}
    def _tr(name,mask,ne=N_ESTIMATORS,d=MAX_DEPTH):
        Xs,ys=X[mask],y[mask]
        if len(Xs)<80 or ys.mean()<0.02 or ys.mean()>0.98: return None,0.0
        Xt,Xv,yt,yv=train_test_split(Xs,ys,test_size=0.2,random_state=42,
            stratify=ys if min(ys.sum(),(1-ys).sum())>=10 else None)
        clf=GradientBoostingClassifier(n_estimators=ne,max_depth=d,
            learning_rate=LEARNING_RATE,subsample=0.8,min_samples_leaf=30,random_state=42)
        clf.fit(Xt,yt)
        auc=roc_auc_score(yv,clf.predict_proba(Xv)[:,1])
        return clf,auc
    for name,mask,ne,d in [
        ("GLOBAL", np.ones(len(X),bool), N_ESTIMATORS, MAX_DEPTH),
        ("EARLY",  turns<=100,           200, 4),
        ("LATE",   turns>=300,           200, 4),
    ]:
        print(f"  {name:<8} ...",end=" ",flush=True)
        clf,auc=_tr(name,mask,ne,d)
        models[name]=(clf,auc)
        print(f"AUC={auc:.3f}  ({mask.sum()} samples)")
    return models

# ── Export ────────────────────────────────────────────────────────
def export(models, path=VALUEFN):
    lines=["# AUTO-GENERATED by train_value_fn.py",""]
    exported=[]
    for name,(clf,auc) in models.items():
        if clf is None or auc<AUC_MIN:
            print(f"  Skip {name}: AUC={auc:.3f}")
            continue
        lines.append(f"{name}_INIT={float(clf.init_.class_prior_[1]):.8f}")
        lines.append(f"{name}_N_TREES={len(clf.estimators_)}")
        lines.append(f"{name}_TREES=[")
        for ta in clf.estimators_:
            t=ta[0].tree_
            fi=[int(v) for v in t.feature]
            th=[round(float(v),6) for v in t.threshold]
            va=[round(float(t.value[i,0,0]),8) for i in range(t.node_count)]
            le=[int(v) for v in t.children_left]
            ri=[int(v) for v in t.children_right]
            lines.append(f"    ({fi},{th},{va},{le},{ri}),")
        lines.append("]"); lines.append("")
        exported.append(name)
        print(f"  Exported {name} AUC={auc:.3f} ({len(clf.estimators_)} trees)")
    lines.append(f"VALUE_FN_MODELS={exported!r}")
    lines.append("VALUE_FN_LOADED=True")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines)+"\n")
    sz=os.path.getsize(path)/1024
    print(f"  -> {path} ({sz:.1f} KB)")
    return len(exported)>0

# ── Load/save records ─────────────────────────────────────────────
def save_records(records, round_id):
    path=os.path.join(DATA_DIR,f"records_{round_id:04d}.json")
    with open(path,"w") as f: json.dump(records,f)
    print(f"  Saved {len(records)} records to {path}")

def load_all_records():
    all_r=[]
    for p in sorted(glob.glob(os.path.join(DATA_DIR,"records_*.json"))):
        try:
            with open(p) as f: all_r.extend(json.load(f))
        except: pass
    return all_r

# ── Main ──────────────────────────────────────────────────────────
if __name__=="__main__":
    print("="*60)
    print("  train_value_fn.py")
    print("="*60)

    # Find enemy agents
    enemy_files=sorted([f for f in glob.glob(os.path.join(ENEMY_DIR,"*.py"))
                        if not os.path.basename(f).startswith("__")])
    if not enemy_files:
        print(f"No agents in '{ENEMY_DIR}'"); sys.exit(1)
    print(f"Found {len(enemy_files)} enemy agent(s)")

    # Load agents
    my_fn=load_agent(MAIN_PY,force=True)
    enemy_agents=[]
    for ef in enemy_files:
        try:
            fn=load_agent(ef)
            enemy_agents.append((os.path.splitext(os.path.basename(ef))[0],fn))
            print(f"  Loaded enemy: {os.path.basename(ef)}")
        except Exception as e:
            print(f"  WARN: could not load {ef}: {e}")

    # Determine round id
    existing=glob.glob(os.path.join(DATA_DIR,"records_*.json"))
    round_id=len(existing)
    print(f"\nRound {round_id}  ({len(enemy_agents)*GAMES_PER_ENEMY} games total)")

    # Collect
    print("\nCollecting game data...")
    new_records=collect_data(my_fn,enemy_agents,round_id=round_id)
    save_records(new_records,round_id)

    # Load all accumulated records
    all_records=load_all_records()[-10000:]
    print(f"\nTotal accumulated samples: {len(all_records)}")

    # Train
    print("\nTraining value function...")
    models=train(all_records)

    # Export
    if models:
        print("\nExporting value_fn.py...")
        export(models)
        print("\nDone. Add value_fn.py to your submission zip.")
    else:
        print("\nNo models trained. Run more games and retry.")