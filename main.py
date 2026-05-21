# ===================================================================
# main.py
#
# Strategy: main_3 (TacticalQuadrantAgent) — proven working logic
# RL layer: GBC value function from value_fn.py improves target
#           scoring when trained weights are available.
#
# Submit as zip: main.py + value_fn.py
# Without value_fn.py runs as pure main_3 heuristic.
# ===================================================================

import math
import os
import sys

# ───────────────────────────────────────────────────────────────
# SECTION 1: VALUE FUNCTION (from value_fn.py if present)
# ───────────────────────────────────────────────────────────────
GLOBAL_TREES=[]; GLOBAL_INIT=0.0; GLOBAL_N_TREES=0
EARLY_TREES =[];  EARLY_INIT=0.0;  EARLY_N_TREES=0
LATE_TREES  =[];   LATE_INIT=0.0;   LATE_N_TREES=0
VALUE_FN_LOADED=False

try:
    from value_fn import GLOBAL_TREES,GLOBAL_INIT,GLOBAL_N_TREES,VALUE_FN_LOADED
    try: from value_fn import EARLY_TREES,EARLY_INIT,EARLY_N_TREES
    except ImportError: pass
    try: from value_fn import LATE_TREES,LATE_INIT,LATE_N_TREES
    except ImportError: pass
except ImportError:
    pass


def _walk(trees, init, feats):
    z = init
    for fi,th,va,le,ri in trees:
        n = 0
        while fi[n] >= 0:
            n = le[n] if feats[fi[n]] <= th[n] else ri[n]
        z += va[n]
    return z

def _sig(z):
    return 1.0/(1.0+math.exp(-z)) if z >= 0 else math.exp(z)/(1.0+math.exp(z))

def value_score(feats, turn=250):
    if not VALUE_FN_LOADED: return None   # None = fall back to heuristic
    if turn <= 100 and EARLY_N_TREES > 0:
        return _sig(_walk(EARLY_TREES, EARLY_INIT, feats))
    if turn >= 300 and LATE_N_TREES > 0:
        return _sig(_walk(LATE_TREES, LATE_INIT, feats))
    if GLOBAL_N_TREES > 0:
        return _sig(_walk(GLOBAL_TREES, GLOBAL_INIT, feats))
    return None


# ───────────────────────────────────────────────────────────────
# SECTION 2: NAMEDTUPLES
# Works with or without kaggle_environments installed.
# ───────────────────────────────────────────────────────────────
try:
    import kaggle_environments.envs.orbit_wars.orbit_wars as _ow
    Planet = _ow.Planet
    Fleet  = _ow.Fleet
except ImportError:
    from collections import namedtuple
    Planet = namedtuple("Planet",
        ["id","owner","x","y","radius","ships","production"])
    Fleet  = namedtuple("Fleet",
        ["id","owner","x","y","angle","from_planet_id","ships"])


# ───────────────────────────────────────────────────────────────
# SECTION 3: CONSTANTS  (main_3 originals, kept exactly)
# ───────────────────────────────────────────────────────────────
MAX_SPEED                  = 6.0
MIN_SHIPS_MINE_ATTACK      = 5
MIN_SHIPS_TARGET_COOP_ATTACK = 20
COOP_PLANET_CAP            = 8
COLLIDE_TICK_THOLD         = 1

FORMULA_DIST               = 100
FORMULA_PROD_MULT          = 15
FORMULA_ENEMY_BONUS_MULT   = 10
FORMULA_TOTAL_SHIPS_PERCENT= 0.7


# ───────────────────────────────────────────────────────────────
# SECTION 4: MAIN_3 STRATEGY — every function kept verbatim
# ───────────────────────────────────────────────────────────────

def get_custom_score(m, t):
    d = math.sqrt((m.x-t.x)**2 + (m.y-t.y)**2)
    min_ships = t.ships + 1
    spd = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,min_ships))/math.log(1000))**1.5
    eta = d / spd
    enemy_produced = eta*t.production if t.owner != -1 else 0
    enemy_bonus    = t.production     if t.owner != -1 else 0
    total_ships    = min_ships + enemy_produced
    return ((FORMULA_DIST-d)
            +(FORMULA_PROD_MULT*t.production)
            +(FORMULA_ENEMY_BONUS_MULT*enemy_bonus)
            -(FORMULA_TOTAL_SHIPS_PERCENT*total_ships)
            -(2*eta))


def get_planets_under_attack(mine, fleets, player, vel, moving_planets):
    mov_traj = {}
    under_attack = {}
    seen = set()
    fleets = [f for f in fleets if f.owner != player]
    for m in mine:
        if m.id in moving_planets:
            mov_traj[m.id] = get_planet_trajectories(m, vel)
    for f in fleets:
        spd = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,f.ships))/math.log(1000))**1.5
        px, py = f.x, f.y
        for tick in range(1, 61):
            nx = f.x + math.cos(f.angle)*spd*tick
            ny = f.y + math.sin(f.angle)*spd*tick
            for m in mine:
                mx,my = (mov_traj[m.id][tick-1] if m.id in moving_planets
                         else (m.x, m.y))
                if collides(px,py,nx,ny,mx,my,m.radius):
                    if (m.id,f.id) not in seen:
                        under_attack.setdefault(m.id,{"planet":m,"fleets":[]})
                        under_attack[m.id]["fleets"].append(
                            {"fleet":f,"arrive_tick":tick})
                        seen.add((m.id,f.id))
            px,py = nx,ny
    return under_attack


def refresh_local_obs(obs, player):
    raw_p  = obs.get("planets",[]) if isinstance(obs,dict) else getattr(obs,"planets",[])
    raw_f  = obs.get("fleets",[])  if isinstance(obs,dict) else getattr(obs,"fleets",[])
    planets= [Planet(*p) for p in raw_p]
    fleets = [Fleet(*f)  for f in raw_f]
    mine   = [p for p in planets if p.owner == player]
    targets= [p for p in planets if p.owner != player]
    return {"planets":planets,"mine":mine,"targets":targets,
            "player":player,"fleets":fleets}


def sun_collision(m, fleet_speed, angle, ticks=61):
    px,py = m.x, m.y
    for tick in range(1, ticks):
        x = m.x + math.cos(angle)*fleet_speed*tick
        y = m.y + math.sin(angle)*fleet_speed*tick
        if collides(px,py,x,y,50,50,10):
            return True
        px,py = x,y
    return False


def calculate_req_ships_moving(attacking_planets, t, base_ships, vel):
    required_ships = base_ships
    planet_trajectories = get_planet_trajectories(t, vel)
    for _ in range(3):
        remainder = required_ships
        max_tick  = 0
        for a_p in attacking_planets:
            p       = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)
            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
            if p_ships <= 0: continue
            spd = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,p_ships))/math.log(1000))**1.5
            found_tick = 0
            for tick,(tx,ty) in enumerate(planet_trajectories, start=1):
                d = math.sqrt((p.x-tx)**2+(p.y-ty)**2)
                if abs(math.floor(d/spd)-tick) <= 1:
                    found_tick = tick; break
            if found_tick > max_tick: max_tick = found_tick
            remainder -= p_ships
        new_req = base_ships + max_tick*t.production
        if new_req == required_ships: break
        required_ships = new_req
    return required_ships


def calculate_req_ships(attacking_planets, t, base_ships):
    required_ships = base_ships
    for _ in range(3):
        remainder = required_ships
        max_tick  = 0
        for a_p in attacking_planets:
            p       = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)
            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
            if p_ships <= 0: continue
            spd  = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,p_ships))/math.log(1000))**1.5
            d    = math.sqrt((p.x-t.x)**2+(p.y-t.y)**2)
            tick = math.floor(d/spd)
            if tick > max_tick: max_tick = tick
            remainder -= p_ships
        new_req = base_ships + max_tick*t.production
        if new_req == required_ships: break
        required_ships = new_req
    return required_ships


def calculate_angle(m, t):
    return math.atan2(t.y-m.y, t.x-m.x)


def collides(x1,y1,x2,y2,cx,cy,r):
    vx,vy = x2-x1, y2-y1
    tcx,tcy = cx-x1, cy-y1
    lsq = vx*vx+vy*vy
    if lsq == 0:
        return (x1-cx)**2+(y1-cy)**2 <= r*r
    t = max(0,min(1,(tcx*vx+tcy*vy)/lsq))
    dx = x1+t*vx-cx; dy = y1+t*vy-cy
    return dx*dx+dy*dy <= r*r


def get_closest_planets_to_target(mine, t):
    return sorted([(m, math.sqrt((m.x-t.x)**2+(m.y-t.y)**2))
                   for m in mine], key=lambda k:k[1])


def update_fleet_trajectories(fleets, fleet_trajectories):
    for ft in fleet_trajectories[:]:
        found = any(f.from_planet_id==ft["mine"].id and
                    abs(f.angle-ft["angle"])<1e-3
                    for f in fleets)
        if found:
            ft["arrive_tick"] = max(0, ft["arrive_tick"]-1)
        else:
            fleet_trajectories.remove(ft)


def update_reinforcement_trajectories(reinforcement_trajectories):
    for rt in reinforcement_trajectories[:]:
        rt["arrive_tick"] -= 1
        if rt["arrive_tick"] <= 0:
            reinforcement_trajectories.remove(rt)


def get_planet_trajectories(p, vel):
    angle = math.atan2(p.y-50, p.x-50)
    r     = math.sqrt((p.x-50)**2+(p.y-50)**2)
    traj  = []
    for tick in range(1, 61):
        a = angle + vel*tick
        traj.append((50+r*math.cos(a), 50+r*math.sin(a)))
    return traj


def fill_moving_planets(obs, moving_planets):
    raw_p  = obs.get("planets",[]) if isinstance(obs,dict) else getattr(obs,"planets",[])
    raw_i  = obs.get("initial_planets",[]) if isinstance(obs,dict) else getattr(obs,"initial_planets",[])
    ibi    = {i[0]: Planet(*i) for i in raw_i}
    for p_raw in raw_p:
        p  = Planet(*p_raw)
        ip = ibi.get(p.id)
        if ip and (p.x,p.y) != (ip.x,ip.y):
            if p.id not in moving_planets:
                moving_planets.append(p.id)


def get_reinforcement_plans(mine, under_attack, reinforcement_trajectories):
    plans = {}
    for p in mine:
        if p.id not in under_attack: continue
        atk_fleets = sorted(under_attack[p.id]["fleets"],
                            key=lambda a:a["arrive_tick"])
        inc_reinf  = sorted([r for r in reinforcement_trajectories
                              if r["target"].id==p.id],
                             key=lambda r:r["arrive_tick"])
        avail   = p.ships
        prev_t  = 0
        r_idx   = 0
        for att in atk_fleets:
            at = att["arrive_tick"]
            avail += (at-prev_t)*p.production
            while r_idx<len(inc_reinf) and inc_reinf[r_idx]["arrive_tick"]<=at:
                avail += inc_reinf[r_idx]["total_ships"]; r_idx+=1
            avail -= att["fleet"].ships
            prev_t = at
            if avail < 0:
                plans[p] = {"ships_needed": max(MIN_SHIPS_MINE_ATTACK,abs(avail)),
                            "needed_by_tick": at}
                break
    return plans


def find_angle_to_moving_planet(p, t, ships, vel):
    spd  = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,ships))/math.log(1000))**1.5
    traj = get_planet_trajectories(t, vel)
    for tick,(tx,ty) in enumerate(traj, start=1):
        dx,dy = tx-p.x, ty-p.y
        dist  = math.sqrt(dx*dx+dy*dy) - p.radius
        if abs(spd*tick - dist) > t.radius: continue
        angle = math.atan2(dy,dx)
        if sun_collision(p, spd, angle): return None,None
        return angle, tick
    return None, None


# ───────────────────────────────────────────────────────────────
# SECTION 5: RL TARGET SCORING
# When value function is loaded, re-ranks the top-K candidates
# by simulating forward and picking highest P(win).
# Falls back to heuristic score when not loaded.
# ───────────────────────────────────────────────────────────────

def _extract_features(planets_raw, player, turn, n_hint=2):
    ms=mp=mpr=mc=ts=tpr=0.0; ed={}
    for p in planets_raw:
        sh=max(0.0,float(p[5])); ts+=sh
        ow=p[1]
        if ow==-1: continue
        pr=float(p[6]); tpr+=pr
        cn=max(0.0,60.0-math.hypot(float(p[2])-50.0,float(p[3])-50.0))
        if ow==player: ms+=sh; mp+=1; mpr+=pr; mc+=cn
        else:
            e=ed.setdefault(ow,{'s':0.,'p':0,'pr':0.,'c':0.})
            e['s']+=sh; e['p']+=1; e['pr']+=pr; e['c']+=cn
    bes=bep=bepr=bec=0.0
    if ed:
        be=max(ed.values(),key=lambda e:e['s']+100*e['p'])
        bes=be['s']; bep=float(be['p']); bepr=be['pr']
        bec=be['c']/max(1,be['p'])
    tp=max(1,len(planets_raw)); sf=lambda a,b:a/b if b>1e-9 else 0.0
    np2=max(2,1+len(ed),n_hint)
    return [turn/500.0,np2/4.0,sf(mp,tp),sf(bep,tp),sf(ms,ts),sf(bes,ts),
            sf(mpr,tpr),sf(bepr,tpr),(mc/max(1,mp))/60.0,bec/60.0,
            sf(ms-bes,ts+1),sf(mp-bep,tp),sf(mpr-bepr,tpr+1),
            1.0 if np2==2 else 0.0,1.0 if np2==4 else 0.0,sf(ms,bes+1)]


def _sim_move(planets_raw, raw_fleets, player, src_id, angle, ships,
              lookahead=20):
    """Lightweight forward sim — returns feature vector after lookahead turns."""
    state = [{'id':p[0],'owner':p[1],'ships':float(p[5]),
              'production':float(p[6]),'x':float(p[2]),'y':float(p[3]),
              'radius':float(p[4])} for p in planets_raw]
    i2i   = {s['id']:i for i,s in enumerate(state)}
    arr   = {}
    ca,sa = math.cos(angle),math.sin(angle)

    # Schedule existing enemy fleets
    for f in raw_fleets:
        if f[1]==player: continue
        fc,fs2,fa2,fsh = f[2],f[3],f[4],f[6]
        spd = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,fsh))/math.log(1000))**1.5
        fca,fsa = math.cos(fa2),math.sin(fa2)
        bp=None; ba=float('inf')
        for p in planets_raw:
            dx=p[2]-fc; dy=p[3]-fs2
            al=dx*fca+dy*fsa
            if al<=0: continue
            if abs(dx*fsa-dy*fca)<=p[4]+0.5 and al<ba: ba=al; bp=p
        if bp:
            eta=max(1,int(math.ceil(ba/spd)))
            if eta<=lookahead:
                idx=i2i.get(bp[0])
                if idx is not None:
                    arr.setdefault(eta,[]).append((idx,f[1],float(fsh)))

    # Schedule candidate move
    si = i2i.get(src_id)
    if si is not None and ships>0:
        state[si]['ships']=max(0.0,state[si]['ships']-ships)
        sx=state[si]['x']+ca*(state[si]['radius']+0.1)
        sy=state[si]['y']+sa*(state[si]['radius']+0.1)
        spd=1.0+(MAX_SPEED-1.0)*(math.log(max(1,ships))/math.log(1000))**1.5
        bp=None; ba=float('inf')
        for p in planets_raw:
            if p[0]==src_id: continue
            dx=p[2]-sx; dy=p[3]-sy
            al=dx*ca+dy*sa
            if al<=0: continue
            if abs(dx*sa-dy*ca)<=p[4]+0.5 and al<ba: ba=al; bp=p
        if bp:
            eta=max(1,int(math.ceil(ba/spd)))
            if eta<=lookahead:
                idx=i2i.get(bp[0])
                if idx is not None:
                    arr.setdefault(eta,[]).append((idx,player,float(ships)))

    def _combat(ps, arriving):
        by={ps['owner']:ps['ships']}
        for ow,sh in arriving: by[ow]=by.get(ow,0.0)+sh
        sf=sorted(by.items(),key=lambda x:-x[1])
        if len(sf)==1: ps['owner']=sf[0][0]; ps['ships']=sf[0][1]; return
        if abs(sf[0][1]-sf[1][1])<0.5: ps['ships']=0.0; return
        ps['owner']=sf[0][0]; ps['ships']=sf[0][1]-sf[1][1]

    for tick in range(1,lookahead+1):
        for s in state:
            if s['owner']!=-1: s['ships']+=s['production']
        if tick in arr:
            pa={}
            for idx,ow,sh in arr[tick]: pa.setdefault(idx,[]).append((ow,sh))
            for idx,a in pa.items(): _combat(state[idx],a)

    # Convert back to raw tuple format for feature extraction
    raw = [[s['id'],s['owner'],s['x'],s['y'],s['radius'],s['ships'],s['production']]
           for s in state]
    return raw


def rl_pick_target(src, candidates, planets_raw, raw_fleets, player, turn,
                   top_k=5):
    """
    Re-rank top_k heuristic candidates using forward sim + value function.
    Returns the best (target, heuristic_score) pair.
    Falls back to best heuristic if value function not loaded.
    """
    if not VALUE_FN_LOADED or not candidates:
        return candidates[0] if candidates else None

    # Only re-rank the top_k to keep it fast
    ranked = candidates[:top_k]
    best   = None; best_pw = -1.0

    for tgt, h_score in ranked:
        angle = calculate_angle(src, tgt)
        ships = max(MIN_SHIPS_MINE_ATTACK, int(tgt.ships)+1)
        # Don't sim if sun blocks path
        spd  = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,ships))/math.log(1000))**1.5
        if sun_collision(src, spd, angle): continue
        sim  = _sim_move(planets_raw, raw_fleets, player,
                         src.id, angle, ships)
        feats= _extract_features(sim, player, turn)
        pw   = value_score(feats, turn)
        if pw is None: return candidates[0]  # no model, fall back
        if pw > best_pw: best_pw=pw; best=(tgt, h_score)

    return best if best else (candidates[0] if candidates else None)


# ───────────────────────────────────────────────────────────────
# SECTION 6: AGENT STATE (per-game, reset on new game)
# ───────────────────────────────────────────────────────────────

class _AgentState:
    def __init__(self):
        self.fleet_trajectories        = []
        self.reinforcement_trajectories= []
        self.moving_planets            = []
        self.steps                     = 0
        self._game_sig                 = None
        self._last_obs_step            = -1

_state = _AgentState()


def _detect_new_game(obs):
    """Return True if this observation belongs to a new game."""
    raw_i  = (obs.get("initial_planets",[]) if isinstance(obs,dict)
              else getattr(obs,"initial_planets",[]))
    player = (obs.get("player",0) if isinstance(obs,dict)
              else getattr(obs,"player",0))
    obs_step=(obs.get("step",0) if isinstance(obs,dict)
              else getattr(obs,"step",0)) or 0
    try:
        sig_tail = tuple((int(p[0]),int(p[5]),int(p[6]))
                         for p in (raw_i or [])[:4])
    except Exception:
        sig_tail = ()
    sig = (player, sig_tail)
    if (sig != _state._game_sig or obs_step == 0
            or obs_step < _state._last_obs_step):
        return True, sig, obs_step
    return False, sig, obs_step


# ───────────────────────────────────────────────────────────────
# SECTION 7: CORE AGENT LOGIC  (main_3 strategy, RL-enhanced)
# ───────────────────────────────────────────────────────────────

def _agent_impl(obs):
    s = _state   # shorthand

    if s.steps < 2:
        s.steps += 1
        return []
    if s.steps == 2:
        fill_moving_planets(obs, s.moving_planets)
        s.steps = 3

    vel    = (obs.get("angular_velocity",0.0) if isinstance(obs,dict)
              else getattr(obs,"angular_velocity",0.0))
    player = (obs.get("player",0) if isinstance(obs,dict)
              else getattr(obs,"player",0))
    turn   = s.steps   # use our own step counter as turn number

    lobs   = refresh_local_obs(obs, player)
    mine   = lobs["mine"]
    targets= lobs["targets"]
    fleets = lobs["fleets"]
    planets= lobs["planets"]

    # Get raw versions for RL sim
    raw_p  = (obs.get("planets",[]) if isinstance(obs,dict)
              else getattr(obs,"planets",[]))
    raw_f_raw=(obs.get("fleets",[]) if isinstance(obs,dict)
               else getattr(obs,"fleets",[]))
    raw_fleets=[Fleet(*f) for f in raw_f_raw]

    comet_ids = set(obs.get("comet_planet_ids",[]) if isinstance(obs,dict)
                    else getattr(obs,"comet_planet_ids",[]))

    update_fleet_trajectories(fleets, s.fleet_trajectories)
    update_reinforcement_trajectories(s.reinforcement_trajectories)

    under_attack = get_planets_under_attack(mine, fleets, player,
                                            vel, s.moving_planets)
    exhausted    = set()
    moves        = []

    if not targets:
        return []

    # ── DEFENSE: send reinforcements ─────────────────────────────
    reinf_plans = get_reinforcement_plans(mine, under_attack,
                                          s.reinforcement_trajectories)
    for p, plan in reinf_plans.items():
        already = any(r["target"].id==p.id and r["arrive_tick"]>=0
                      for r in s.reinforcement_trajectories)
        if already: continue
        ships_needed  = plan["ships_needed"]
        needed_by_tick= plan["needed_by_tick"]
        nearest = get_closest_planets_to_target(mine, p)
        for p_np, _ in nearest:
            if p_np.id==p.id or p_np.id in exhausted: continue
            avail = p_np.ships
            avail -= sum(r["total_ships"] for r in s.reinforcement_trajectories
                         if r["mine"].id==p_np.id)
            if p_np.id in under_attack:
                avail = max(0, avail - sum(a["fleet"].ships
                            for a in under_attack[p_np.id]["fleets"]))
            sent = max(MIN_SHIPS_MINE_ATTACK, ships_needed)
            if avail < sent: continue
            if p.id not in s.moving_planets:
                angle_np = math.atan2(p.y-p_np.y, p.x-p_np.x)
                spd  = 1.0+(MAX_SPEED-1.0)*(math.log(sent)/math.log(1000))**1.5
                d    = math.sqrt((p.x-p_np.x)**2+(p.y-p_np.y)**2)
                if math.floor(d/spd) > needed_by_tick: continue
            else:
                angle_np, at = find_angle_to_moving_planet(
                    p_np, p, sent, vel)
                if angle_np is None: continue
            moves.append([p_np.id, angle_np, sent])
            exhausted.add(p_np.id)
            d2 = math.sqrt((p.x-p_np.x)**2+(p.y-p_np.y)**2)
            spd2= 1.0+(MAX_SPEED-1.0)*(math.log(sent)/math.log(1000))**1.5
            s.reinforcement_trajectories.append({
                "mine":p_np,"target":p,"angle":angle_np,
                "total_ships":sent,
                "arrive_tick":math.floor(d2/spd2)})
            break

    # ── ATTACK: each planet picks its best target ─────────────────
    owned_count = len(mine)
    total_count = len(planets)

    for m in sorted(mine, key=lambda p: p.ships, reverse=True):
        if m.id in exhausted: continue
        if m.ships < MIN_SHIPS_MINE_ATTACK: continue

        # Score all non-comet targets
        scored = []
        for t in targets:
            if t.id in comet_ids: continue
            scored.append((t, get_custom_score(m, t)))
        scored.sort(key=lambda x: x[1], reverse=True)
        if not scored: continue

        # ── RL target re-ranking ──────────────────────────────────
        # When value function is loaded, re-rank top-5 by P(win).
        # Otherwise use heuristic order unchanged.
        best_pair = rl_pick_target(m, scored, raw_p, raw_fleets,
                                   player, turn, top_k=5)
        if best_pair is None: continue

        # Try best RL/heuristic target first, fall back through top-3
        top3 = [best_pair] + [x for x in scored[:3] if x[0].id != best_pair[0].id]

        for t, _ in top3:
            m_avail = m.ships
            if m.id in under_attack:
                m_avail = max(0, m_avail - sum(
                    a["fleet"].ships for a in under_attack[m.id]["fleets"]))
            if m_avail < MIN_SHIPS_MINE_ATTACK: continue

            # Ships already en route to this target
            en_route = sum(ft["total_ships"] for ft in s.fleet_trajectories
                           if ft["target"].id==t.id)
            needed_now = t.ships+1 + (3*t.production if t.owner!=-1 else 0)

            if owned_count < total_count*0.75:
                if en_route >= needed_now: continue

            base_ships = max(MIN_SHIPS_MINE_ATTACK, needed_now-en_route)

            # ── Single planet attack ──────────────────────────────
            if m_avail >= base_ships:
                if t.id in s.moving_planets:
                    total = base_ships
                    for _ in range(3):
                        angle, arrive = find_angle_to_moving_planet(
                            m, t, total, vel)
                        if angle is None: break
                        new_total = (base_ships+arrive*t.production
                                     if t.owner!=-1 else base_ships)
                        if new_total > m_avail: angle=None; break
                        if new_total == total: break
                        total = new_total
                    if angle is None: continue
                    spd2 = 1.0+(MAX_SPEED-1.0)*(math.log(max(1,total))/math.log(1000))**1.5
                    if sun_collision(m, spd2, angle): continue
                    moves.append([m.id, angle, total])
                    exhausted.add(m.id)
                    s.fleet_trajectories.append({
                        "mine":m,"target":t,"angle":angle,
                        "total_ships":total,"arrive_tick":arrive})
                    break
                else:  # static target
                    angle = calculate_angle(m, t)
                    total = base_ships
                    d3 = math.sqrt((t.x-m.x)**2+(t.y-m.y)**2)
                    arrive = 0
                    if t.owner != -1:
                        for _ in range(3):
                            spd3=1.0+(MAX_SPEED-1.0)*(math.log(max(1,total))/math.log(1000))**1.5
                            arrive=math.floor(d3/spd3)
                            new_total=base_ships+arrive*t.production
                            if new_total>m_avail: angle=None; break
                            if new_total==total: break
                            total=new_total
                    if angle is None: continue
                    spd4=1.0+(MAX_SPEED-1.0)*(math.log(max(1,total))/math.log(1000))**1.5
                    if sun_collision(m, spd4, angle): continue
                    if arrive==0:
                        arrive=math.floor(d3/spd4)
                    moves.append([m.id, angle, total])
                    exhausted.add(m.id)
                    s.fleet_trajectories.append({
                        "mine":m,"target":t,"angle":angle,
                        "total_ships":total,"arrive_tick":arrive})
                    break

            # ── Cooperative attack ────────────────────────────────
            elif (m_avail < base_ships
                  and len(mine) > 1
                  and t.ships >= MIN_SHIPS_TARGET_COOP_ATTACK):
                accum = m_avail
                attacking = [{"planet":m,"ships":m_avail}]
                near = get_closest_planets_to_target(mine, t)
                safe_near = []
                for p2,d4 in near:
                    if p2.id==m.id or p2.id in exhausted: continue
                    av2 = p2.ships
                    if p2.id in under_attack:
                        av2=max(0,av2-sum(a["fleet"].ships
                                for a in under_attack[p2.id]["fleets"]))
                    if av2 < MIN_SHIPS_MINE_ATTACK: continue
                    safe_near.append((p2,d4,av2))
                coop_sent = False
                for p2,d4,av2 in safe_near:
                    if coop_sent: break
                    attacking.append({"planet":p2,"ships":av2})
                    accum += av2
                    if len(attacking) > COOP_PLANET_CAP: break
                    if accum < base_ships: continue
                    if t.id not in s.moving_planets:
                        if t.owner == -1:
                            rem = base_ships; planned = []
                            for a_p in attacking:
                                pp=a_p["planet"]
                                ps=min(a_p["ships"],rem)
                                if ps>0: ps=min(a_p["ships"],max(ps,MIN_SHIPS_MINE_ATTACK))
                                if ps<=0: continue
                                ang2=calculate_angle(pp,t)
                                sp2=1.0+(MAX_SPEED-1.0)*(math.log(ps)/math.log(1000))**1.5
                                at2=math.floor(math.sqrt((pp.x-t.x)**2+(pp.y-t.y)**2)/sp2)
                                if sun_collision(pp,sp2,ang2): break
                                rem-=ps; planned.append([pp,ang2,ps,at2])
                            if rem>0: continue
                        else:
                            req=calculate_req_ships(attacking,t,base_ships)
                            if accum<req: continue
                            rem=req; planned=[]
                            for a_p in attacking:
                                pp=a_p["planet"]
                                ps=min(a_p["ships"],rem)
                                if ps>0: ps=min(a_p["ships"],max(ps,MIN_SHIPS_MINE_ATTACK))
                                if ps<=0: continue
                                ang2=calculate_angle(pp,t)
                                sp2=1.0+(MAX_SPEED-1.0)*(math.log(ps)/math.log(1000))**1.5
                                at2=math.floor(math.sqrt((pp.x-t.x)**2+(pp.y-t.y)**2)/sp2)
                                if sun_collision(pp,sp2,ang2): continue
                                rem-=ps; planned.append([pp,ang2,ps,at2])
                            if rem>0: continue
                        for mv in planned:
                            s.fleet_trajectories.append({
                                "mine":mv[0],"target":t,"angle":mv[1],
                                "total_ships":mv[2],"arrive_tick":mv[3]})
                            exhausted.add(mv[0].id)
                            moves.append([mv[0].id,mv[1],mv[2]])
                        coop_sent=True; break
                    else:
                        if t.owner==-1:
                            rem=base_ships; planned=[]
                            for a_p in attacking:
                                pp=a_p["planet"]
                                ps=min(a_p["ships"],rem)
                                if ps>0: ps=min(a_p["ships"],max(ps,MIN_SHIPS_MINE_ATTACK))
                                if ps<=0: continue
                                ang2,at2=find_angle_to_moving_planet(pp,t,ps,vel)
                                if ang2 is None: continue
                                planned.append([pp,ang2,ps,at2]); rem-=ps
                            if rem>0: continue
                        else:
                            req=calculate_req_ships_moving(attacking,t,base_ships,vel)
                            if accum<req: continue
                            rem=req; planned=[]
                            for a_p in attacking:
                                pp=a_p["planet"]
                                ps=min(a_p["ships"],rem)
                                if ps>0: ps=min(a_p["ships"],max(ps,MIN_SHIPS_MINE_ATTACK))
                                if ps<=0: continue
                                ang2,at2=find_angle_to_moving_planet(pp,t,ps,vel)
                                if ang2 is None: continue
                                rem-=ps; planned.append([pp,ang2,ps,at2])
                            if rem>0: continue
                        for mv in planned:
                            s.fleet_trajectories.append({
                                "mine":mv[0],"target":t,"angle":mv[1],
                                "total_ships":mv[2],"arrive_tick":mv[3]})
                            exhausted.add(mv[0].id)
                            moves.append([mv[0].id,mv[1],mv[2]])
                        coop_sent=True; break

    s.steps += 1
    return moves


# ───────────────────────────────────────────────────────────────
# SECTION 8: SAFE WRAPPER + ENTRY POINT
# ───────────────────────────────────────────────────────────────

def _sanitize(moves):
    out = []
    for mv in (moves or []):
        try:
            pid   = int(mv[0])
            angle = float(mv[1])
            ships = int(mv[2])
            if ships > 0:
                out.append([pid, angle, ships])
        except Exception:
            pass
    return out


def agent(obs, config=None):
    global _state
    try:
        is_new, sig, obs_step = _detect_new_game(obs)
        if is_new:
            _state = _AgentState()
            _state._game_sig      = sig
            _state._last_obs_step = obs_step
        else:
            _state._last_obs_step = obs_step
        return _sanitize(_agent_impl(obs))
    except Exception as e:
        return []