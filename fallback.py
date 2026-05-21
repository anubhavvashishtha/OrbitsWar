# ===================================================================
# main.py  —  ClusterOrbitAgent with RL Value Function
#
# Submit as a zip: main.py + value_fn.py
# If value_fn.py is absent, the agent falls back to pure heuristics.
#
# Decision points where RL improves over heuristics:
#   1. target_choice  — which planet to attack
#   2. defend_or_beam — should I pause beam to defend?
#   3. which_defender — which cluster planet best defends?
#   4. fleet_size     — how many ships to send
#   5. grow_routing   — where to route peacetime ships
# ===================================================================

import math

# ───────────────────────────────────────────────────────────────
# SECTION 1: LOAD VALUE FUNCTION (from value_fn.py if present)
# ───────────────────────────────────────────────────────────────

# These get overwritten if value_fn.py is present
GLOBAL_TREES = [];  GLOBAL_INIT = 0.0;  GLOBAL_N_TREES = 0
EARLY_TREES  = [];  EARLY_INIT  = 0.0;  EARLY_N_TREES  = 0
LATE_TREES   = [];  LATE_INIT   = 0.0;  LATE_N_TREES   = 0
ATTACK_TREES = [];  ATTACK_INIT = 0.0;  ATTACK_N_TREES = 0
DEFEND_TREES = [];  DEFEND_INIT = 0.0;  DEFEND_N_TREES = 0
VALUE_FN_LOADED = False

try:
    from value_fn import (
        GLOBAL_TREES, GLOBAL_INIT, GLOBAL_N_TREES,
        VALUE_FN_LOADED,
    )
    # Optional specialised models — ignore if absent
    try:
        from value_fn import EARLY_TREES, EARLY_INIT, EARLY_N_TREES
    except ImportError:
        pass
    try:
        from value_fn import LATE_TREES, LATE_INIT, LATE_N_TREES
    except ImportError:
        pass
    try:
        from value_fn import ATTACK_TREES, ATTACK_INIT, ATTACK_N_TREES
    except ImportError:
        pass
    try:
        from value_fn import DEFEND_TREES, DEFEND_INIT, DEFEND_N_TREES
    except ImportError:
        pass
except ImportError:
    pass  # Pure heuristic mode — still works fine


def _walk_trees(trees, init, features):
    """Walk GBC trees and return raw logit (no sigmoid)."""
    z = init
    for feat_idx, thresh, vals, left, right in trees:
        node = 0
        while feat_idx[node] >= 0:
            if features[feat_idx[node]] <= thresh[node]:
                node = left[node]
            else:
                node = right[node]
        z += vals[node]
    return z


def _sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def value_score(features, current_turn=250):
    """
    Returns P(win) from 0.0 to 1.0.
    Picks EARLY/LATE specialised model if available, else GLOBAL.
    Falls back to 0.5 if no model loaded.
    """
    if not VALUE_FN_LOADED:
        return 0.5

    if current_turn <= 100 and EARLY_N_TREES > 0:
        return _sigmoid(_walk_trees(EARLY_TREES, EARLY_INIT, features))
    if current_turn >= 300 and LATE_N_TREES > 0:
        return _sigmoid(_walk_trees(LATE_TREES, LATE_INIT, features))
    if GLOBAL_N_TREES > 0:
        return _sigmoid(_walk_trees(GLOBAL_TREES, GLOBAL_INIT, features))
    return 0.5


def attack_score(features):
    """Specialised attack model, falls back to global."""
    if ATTACK_N_TREES > 0:
        return _sigmoid(_walk_trees(ATTACK_TREES, ATTACK_INIT, features))
    return value_score(features)


def defend_score(features):
    """Specialised defend model, falls back to global."""
    if DEFEND_N_TREES > 0:
        return _sigmoid(_walk_trees(DEFEND_TREES, DEFEND_INIT, features))
    return value_score(features)


# ───────────────────────────────────────────────────────────────
# SECTION 2: GAME CONSTANTS
# ───────────────────────────────────────────────────────────────

CENTER_X = CENTER_Y = 50.0
SUN_R    = 10.0
MAX_SPEED = 6.0
SUN_SAFETY = 1.5
ROTATION_LIMIT = 50.0
LAUNCH_CLEARANCE = 0.1
INTERCEPT_TOLERANCE = 1
HORIZON = 110
ROUTE_SEARCH_HORIZON = 60

CLUSTER_BASE_RADIUS  = 10.0
PHASE_LENGTH         = 100
ATTACK_PHASE_CUTOFF  = 300

TIER1_MIN_PRODUCTION = 3
TIER1_MAX_SHIPS      = 9
TIER2_MIN_SHIPS      = 11
TIER2_MAX_SHIPS      = 25

DEFENSE_RESERVE_PCT      = 0.30
DEFENSE_SEND_RESERVE_PCT = 0.10
CROSS_CLUSTER_HELP_RATIO = 1.2
POST_CAPTURE_LOCKOUT     = 7
BUFF_WAVE_MIN  = 6
BUFF_WAVE_MAX  = 10
SMALL_PLANET_SHIP_CAP = 50

# RL thresholds — how confident the value function must be to
# override heuristic defaults
VF_ATTACK_MIN_CONFIDENCE  = 0.52   # switch from heuristic target if VF says > 52%
VF_DEFEND_MIN_CONFIDENCE  = 0.55   # pause beam to defend if VF says defending is > 55%
VF_FLEET_UPSIZE_THRESHOLD = 0.04   # send more ships if it improves P(win) by > 4pp
VF_SIM_LOOKAHEAD          = 20     # forward-sim turns for each candidate evaluation

# ───────────────────────────────────────────────────────────────
# SECTION 3: NAMEDTUPLES (avoid importing from kaggle_environments
# so the file works standalone in tests)
# ───────────────────────────────────────────────────────────────

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import (
        Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT
    )
    CENTER_X = CENTER_Y = float(CENTER)
except ImportError:
    from collections import namedtuple
    Planet = namedtuple("Planet",
        ["id","owner","x","y","radius","ships","production"])
    Fleet  = namedtuple("Fleet",
        ["id","owner","x","y","angle","from_planet_id","ships"])


# ───────────────────────────────────────────────────────────────
# SECTION 4: UTILITY / GEOMETRY  (preserved from original)
# ───────────────────────────────────────────────────────────────

def dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def is_static_planet(planet, initial_by_id):
    init = initial_by_id.get(planet.id, planet)
    return dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(max(1,ships)) / math.log(1000.0)))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio ** 1.5)


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    lsq = dx*dx + dy*dy
    if lsq <= 1e-9:
        return dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px-x1)*dx + (py-y1)*dy) / lsq))
    return dist(px, py, x1 + t*dx, y1 + t*dy)


def segment_hits_sun(x1, y1, x2, y2, safety=SUN_SAFETY):
    return point_to_segment_distance(
        CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def launch_point(sx, sy, sr, angle):
    c = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle)*c, sy + math.sin(angle)*c


def actual_path_geometry(sx, sy, sr, tx, ty, tr):
    angle = math.atan2(ty-sy, tx-sx)
    lx, ly = launch_point(sx, sy, sr, angle)
    hit_dist = max(0.0, dist(sx,sy,tx,ty) - (sr+LAUNCH_CLEARANCE) - tr)
    return (angle, lx, ly,
            lx + math.cos(angle)*hit_dist,
            ly + math.sin(angle)*hit_dist,
            hit_dist)


def safe_angle_and_distance(sx, sy, sr, tx, ty, tr):
    angle, lx, ly, ex, ey, hit_dist = actual_path_geometry(sx,sy,sr,tx,ty,tr)
    if segment_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_dist


def predict_planet_position(planet, initial_by_id, angular_velocity, turns):
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return CENTER_X + r*math.cos(new_ang), CENTER_Y + r*math.sin(new_ang)


def estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1,ships)))))
    return angle, turns


def search_safe_intercept(src, target, ships, initial_by_id, ang_vel):
    best, best_score = None, None
    for ct in range(1, min(HORIZON, ROUTE_SEARCH_HORIZON)+1):
        pos = predict_planet_position(target, initial_by_id, ang_vel, ct)
        est = estimate_arrival(src.x, src.y, src.radius,
                               pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - ct) > INTERCEPT_TOLERANCE:
            continue
        actual_turns = max(turns, ct)
        actual_pos = predict_planet_position(
            target, initial_by_id, ang_vel, actual_turns)
        confirm = estimate_arrival(src.x, src.y, src.radius,
                                   actual_pos[0], actual_pos[1],
                                   target.radius, ships)
        if confirm is None:
            continue
        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue
        score = (delta, confirm[1], ct)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])
    return best


def aim_with_prediction(src, target, ships, initial_by_id, ang_vel):
    est = estimate_arrival(src.x, src.y, src.radius,
                           target.x, target.y, target.radius, ships)
    target_is_static = is_static_planet(target, initial_by_id)
    if est is None:
        return (None if target_is_static
                else search_safe_intercept(src, target, ships, initial_by_id, ang_vel))
    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        ntx, nty = predict_planet_position(target, initial_by_id, ang_vel, turns)
        next_est = estimate_arrival(src.x, src.y, src.radius,
                                    ntx, nty, target.radius, ships)
        if next_est is None:
            return (None if target_is_static
                    else search_safe_intercept(src, target, ships, initial_by_id, ang_vel))
        if (abs(ntx-tx) < 0.3 and abs(nty-ty) < 0.3
                and abs(next_est[1]-turns) <= INTERCEPT_TOLERANCE):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est
    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty,
                                 target.radius, ships)
    if final_est is None:
        return search_safe_intercept(src, target, ships, initial_by_id, ang_vel)
    return final_est[0], final_est[1], tx, ty


# ───────────────────────────────────────────────────────────────
# SECTION 5: FORWARD SIMULATOR
# Used by the value function to score candidate moves.
# Lightweight — no Kaggle env overhead.
# ───────────────────────────────────────────────────────────────

def _resolve_combat(planet_state, arriving):
    by_owner = {planet_state['owner']: planet_state['ships']}
    for owner, ships in arriving:
        by_owner[owner] = by_owner.get(owner, 0.0) + ships
    if not by_owner:
        return
    sorted_f = sorted(by_owner.items(), key=lambda x: -x[1])
    if len(sorted_f) == 1:
        planet_state['owner'] = sorted_f[0][0]
        planet_state['ships'] = sorted_f[0][1]
        return
    top_owner, top = sorted_f[0]
    second = sorted_f[1][1]
    if abs(top - second) < 0.5:
        planet_state['ships'] = 0.0
        return
    planet_state['owner'] = top_owner
    planet_state['ships'] = top - second


def _ray_first_planet(fx, fy, fangle, fships, planets, exclude_pid=None):
    cos_a = math.cos(fangle)
    sin_a = math.sin(fangle)
    speed = fleet_speed(fships)
    best_p, best_along = None, float('inf')
    for p in planets:
        if p.id == exclude_pid:
            continue
        dx = p.x - fx; dy = p.y - fy
        along = dx*cos_a + dy*sin_a
        if along <= 0:
            continue
        perp = abs(dx*sin_a - dy*cos_a)
        if perp <= p.radius + 0.5 and along < best_along:
            hx = fx + cos_a*along; hy = fy + sin_a*along
            if point_to_segment_distance(
                    CENTER_X, CENTER_Y, fx, fy, hx, hy) >= SUN_R:
                best_along = along
                best_p = p
    if best_p:
        eta = max(1, int(math.ceil(best_along / speed)))
        return best_p, eta
    return None, None


def simulate_forward(planets, raw_fleets, my_player,
                     src_id, move_angle, move_ships,
                     lookahead=VF_SIM_LOOKAHEAD):
    """
    Simulate `lookahead` turns given a candidate move.
    Returns list of state dicts for feature extraction.
    """
    state = [
        {'id': p.id, 'owner': p.owner, 'ships': float(p.ships),
         'production': float(p.production), 'x': float(p.x),
         'y': float(p.y), 'radius': float(p.radius)}
        for p in planets
    ]
    id_to_idx = {s['id']: i for i, s in enumerate(state)}
    arrivals = {}   # tick -> [(planet_idx, owner, ships)]

    # Schedule existing fleets
    for f in raw_fleets:
        fowner, fx, fy, fangle = f[1], f[2], f[3], f[4]
        from_pid, fships = f[5], f[6]
        tgt, eta = _ray_first_planet(fx, fy, fangle, fships,
                                     planets, exclude_pid=from_pid)
        if tgt and 1 <= eta <= lookahead:
            idx = id_to_idx.get(tgt.id)
            if idx is not None:
                arrivals.setdefault(eta, []).append(
                    (idx, fowner, float(fships)))

    # Schedule candidate move
    src_idx = id_to_idx.get(src_id)
    if src_idx is not None and move_ships > 0:
        state[src_idx]['ships'] = max(
            0.0, state[src_idx]['ships'] - move_ships)
        cos_a = math.cos(move_angle); sin_a = math.sin(move_angle)
        sx = state[src_idx]['x'] + cos_a*(state[src_idx]['radius']+0.1)
        sy = state[src_idx]['y'] + sin_a*(state[src_idx]['radius']+0.1)
        spd = fleet_speed(move_ships)
        best_p, best_along = None, float('inf')
        for p in planets:
            if p.id == src_id:
                continue
            dx = p.x-sx; dy = p.y-sy
            along = dx*cos_a + dy*sin_a
            if along <= 0: continue
            perp = abs(dx*sin_a - dy*cos_a)
            if perp <= p.radius+0.5 and along < best_along:
                best_along = along; best_p = p
        if best_p:
            eta = max(1, int(math.ceil(best_along/spd)))
            if eta <= lookahead:
                idx = id_to_idx.get(best_p.id)
                if idx is not None:
                    arrivals.setdefault(eta, []).append(
                        (idx, my_player, float(move_ships)))

    # One opponent counter-move at tick 1 (reduces over-optimism)
    for s in state:
        if s['owner'] != -1 and s['owner'] != my_player and s['ships'] >= 10:
            opp = s['owner']
            opp_src_idx = max(
                (i for i, s2 in enumerate(state) if s2['owner'] == opp),
                key=lambda i: state[i]['ships'], default=None)
            if opp_src_idx is None: break
            my_tgt_idx = min(
                (i for i, s2 in enumerate(state)
                 if s2['owner'] == my_player),
                key=lambda i: state[i]['ships'], default=None)
            if my_tgt_idx is None: break
            opp_ships = int(state[opp_src_idx]['ships'] * 0.4)
            if opp_ships >= 5:
                arrivals.setdefault(1, []).append(
                    (my_tgt_idx, opp, float(opp_ships)))
                state[opp_src_idx]['ships'] = max(
                    0.0, state[opp_src_idx]['ships'] - opp_ships)
            break

    # Run ticks
    for tick in range(1, lookahead+1):
        for s in state:
            if s['owner'] != -1:
                s['ships'] += s['production']
        if tick in arrivals:
            pa = {}
            for idx, owner, ships in arrivals[tick]:
                pa.setdefault(idx, []).append((owner, ships))
            for idx, arr in pa.items():
                _resolve_combat(state[idx], arr)

    return state


# ───────────────────────────────────────────────────────────────
# SECTION 6: FEATURE EXTRACTION
# 16 scalar features describing the board from one player's POV.
# ───────────────────────────────────────────────────────────────

def extract_features(state, my_player, current_turn, n_players_hint=2):
    my_ships = my_planets = my_prod = my_cen = 0.0
    total_ships = total_prod = 0.0
    enemy_data = {}

    for s in state:
        sh = max(0.0, float(s.get('ships', 0)))
        total_ships += sh
        owner = s.get('owner', -1)
        if owner == -1:
            continue
        prod = float(s.get('production', 0))
        total_prod += prod
        cen = max(0.0, 60.0 - math.hypot(
            float(s.get('x', 50))-50.0,
            float(s.get('y', 50))-50.0))
        if owner == my_player:
            my_ships += sh
            my_planets += 1
            my_prod += prod
            my_cen += cen
        else:
            e = enemy_data.setdefault(owner,
                {'ships': 0.0, 'planets': 0, 'prod': 0.0, 'cen': 0.0})
            e['ships'] += sh; e['planets'] += 1
            e['prod'] += prod; e['cen'] += cen

    be_ships = be_pl = be_prod = be_cen = 0.0
    if enemy_data:
        best_e = max(enemy_data.values(),
                     key=lambda e: e['ships'] + 100*e['planets'])
        be_ships = best_e['ships']; be_pl = float(best_e['planets'])
        be_prod = best_e['prod']
        be_cen = best_e['cen'] / max(1, best_e['planets'])

    total_pl = max(1, len(state))
    s = lambda a, b: a/b if b > 1e-9 else 0.0
    n_pl = max(2, 1 + len(enemy_data), n_players_hint)

    return [
        current_turn / 500.0,               # 0  game phase
        n_pl / 4.0,                         # 1  player count
        s(my_planets, total_pl),            # 2  my planet %
        s(be_pl, total_pl),                 # 3  enemy planet %
        s(my_ships, total_ships),           # 4  my ship %
        s(be_ships, total_ships),           # 5  enemy ship %
        s(my_prod, total_prod),             # 6  my prod %
        s(be_prod, total_prod),             # 7  enemy prod %
        (my_cen / max(1, my_planets))/60.0, # 8  my centrality
        be_cen / 60.0,                      # 9  enemy centrality
        s(my_ships-be_ships, total_ships+1),# 10 ship lead
        s(my_planets-be_pl, total_pl),      # 11 planet lead
        s(my_prod-be_prod, total_prod+1),   # 12 prod lead
        1.0 if n_pl==2 else 0.0,            # 13 2P flag
        1.0 if n_pl==4 else 0.0,            # 14 4P flag
        s(my_ships, be_ships+1),            # 15 ship ratio
    ]


def _live_features(planets, player, turn):
    """Extract features directly from a list of Planet namedtuples."""
    state = [{'owner': p.owner, 'ships': float(p.ships),
              'production': float(p.production),
              'x': float(p.x), 'y': float(p.y)}
             for p in planets]
    return extract_features(state, player, turn)


# ───────────────────────────────────────────────────────────────
# SECTION 7: CLUSTER SYSTEM
# ───────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1


class Cluster:
    __slots__ = ['planets','total_ships','total_production',
                 'center_x','center_y','my_count','enemy_count',
                 'neutral_count','is_moving','planet_ids']

    def __init__(self):
        self.planets = []
        self.total_ships = self.total_production = 0.0
        self.center_x = self.center_y = 0.0
        self.my_count = self.enemy_count = self.neutral_count = 0
        self.is_moving = False
        self.planet_ids = set()

    def finalize(self, player):
        if not self.planets: return
        n = len(self.planets)
        self.center_x = sum(p.x for p in self.planets) / n
        self.center_y = sum(p.y for p in self.planets) / n
        self.total_ships = sum(p.ships for p in self.planets)
        self.total_production = sum(p.production for p in self.planets)
        self.planet_ids = {p.id for p in self.planets}
        for p in self.planets:
            if p.owner == player:      self.my_count += 1
            elif p.owner == -1:        self.neutral_count += 1
            else:                      self.enemy_count += 1


def build_clusters(planets, player, initial_by_id):
    result = []
    for is_moving in (False, True):
        grp = [p for p in planets
               if (not is_static_planet(p, initial_by_id)) == is_moving]
        if not grp: continue
        n = len(grp)
        uf = UnionFind(n)
        for i in range(n):
            for j in range(i+1, n):
                pi, pj = grp[i], grp[j]
                if dist(pi.x, pi.y, pj.x, pj.y) <= (
                        CLUSTER_BASE_RADIUS + pi.radius +
                        CLUSTER_BASE_RADIUS + pj.radius):
                    uf.union(i, j)
        groups = {}
        for i in range(n):
            groups.setdefault(uf.find(i), []).append(grp[i])
        for members in groups.values():
            c = Cluster()
            c.planets = members
            c.is_moving = is_moving
            c.finalize(player)
            result.append(c)
    return result


def find_cluster_for_planet(pid, clusters):
    for c in clusters:
        if pid in c.planet_ids:
            return c
    return None


# ───────────────────────────────────────────────────────────────
# SECTION 8: FLEET TRACKING
# ───────────────────────────────────────────────────────────────

def track_threats(raw_fleets, player, all_planets, current_turn):
    """
    Returns {planet_id: [(eta_turn, fleet_owner, fleet_ships), ...]}
    sorted by eta_turn (soonest first).
    """
    threats = {}
    for f in raw_fleets:
        fowner = f[1]
        if fowner == player or fowner == -1:
            continue
        fid, fx, fy, fangle = f[0], f[2], f[3], f[4]
        from_pid, fships = f[5], f[6]
        tgt, eta = _ray_first_planet(fx, fy, fangle, fships,
                                     all_planets, exclude_pid=from_pid)
        if tgt is None:
            continue
        entry = (current_turn + eta, fowner, fships)
        threats.setdefault(tgt.id, []).append(entry)
    for pid in threats:
        threats[pid].sort()
    return threats


def find_enemy_under_attack(threats, player, planets_by_id):
    """
    Enemies being hit by OTHER enemies (not us) → vulture targets.
    Returns {pid: total_incoming_ships}.
    """
    result = {}
    for pid, tlist in threats.items():
        p = planets_by_id.get(pid)
        if p is None or p.owner == player or p.owner == -1:
            continue
        incoming = sum(ships for (_, fowner, ships) in tlist
                       if fowner != player and fowner != p.owner)
        if incoming > 0:
            result[pid] = incoming
    return result


# ───────────────────────────────────────────────────────────────
# SECTION 9: HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────

def get_phase(turn):
    if turn <= PHASE_LENGTH:     return 1
    if turn <= PHASE_LENGTH*2:   return 2
    if turn <= PHASE_LENGTH*3:   return 3
    return 4


def max_sendable(available, is_defense=False):
    pct = DEFENSE_SEND_RESERVE_PCT if is_defense else DEFENSE_RESERVE_PCT
    return max(0, available - max(1, int(available * pct)))


def _get(obs, field, default=None):
    if hasattr(obs, field): return getattr(obs, field)
    if isinstance(obs, dict): return obs.get(field, default)
    return default


# ───────────────────────────────────────────────────────────────
# SECTION 10: RL-GUIDED DECISION HELPERS
# These are the four specific decisions improved by the value fn.
# ───────────────────────────────────────────────────────────────

def vf_pick_attack_target(src_planet, candidates, planets, raw_fleets,
                           player, available, initial_by_id, ang_vel,
                           current_turn, targeted_this_turn):
    """
    RL decision 1: WHICH PLANET TO ATTACK.
    Scores each candidate by simulating forward and calling value_score().
    Falls back to heuristic (Tier/distance) if no model loaded.
    Returns (best_target, needed_ships, p_win) or None.
    """
    sendable = max_sendable(available)
    if sendable <= 0:
        return None

    best_result = None
    best_p_win  = -1.0

    for c in candidates:
        safe = safe_angle_and_distance(src_planet.x, src_planet.y,
                                       src_planet.radius,
                                       c.x, c.y, c.radius)
        if safe is None:
            continue
        angle = safe[0]

        aim = aim_with_prediction(src_planet, c, sendable,
                                  initial_by_id, ang_vel)
        if aim is None:
            continue
        angle, turns, tx, ty = aim

        if c.owner == -1:
            needed = int(c.ships) + 1
        else:
            needed = int(c.ships + c.production * turns) + 1

        if needed > sendable:
            continue

        if VALUE_FN_LOADED:
            sim = simulate_forward(planets, raw_fleets, player,
                                   src_planet.id, angle, needed)
            feats = extract_features(sim, player, current_turn)
            p_win = attack_score(feats)
        else:
            # Heuristic fallback score
            d = dist(src_planet.x, src_planet.y, c.x, c.y)
            p_win = (c.production * 15 - d * 0.5 - needed * 0.3) / 100.0

        if p_win > best_p_win:
            best_p_win = p_win
            best_result = (c, needed, p_win)

    # Only commit if above confidence threshold (or no model)
    if best_result:
        if VALUE_FN_LOADED and best_result[2] < VF_ATTACK_MIN_CONFIDENCE:
            return None
    return best_result


def vf_should_defend_over_beam(beam_src, beam_tgt, threatened_planet,
                                 threat_ships, planets, raw_fleets,
                                 player, initial_by_id, ang_vel,
                                 current_turn):
    """
    RL decision 2: DEFEND OR KEEP BEAMING?
    Simulates both futures, returns True if defending wins more.
    Falls back to True (always defend) if no model.
    """
    if not VALUE_FN_LOADED:
        return True

    # Future A: keep beaming
    aim_beam = aim_with_prediction(beam_src, beam_tgt,
                                   beam_src.production,
                                   initial_by_id, ang_vel)
    score_beam = 0.4  # default: beaming is slightly bad if we can't aim
    if aim_beam:
        sim_b = simulate_forward(planets, raw_fleets, player,
                                 beam_src.id, aim_beam[0],
                                 beam_src.production)
        feats_b = extract_features(sim_b, player, current_turn)
        score_beam = defend_score(feats_b)

    # Future B: send defense ships
    d_angle = math.atan2(threatened_planet.y - beam_src.y,
                         threatened_planet.x - beam_src.x)
    def_ships = max_sendable(beam_src.ships, is_defense=True)
    sim_d = simulate_forward(planets, raw_fleets, player,
                              beam_src.id, d_angle, def_ships)
    feats_d = extract_features(sim_d, player, current_turn)
    score_def = defend_score(feats_d)

    # Defend if it's meaningfully better
    return score_def > score_beam + VF_DEFEND_MIN_CONFIDENCE - 0.5


def vf_pick_defender(cluster_helpers, threatened_planet, planets,
                      raw_fleets, player, initial_by_id, ang_vel,
                      current_turn, deficit):
    """
    RL decision 3: WHICH PLANET IN THE CLUSTER SHOULD DEFEND?
    Simulates each helper sending ships, picks the one whose departure
    hurts the cluster least (highest P(win) after sending).
    Falls back to nearest-first if no model.
    Returns sorted list of (helper_planet, send_ships).
    """
    if not VALUE_FN_LOADED:
        # Heuristic: nearest helpers first
        helpers_sorted = sorted(
            cluster_helpers,
            key=lambda h: dist(h.x, h.y,
                               threatened_planet.x, threatened_planet.y))
        result = []
        remaining = deficit
        for h in helpers_sorted:
            if remaining <= 0: break
            send = min(max_sendable(h.ships, True), remaining + 3)
            if send > 0:
                result.append((h, send))
                remaining -= send
        return result

    # Score each helper: who can send ships while keeping cluster healthy?
    scored = []
    d_angle = math.atan2(threatened_planet.y - threatened_planet.y,
                         threatened_planet.x - threatened_planet.x)
    for h in cluster_helpers:
        send = min(max_sendable(h.ships, True), deficit + 3)
        if send <= 0:
            continue
        angle = math.atan2(threatened_planet.y - h.y,
                           threatened_planet.x - h.x)
        sim = simulate_forward(planets, raw_fleets, player,
                               h.id, angle, send)
        feats = extract_features(sim, player, current_turn)
        p_win = defend_score(feats)
        scored.append((p_win, h, send))

    scored.sort(key=lambda x: -x[0])  # highest P(win) first

    result = []
    remaining = deficit
    for p_win, h, send in scored:
        if remaining <= 0: break
        actual_send = min(send, remaining + 3)
        result.append((h, actual_send))
        remaining -= actual_send
    return result


def vf_pick_fleet_size(src_planet, tgt_planet, base_needed, max_sendable_ships,
                        planets, raw_fleets, player, initial_by_id, ang_vel,
                        current_turn):
    """
    RL decision 4: HOW MANY SHIPS TO SEND?
    Tests base_needed, 1.5×, and 2× amounts.
    Picks the amount that gives the highest P(win) improvement over base.
    Falls back to base_needed if no model or no improvement found.
    Returns chosen ship count.
    """
    if not VALUE_FN_LOADED:
        return base_needed

    aim = aim_with_prediction(src_planet, tgt_planet, base_needed,
                               initial_by_id, ang_vel)
    if aim is None:
        return base_needed

    angle = aim[0]
    base_sim = simulate_forward(planets, raw_fleets, player,
                                src_planet.id, angle, base_needed)
    base_feats = extract_features(base_sim, player, current_turn)
    base_score = attack_score(base_feats)

    best_count = base_needed
    best_score = base_score

    for multiplier in (1.5, 2.0):
        candidate = int(base_needed * multiplier)
        if candidate > max_sendable_ships:
            break
        sim = simulate_forward(planets, raw_fleets, player,
                               src_planet.id, angle, candidate)
        feats = extract_features(sim, player, current_turn)
        score = attack_score(feats)
        # Only upsize if it gives a meaningful improvement
        if score > best_score + VF_FLEET_UPSIZE_THRESHOLD:
            best_score = score
            best_count = candidate

    return best_count


# ───────────────────────────────────────────────────────────────
# SECTION 11: MAIN AGENT CLASS
# ───────────────────────────────────────────────────────────────

class ClusterOrbitAgent:
    def __init__(self):
        self._initial_by_id = {}
        self.current_turn   = 0
        self.last_ships      = {}
        self.last_owner      = {}
        self.last_ejected    = {}
        self.capture_turns   = {}   # pid -> turn captured
        self.beam_targets    = {}   # src_pid -> tgt_pid
        self.buff_targets    = {}   # src_pid -> (tgt_pid, start_turn)

    def __call__(self, obs, config=None):
        self.current_turn += 1
        turn = self.current_turn

        # ── Parse observation ──────────────────────────────────
        player    = _get(obs, "player", 0)
        raw_p     = _get(obs, "planets", [])
        ang_vel   = _get(obs, "angular_velocity", 0.03)
        comet_ids = set(_get(obs, "comet_planet_ids", []))
        raw_f     = _get(obs, "fleets", [])

        all_planets  = [Planet(*p) for p in raw_p]
        planets      = [p for p in all_planets if p.id not in comet_ids]
        planets_by_id = {p.id: p for p in planets}

        if not self._initial_by_id:
            raw_init = _get(obs, "initial_planets", None)
            if raw_init:
                self._initial_by_id = {p[0]: Planet(*p) for p in raw_init}
            else:
                self._initial_by_id = {p.id: p for p in all_planets}

        my_planets = [p for p in planets if p.owner == player]
        if not my_planets:
            self._update(planets, [])
            return []

        # ── Build clusters ─────────────────────────────────────
        clusters   = build_clusters(planets, player, self._initial_by_id)
        my_clusters = [c for c in clusters if c.my_count > 0]

        # ── Fleet tracking ─────────────────────────────────────
        threats = track_threats(raw_f, player, planets, turn)
        enemy_under_attack = find_enemy_under_attack(
            threats, player, planets_by_id)

        # ── Track captures ─────────────────────────────────────
        for p in planets:
            if p.owner == player:
                if self.last_owner.get(p.id, -1) != player:
                    self.capture_turns[p.id] = turn

        # ── Retroactive attack detection (fallback) ────────────
        retro_attacked = set()
        for p in planets:
            if (self.last_owner.get(p.id) == player
                    and p.owner == player):
                ejected  = self.last_ejected.get(p.id, 0)
                expected = self.last_ships.get(p.id, 0) + p.production - ejected
                if p.ships < expected - 0.5:
                    retro_attacked.add(p.id)

        phase = get_phase(turn)

        # ── Clean stale beam/buff ──────────────────────────────
        my_ids = {p.id for p in my_planets}
        for pid in list(self.beam_targets):
            if pid not in my_ids:
                del self.beam_targets[pid]
            else:
                t = planets_by_id.get(self.beam_targets[pid])
                if t and t.owner == player:
                    del self.beam_targets[pid]
        for pid in list(self.buff_targets):
            if pid not in my_ids:
                del self.buff_targets[pid]

        # ══════════════════════════════════════════════════════
        # STEP 1: BUILD DEFENSE ASSIGNMENTS
        # Uses RL decision 3 (which_defender) to pick the best
        # planet from each cluster to send reinforcements.
        # ══════════════════════════════════════════════════════
        defense_assignments = {}   # src_pid -> (tgt_pid, ships)
        assigned_defenders  = set()

        my_threatened = []
        for pid, tlist in threats.items():
            p = planets_by_id.get(pid)
            if p and p.owner == player:
                earliest = tlist[0][0]  # (eta_turn, owner, ships)
                total_threat = sum(s for (_, __, s) in tlist)
                my_threatened.append((earliest, pid, total_threat))
        my_threatened.sort()

        for earliest_eta, pid, total_threat in my_threatened:
            planet = planets_by_id[pid]
            turns_until = max(1, earliest_eta - turn)
            future_ships = planet.ships + planet.production * turns_until

            if future_ships >= total_threat * 1.1:
                continue  # self-defending

            deficit = int(total_threat - future_ships) + 1
            cluster = find_cluster_for_planet(pid, my_clusters)
            helpers = []
            if cluster:
                helpers = [cp for cp in cluster.planets
                           if cp.owner == player and cp.id != pid
                           and cp.id not in assigned_defenders]

            # Cross-cluster help if needed
            helper_cap = sum(max_sendable(h.ships, True) for h in helpers)
            if helper_cap < deficit:
                if (cluster
                        and total_threat > cluster.total_ships
                                          * CROSS_CLUSTER_HELP_RATIO):
                    for oc in my_clusters:
                        if oc is not cluster:
                            helpers += [
                                cp for cp in oc.planets
                                if cp.owner == player
                                and cp.id not in assigned_defenders
                                and cp.id != pid
                            ]

            # RL: pick best defenders (decision 3)
            assigned = vf_pick_defender(
                helpers, planet, planets, raw_f, player,
                self._initial_by_id, ang_vel, turn, deficit)

            for h, send in assigned:
                defense_assignments[h.id] = (pid, send)
                assigned_defenders.add(h.id)

        # Retroactive fallback
        for pid in retro_attacked:
            if pid in threats:
                continue
            planet = planets_by_id.get(pid)
            if planet is None or planet.owner != player:
                continue
            cluster = find_cluster_for_planet(pid, my_clusters)
            if cluster:
                helpers = [cp for cp in cluster.planets
                           if cp.owner == player and cp.id != pid
                           and cp.id not in assigned_defenders]
                for h in helpers[:1]:
                    send = max(1, min(5, max_sendable(h.ships, True)))
                    if send > 0:
                        defense_assignments[h.id] = (pid, send)
                        assigned_defenders.add(h.id)

        # ══════════════════════════════════════════════════════
        # STEP 2: PER-PLANET DECISION LADDER
        # ══════════════════════════════════════════════════════
        moves = []
        available = {p.id: p.ships for p in my_planets}
        targeted_this_turn = set()
        has_static = any(is_static_planet(p, self._initial_by_id)
                         for p in my_planets)
        sorted_planets = sorted(my_planets, key=lambda p: -p.production)

        for planet in sorted_planets:
            pid = planet.id

            # ── P1: Lockout ────────────────────────────────────
            if pid in self.capture_turns:
                if turn - self.capture_turns[pid] < POST_CAPTURE_LOCKOUT:
                    continue

            # ── P0: Bootstrap (no static base yet) ────────────
            if not has_static and not is_static_planet(
                    planet, self._initial_by_id):
                statics = [p for p in planets
                           if is_static_planet(p, self._initial_by_id)
                           and p.owner != player
                           and p.id not in targeted_this_turn]
                if statics:
                    tgt = min(statics,
                              key=lambda t: (t.ships,
                                             dist(planet.x, planet.y,
                                                  t.x, t.y)))
                    needed = int(tgt.ships) + 1
                    mv = self._fire(planet, tgt, needed, available, ang_vel)
                    if mv:
                        moves.append(mv)
                        targeted_this_turn.add(tgt.id)
                continue

            # ── P2: Defense assignment ─────────────────────────
            if pid in defense_assignments:
                tgt_pid, ships_to_send = defense_assignments[pid]
                tgt = planets_by_id.get(tgt_pid)
                if tgt and ships_to_send > 0:
                    mv = self._fire(planet, tgt, ships_to_send,
                                    available, ang_vel, is_defense=True)
                    if mv:
                        moves.append(mv)
                continue  # defense is absolute priority

            # ── P3: Beam mode ──────────────────────────────────
            if pid in self.beam_targets:
                tgt_pid = self.beam_targets[pid]
                tgt = planets_by_id.get(tgt_pid)
                if tgt and tgt.owner != player:
                    # RL decision 2: defend or keep beaming?
                    should_pause = False
                    if pid in threats or pid in retro_attacked:
                        # find what's threatening us
                        t_list = threats.get(pid, [])
                        threat_ships = sum(s for (_, __, s) in t_list)
                        if threat_ships > 0:
                            should_pause = vf_should_defend_over_beam(
                                planet, tgt, planet, threat_ships,
                                planets, raw_f, player,
                                self._initial_by_id, ang_vel, turn)

                    if should_pause:
                        del self.beam_targets[pid]
                        # Fall through to defense/expansion
                    else:
                        beam_ships = planet.production
                        if planet.production < 3:
                            beam_ships = max(1, planet.production // 2)

                        # RL decision 4: fleet size
                        beam_ships = vf_pick_fleet_size(
                            planet, tgt, beam_ships,
                            max_sendable(available[pid]),
                            planets, raw_f, player,
                            self._initial_by_id, ang_vel, turn)

                        mv = self._fire(planet, tgt, beam_ships,
                                        available, ang_vel)
                        if mv:
                            moves.append(mv)
                        continue
                else:
                    del self.beam_targets[pid]
                    if (tgt and tgt.owner == player
                            and planet.production < 3
                            and tgt_pid in self.capture_turns):
                        self.buff_targets[pid] = (tgt_pid, turn)

            # ── P4: Buff mode ──────────────────────────────────
            if pid in self.buff_targets:
                tgt_pid, start_turn = self.buff_targets[pid]
                tgt = planets_by_id.get(tgt_pid)
                elapsed = turn - start_turn
                cluster = find_cluster_for_planet(pid, my_clusters)
                csz = len(cluster.planets) if cluster else 1
                max_buff = min(BUFF_WAVE_MAX, BUFF_WAVE_MIN + csz)
                if (tgt and tgt.owner == player
                        and elapsed < max_buff
                        and tgt.ships < SMALL_PLANET_SHIP_CAP):
                    buff_ships = max(1, planet.production // 2)
                    mv = self._fire(planet, tgt, buff_ships,
                                    available, ang_vel, True)
                    if mv:
                        moves.append(mv)
                    continue
                else:
                    del self.buff_targets[pid]

            # ── P4.5: Join cluster-mate's beam ────────────────
            if (phase <= 3 and turn <= ATTACK_PHASE_CUTOFF):
                cluster = find_cluster_for_planet(pid, my_clusters)
                joined = False
                if cluster:
                    for mate_pid in cluster.planet_ids:
                        if mate_pid != pid and mate_pid in self.beam_targets:
                            tgt_pid = self.beam_targets[mate_pid]
                            tgt = planets_by_id.get(tgt_pid)
                            if tgt and tgt.owner != player:
                                bs = planet.production
                                if planet.production < 3:
                                    bs = max(1, planet.production // 2)
                                bs = vf_pick_fleet_size(
                                    planet, tgt, bs,
                                    max_sendable(available[pid]),
                                    planets, raw_f, player,
                                    self._initial_by_id, ang_vel, turn)
                                mv = self._fire(planet, tgt, bs,
                                                available, ang_vel)
                                if mv:
                                    moves.append(mv)
                                    self.beam_targets[pid] = tgt_pid
                                    joined = True
                                break
                if joined:
                    continue

            # ── P5: Phase-based offense ────────────────────────
            allow_expansion = (phase <= 3 and turn <= ATTACK_PHASE_CUTOFF)
            sendable = max_sendable(available[pid])

            if sendable > 0:
                # 5a: Vulture (all phases)
                vulture = self._find_vulture_target(
                    planet, enemy_under_attack, planets_by_id,
                    available[pid], targeted_this_turn, ang_vel)
                if vulture:
                    tgt, needed = vulture
                    needed = vf_pick_fleet_size(
                        planet, tgt, needed, sendable,
                        planets, raw_f, player,
                        self._initial_by_id, ang_vel, turn)
                    mv = self._fire(planet, tgt, needed, available, ang_vel)
                    if mv:
                        moves.append(mv)
                        targeted_this_turn.add(tgt.id)
                        self.beam_targets[pid] = tgt.id
                        continue

                # 5b: RL-guided tier attack (phases 1-3)
                if allow_expansion:
                    candidates = self._build_tier_candidates(
                        planet, planets, player, available[pid],
                        targeted_this_turn, clusters)

                    rl_result = vf_pick_attack_target(
                        planet, candidates, planets, raw_f,
                        player, available[pid],
                        self._initial_by_id, ang_vel,
                        turn, targeted_this_turn)

                    if rl_result:
                        tgt, needed, p_win = rl_result
                        needed = vf_pick_fleet_size(
                            planet, tgt, needed, sendable,
                            planets, raw_f, player,
                            self._initial_by_id, ang_vel, turn)
                        mv = self._fire(planet, tgt, needed, available, ang_vel)
                        if mv:
                            moves.append(mv)
                            targeted_this_turn.add(tgt.id)
                            if tgt.owner != -1:
                                self.beam_targets[pid] = tgt.id
                            continue

            # ── P6: Buff opportunity ───────────────────────────
            if planet.production < 3 and pid not in self.buff_targets:
                cluster = find_cluster_for_planet(pid, my_clusters)
                if cluster:
                    for cp in cluster.planets:
                        if (cp.owner == player and cp.id in self.capture_turns
                                and cp.id != pid
                                and cp.ships < SMALL_PLANET_SHIP_CAP):
                            if turn - self.capture_turns[cp.id] >= POST_CAPTURE_LOCKOUT:
                                self.buff_targets[pid] = (cp.id, turn)
                                mv = self._fire(planet, cp,
                                                max(1, planet.production//2),
                                                available, ang_vel, True)
                                if mv: moves.append(mv)
                                break

            # ── P7: Peacetime growth ───────────────────────────
            cluster = find_cluster_for_planet(pid, my_clusters)
            if cluster and cluster.my_count > 1:
                my_in_c = [cp for cp in cluster.planets
                           if cp.owner == player and cp.id != pid]
                weak = [cp for cp in my_in_c
                        if cp.ships < SMALL_PLANET_SHIP_CAP]
                if weak:
                    avg = cluster.total_ships / max(1, cluster.my_count)
                    excess = int(planet.ships - avg)
                    if excess > 5:
                        weakest = min(weak, key=lambda cp: cp.ships)
                        send = min(excess, max_sendable(available[pid]))
                        if send > 0:
                            mv = self._fire(planet, weakest, send,
                                            available, ang_vel, True)
                            if mv: moves.append(mv)

        self._update(planets, moves)
        return moves

    # ── Helper methods ─────────────────────────────────────────

    def _fire(self, src, tgt, ships_wanted, available,
              ang_vel, is_defense=False):
        sendable = max_sendable(available[src.id], is_defense=is_defense)
        send = min(ships_wanted, sendable)
        if send <= 0:
            return None
        aim = aim_with_prediction(src, tgt, send, self._initial_by_id, ang_vel)
        if aim is None:
            return None
        available[src.id] -= send
        return [src.id, aim[0], send]

    def _build_tier_candidates(self, planet, all_planets, player,
                                available_ships, targeted_this_turn,
                                clusters):
        """Build ranked candidate targets (Tier1 → Tier2 → Tier3)."""
        sendable = max_sendable(available_ships)
        t1, t2, t3 = [], [], []
        for c in all_planets:
            if c.owner == player or c.id in targeted_this_turn:
                continue
            safe = safe_angle_and_distance(planet.x, planet.y,
                                           planet.radius,
                                           c.x, c.y, c.radius)
            if safe is None:
                continue
            d = dist(planet.x, planet.y, c.x, c.y)
            tc = find_cluster_for_planet(c.id, clusters)
            if (tc and tc.enemy_count >= 3 and c.owner != -1
                    and not (c.production >= TIER1_MIN_PRODUCTION
                             and c.ships <= TIER1_MAX_SHIPS)):
                continue
            if c.production >= TIER1_MIN_PRODUCTION and c.ships <= TIER1_MAX_SHIPS:
                t1.append((d, c))
            elif TIER2_MIN_SHIPS <= c.ships <= TIER2_MAX_SHIPS:
                t2.append((d, c))
            elif c.owner == -1:
                t3.append((d, c))
            elif c.ships < sendable * 0.6:
                t3.append((d, c))
        # Return flat list preserving tier priority order
        result = []
        for tier in (t1, t2, t3):
            tier.sort()
            result.extend(c for _, c in tier)
        return result

    def _find_vulture_target(self, planet, enemy_under_attack,
                              planets_by_id, available_ships,
                              targeted_this_turn, ang_vel):
        sendable = max_sendable(available_ships)
        if sendable <= 0:
            return None
        best = None; best_score = -float('inf')
        for epid, incoming in enemy_under_attack.items():
            if epid in targeted_this_turn:
                continue
            ep = planets_by_id.get(epid)
            if ep is None:
                continue
            est_remaining = max(1, ep.ships - int(incoming * 0.5))
            est = estimate_arrival(planet.x, planet.y, planet.radius,
                                   ep.x, ep.y, ep.radius, sendable)
            if est is None:
                continue
            _, turns = est
            garrison = est_remaining + ep.production * turns
            needed = int(garrison) + 1
            if needed > sendable:
                continue
            score = ep.production * 15 - needed
            if score > best_score:
                best_score = score
                best = (ep, needed)
        return best

    def _update(self, planets, moves):
        self.last_ships   = {p.id: p.ships for p in planets}
        self.last_owner   = {p.id: p.owner for p in planets}
        self.last_ejected = {}
        for src_id, _, ships in moves:
            self.last_ejected[src_id] = (
                self.last_ejected.get(src_id, 0) + ships)


# ───────────────────────────────────────────────────────────────
# SECTION 12: ENTRY POINT
# ───────────────────────────────────────────────────────────────

agent_instance = ClusterOrbitAgent()

def agent(obs, config=None):
    return agent_instance(obs, config)
