"""
features.py
-----------
Converts a raw Orbit Wars observation into structured feature tensors
for the neural decision network.

Key design decisions:
  - ALL features are ego-centric and normalized to [0, 1] or [-1, 1].
  - Planet features are *relative* to the querying player — no raw coordinates.
  - The (source, target) edge feature is the main learning signal.
  - This module has zero dependency on main.py at training time; it only
    needs the raw observation dict.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict, Optional

# ── constants mirrored from the env ────────────────────────────────────────
BOARD        = 100.0
CENTER_X     = 50.0
CENTER_Y     = 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0
TOTAL_STEPS  = 500
ROT_LIMIT    = 50.0   # orbital_radius + radius < this → orbiting

# ── helpers ─────────────────────────────────────────────────────────────────

def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    r = math.log(max(1, ships)) / math.log(1000)
    r = max(0.0, min(1.0, r))
    return 1.0 + (MAX_SPEED - 1.0) * (r ** 1.5)

def _is_orbiting(px, py, pr, initial_by_id, pid):
    init = initial_by_id.get(pid)
    if init is None:
        return False
    r = _dist(init[2], init[3], CENTER_X, CENTER_Y)
    return (r + init[4]) < ROT_LIMIT

def _path_hits_sun(sx, sy, tx, ty, safety=1.5):
    dx, dy = tx - sx, ty - sy
    seg_sq = dx*dx + dy*dy
    if seg_sq < 1e-9:
        return _dist(CENTER_X, CENTER_Y, sx, sy) < SUN_R + safety
    t = ((CENTER_X - sx)*dx + (CENTER_Y - sy)*dy) / seg_sq
    t = max(0.0, min(1.0, t))
    px = sx + t*dx
    py = sy + t*dy
    return _dist(CENTER_X, CENTER_Y, px, py) < SUN_R + safety

def _travel_turns(sx, sy, sr, tx, ty, tr, ships):
    angle = math.atan2(ty - sy, tx - sx)
    lx = sx + math.cos(angle) * (sr + 0.1)
    ly = sy + math.sin(angle) * (sr + 0.1)
    d = max(0.0, _dist(sx, sy, tx, ty) - (sr + 0.1) - tr)
    ex = lx + math.cos(angle) * d
    ey = ly + math.sin(angle) * d
    if _path_hits_sun(lx, ly, ex, ey):
        return None
    spd = _fleet_speed(max(1, ships))
    return max(1, int(math.ceil(d / spd)))

# ── global state features (31-d, matches _extract_features in main.py) ──────

def global_features(obs: dict) -> List[float]:
    """
    31 global features describing the whole board state from the
    perspective of `obs['player']`. Compatible with value_fn_1v1 input.
    """
    player      = obs.get('player', 0)
    step        = obs.get('step', 0) or 0
    planets_raw = obs.get('planets', []) or []
    fleets_raw  = obs.get('fleets', []) or []

    ms = mp = mpr = mc = ts = tpr = 0.0
    ed: Dict = {}
    msl, mprl, mpos, epos = [], [], [], []

    for p in planets_raw:
        sh = max(0.0, float(p[5])); ts += sh
        ow = p[1]
        if ow == -1: continue
        pr = float(p[6]); tpr += pr
        cn = max(0.0, 60.0 - math.hypot(float(p[2]) - 50, float(p[3]) - 50))
        if ow == player:
            ms += sh; mp += 1; mpr += pr; mc += cn
            msl.append(sh); mprl.append(pr)
            mpos.append((float(p[2]), float(p[3])))
        else:
            e = ed.setdefault(ow, {'s': 0., 'p': 0, 'pr': 0., 'c': 0.,
                                    'sl': [], 'prl': []})
            e['s'] += sh; e['p'] += 1; e['pr'] += pr; e['c'] += cn
            e['sl'].append(sh); e['prl'].append(pr)
            epos.append((float(p[2]), float(p[3])))

    bes = bep = bepr = bec = 0.0; best_ow = None
    if ed:
        best_ow, be = max(ed.items(), key=lambda kv: kv[1]['s'] + 100 * kv[1]['p'])
        bes = be['s']; bep = float(be['p']); bepr = be['pr']
        bec = be['c'] / max(1, be['p'])

    tp   = max(1, len(planets_raw))
    sf   = lambda a, b: a / b if b > 1e-9 else 0.0
    np2  = max(2, 1 + len(ed))
    # clamp to 2 or 4 for the 1-hot
    np2  = 2 if np2 <= 2 else 4

    my_max_sh    = max(msl) if msl else 0.0
    be_sl        = ed[best_ow]['sl'] if best_ow else []
    enemy_max_sh = max(be_sl) if be_sl else 0.0
    neutral_cnt  = sum(1 for p in planets_raw if p[1] == -1)
    my_top3_pr   = sum(sorted(mprl, reverse=True)[:3]) if mprl else 0.0
    be_prl       = ed[best_ow]['prl'] if best_ow else []
    enemy_top3_pr = sum(sorted(be_prl, reverse=True)[:3]) if be_prl else 0.0

    min_edist = 1.0
    if mpos and epos:
        min_edist = min(math.hypot(mx - ex, my - ey)
                        for mx, my in mpos
                        for ex, ey in epos) / 100.0

    mfs = efs = 0.0; mfc = efc = 0
    for f in fleets_raw:
        fow = f[1]; fsh = float(f[6])
        if fow == player: mfs += fsh; mfc += 1
        elif fow != -1:   efs += fsh; efc += 1

    ta = ts + mfs + efs + 1.0
    return [
        step / 500.0,
        np2 / 4.0,
        sf(mp, tp), sf(bep, tp),
        sf(ms, ts), sf(bes, ts),
        sf(mpr, tpr), sf(bepr, tpr),
        (mc / max(1, mp)) / 60.0, bec / 60.0,
        sf(ms - bes, ts + 1), sf(mp - bep, tp), sf(mpr - bepr, tpr + 1),
        1.0 if np2 == 2 else 0.0, 1.0 if np2 == 4 else 0.0,
        sf(ms, bes + 1),
        my_max_sh / max(1.0, ts), enemy_max_sh / max(1.0, ts),
        neutral_cnt / tp,
        sf(my_top3_pr, tpr), sf(enemy_top3_pr, tpr),
        min(1.0, min_edist),
        sf(mfs, ta), sf(efs, ta),
        min(1.0, mfc / 8.0), min(1.0, efc / 8.0),
        sf(mfs, ms + 1.0), sf(efs, bes + 1.0),
        sf(mpr, mp * 5.0 + 1.0),
        min(1.0, (step / 500.0) * sf(ms, ts + 1.0)),
        float(np2),
    ]

# ── fleet arrival table ───────────────────────────────────────────────────

def _fleet_target_planet(fleet, planets_raw):
    """
    Ray-sphere intersection: which planet is this fleet heading toward?
    Fleet format: [id, owner, x, y, angle, from_planet_id, ships]
    Returns (planet_id, turns_to_arrive) or (None, None).
    """
    fx, fy   = float(fleet[2]), float(fleet[3])
    angle    = float(fleet[4])
    ships    = max(1, int(fleet[6]))
    dir_x    = math.cos(angle)
    dir_y    = math.sin(angle)
    speed    = _fleet_speed(ships)

    best_id    = None
    best_turns = float('inf')

    for p in planets_raw:
        px, py, pr = float(p[2]), float(p[3]), float(p[4])
        dx = px - fx
        dy = py - fy
        proj = dx * dir_x + dy * dir_y
        if proj <= 0:
            continue
        perp_sq = dx*dx + dy*dy - proj*proj
        if perp_sq >= pr * pr:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, pr*pr - perp_sq)))
        turns = hit_d / speed
        if turns < best_turns:
            best_turns = turns
            best_id    = p[0]

    return best_id, best_turns


def fleet_arrivals_table(fleets_raw, planets_raw, player, total_ships):
    """
    Single pass over in-flight fleets → per-planet arrival summary.

    Returns dict: planet_id → {
        'friendly_ships' : int   (total friendly ships en route here),
        'enemy_ships'    : int   (total enemy ships en route here),
        'first_threat'   : float (turns until first enemy arrives, 999=none),
        'first_friendly' : float (turns until first friendly arrives, 999=none),
    }
    """
    table = {p[0]: {'friendly_ships': 0, 'enemy_ships': 0,
                    'first_threat': 999.0, 'first_friendly': 999.0}
             for p in planets_raw}

    for f in (fleets_raw or []):
        owner = f[1]
        ships = int(f[6])
        if ships <= 0:
            continue
        target_id, turns = _fleet_target_planet(f, planets_raw)
        if target_id is None or target_id not in table:
            continue
        row = table[target_id]
        if owner == player:
            row['friendly_ships'] += ships
            row['first_friendly'] = min(row['first_friendly'], turns)
        else:
            row['enemy_ships'] += ships
            row['first_threat'] = min(row['first_threat'], turns)

    return table


# ── per-planet features (relative to player) ─────────────────────────────
# These are the features used to represent each planet as a *node*.

N_PLANET_FEATS = 24   # 18 base + 6 fleet-state features

def planet_features(p, player: int, step: int,
                    total_ships: float, total_prod: float,
                    initial_by_id: dict,
                    my_total_ships: float,
                    enemy_total_ships: float,
                    arrivals: dict = None) -> List[float]:
    """
    24 features for a single planet, ego-centric w.r.t. `player`.
    p is a raw planet list: [id, owner, x, y, radius, ships, production]

    Features 0-17: base planet features (unchanged)
    Features 18-23: fleet-state features (new)
      18. friendly_ships_arriving  — friendly fleets en route here / total_ships
      19. enemy_ships_arriving     — enemy fleets en route here / total_ships
      20. net_arrival_balance      — (friendly - enemy) / total_ships, clamp [-1,1]
      21. turns_to_first_threat    — 0=imminent, 1=no threat ≤100 turns
      22. already_covered          — 1 if any friendly fleet is heading here
      23. contested                — 1 if both friendly and enemy fleets en route
    """
    pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    # ownership encoding: 1-hot [mine, enemy, neutral]
    is_mine    = 1.0 if owner == player else 0.0
    is_enemy   = 1.0 if (owner != player and owner != -1) else 0.0
    is_neutral = 1.0 if owner == -1 else 0.0

    # position: distance from center, quadrant angle normalized
    cx = x - CENTER_X
    cy = y - CENTER_Y
    dist_center = math.hypot(cx, cy) / 50.0
    angle_norm  = math.atan2(cy, cx) / math.pi

    # orbiting vs static
    r_from_center = math.hypot(cx, cy)
    is_orbiting = 1.0 if (r_from_center + radius) < ROT_LIMIT else 0.0

    # economic
    ships_norm = min(1.0, ships / max(1.0, total_ships))
    prod_norm  = prod / 5.0

    # relative military value
    ships_vs_mine  = min(2.0, ships / max(1.0, my_total_ships))
    ships_vs_enemy = min(2.0, ships / max(1.0, enemy_total_ships))

    dist_sun       = max(0.0, r_from_center - SUN_R) / 50.0
    prod_rate_norm = prod / 5.0
    phase          = step / TOTAL_STEPS
    econ_share     = min(1.0, (prod / 5.0) / max(1.0, total_prod / 5.0))

    base = [
        is_mine, is_enemy, is_neutral,
        dist_center, angle_norm,
        is_orbiting,
        ships_norm, prod_norm,
        ships_vs_mine, ships_vs_enemy,
        dist_sun,
        prod_rate_norm,
        phase,
        econ_share,
        min(1.0, radius / 3.5),
        1.0 if ships < 10 else 0.0,
        1.0 if prod >= 4 else 0.0,
        1.0 if ships > 100 else 0.0,
    ]

    # ── fleet-state features ──────────────────────────────────────────────
    arr = (arrivals or {}).get(pid, {'friendly_ships': 0, 'enemy_ships': 0,
                                      'first_threat': 999.0, 'first_friendly': 999.0})
    ts = max(1.0, total_ships)
    friendly_arr      = min(1.0, arr['friendly_ships'] / ts)
    enemy_arr         = min(1.0, arr['enemy_ships'] / ts)
    net_balance       = max(-1.0, min(1.0, (arr['friendly_ships'] - arr['enemy_ships']) / ts))
    turns_to_threat   = min(1.0, arr['first_threat'] / 100.0)
    already_covered   = 1.0 if arr['friendly_ships'] > 0 else 0.0
    contested         = 1.0 if arr['friendly_ships'] > 0 and arr['enemy_ships'] > 0 else 0.0

    return base + [friendly_arr, enemy_arr, net_balance,
                   turns_to_threat, already_covered, contested]

# ── edge features: (source_planet, target_planet) ─────────────────────────
# These are appended to describe the *relationship* between a pair of planets.

N_EDGE_FEATS = 17   # 14 base + 3 fleet-state features

def edge_features(src, tgt, player: int, step: int,
                  total_ships: float,
                  arrivals: dict = None) -> List[float]:
    """
    17 features for an (src, tgt) directed edge.
    src / tgt are raw planet lists.

    Features 0-13: base edge features (unchanged)
    Features 14-16: fleet-state features (new)
      14. src_can_spare      — (src.ships - enemy_arriving_at_src) / total_ships, [-1,1]
      15. tgt_net_balance    — (friendly - enemy) arriving at tgt / total_ships, [-1,1]
      16. src_threat_urgent  — 1 if enemy fleet hits src within 5 turns
    """
    sx, sy, sr = src[2], src[3], src[4]
    tx, ty, tr = tgt[2], tgt[3], tgt[4]

    direct_dist = _dist(sx, sy, tx, ty) / BOARD
    sun_blocked = 1.0 if _path_hits_sun(sx, sy, tx, ty) else 0.0

    mid_ships = max(1, int(src[5] * 0.5))
    raw_turns = _travel_turns(sx, sy, sr, tx, ty, tr, mid_ships)
    eta_norm  = min(1.0, (raw_turns or 999) / 100.0)

    angle = math.atan2(ty - sy, tx - sx) / math.pi

    tgt_ships = max(0.0, tgt[5])
    src_ships = max(0.0, src[5])
    strength_ratio = min(2.0, src_ships / max(1.0, tgt_ships + 1))
    delta_norm = max(-1.0, min(1.0, (src_ships - tgt_ships) / max(1.0, total_ships)))

    tgt_owner    = tgt[1]
    is_expand    = 1.0 if tgt_owner == -1 else 0.0
    is_attack    = 1.0 if (tgt_owner != -1 and tgt_owner != player) else 0.0
    is_reinforce = 1.0 if tgt_owner == player else 0.0

    remaining_turns = TOTAL_STEPS - step
    if raw_turns is not None:
        prod_value = (tgt[6] * max(0, remaining_turns - raw_turns)) / (5 * TOTAL_STEPS)
    else:
        prod_value = 0.0

    def _orb(p):
        r = math.hypot(p[2] - CENTER_X, p[3] - CENTER_Y)
        return 1.0 if (r + p[4]) < ROT_LIMIT else 0.0

    both_orbiting = _orb(src) * _orb(tgt)

    base = [
        direct_dist, sun_blocked, eta_norm,
        angle,
        min(1.0, strength_ratio / 2.0),
        delta_norm,
        is_expand, is_attack, is_reinforce,
        min(1.0, prod_value),
        both_orbiting,
        min(1.0, tgt_ships / max(1.0, total_ships)),
        min(1.0, src_ships / max(1.0, total_ships)),
        tgt[6] / 5.0,
    ]

    # ── fleet-state edge features ─────────────────────────────────────────
    ts = max(1.0, total_ships)
    src_arr = (arrivals or {}).get(src[0], {'enemy_ships': 0, 'first_threat': 999.0})
    tgt_arr = (arrivals or {}).get(tgt[0], {'friendly_ships': 0, 'enemy_ships': 0})

    src_can_spare     = max(-1.0, min(1.0, (src_ships - src_arr['enemy_ships']) / ts))
    tgt_net_balance   = max(-1.0, min(1.0, (tgt_arr['friendly_ships'] - tgt_arr['enemy_ships']) / ts))
    src_threat_urgent = 1.0 if src_arr['first_threat'] <= 5 else 0.0

    return base + [src_can_spare, tgt_net_balance, src_threat_urgent]

# ── full sample builder ──────────────────────────────────────────────────────

N_GLOBAL_FEATS   = 31
N_SRC_FEATS      = N_PLANET_FEATS       # 24
N_TGT_FEATS      = N_PLANET_FEATS       # 24
N_EDGE_FEATS_TOT = N_EDGE_FEATS         # 17
# total per edge: 31 + 24 + 24 + 17 = 96
N_TOTAL_FEATS    = N_GLOBAL_FEATS + N_SRC_FEATS + N_TGT_FEATS + N_EDGE_FEATS_TOT

def build_edge_sample(obs: dict, src_id: int, tgt_id: int,
                      arrivals: dict = None) -> Optional[List[float]]:
    """
    Build the full 96-dim feature vector for an (src, tgt) action pair.
    Pass pre-computed arrivals table for efficiency; computes it if None.
    Returns None if either planet is not found in obs.
    """
    player      = obs.get('player', 0)
    step        = obs.get('step', 0) or 0
    planets_raw = obs.get('planets', []) or []
    fleets_raw  = obs.get('fleets', []) or []

    planet_map = {p[0]: p for p in planets_raw}
    src = planet_map.get(src_id)
    tgt = planet_map.get(tgt_id)
    if src is None or tgt is None:
        return None

    total_ships = sum(max(0, p[5]) for p in planets_raw) + \
                  sum(max(0, f[6]) for f in fleets_raw) + 1.0
    total_prod  = sum(p[6] for p in planets_raw if p[1] != -1) + 1.0
    my_total    = sum(max(0, p[5]) for p in planets_raw if p[1] == player) + \
                  sum(max(0, f[6]) for f in fleets_raw if f[1] == player) + 1.0
    enemy_total = sum(max(0, p[5]) for p in planets_raw if p[1] not in (-1, player)) + \
                  sum(max(0, f[6]) for f in fleets_raw if f[1] not in (-1, player)) + 1.0

    initial_by_id = {p[0]: p for p in (obs.get('initial_planets') or [])}

    if arrivals is None:
        arrivals = fleet_arrivals_table(fleets_raw, planets_raw, player, total_ships)

    gf = global_features(obs)
    sf = planet_features(src, player, step, total_ships, total_prod,
                         initial_by_id, my_total, enemy_total, arrivals)
    tf = planet_features(tgt, player, step, total_ships, total_prod,
                         initial_by_id, my_total, enemy_total, arrivals)
    ef = edge_features(src, tgt, player, step, total_ships, arrivals)

    return gf + sf + tf + ef  # length = 96


def build_all_candidate_edges(obs: dict) -> List[Tuple[int, int, List[float]]]:
    """
    Returns all (src_id, tgt_id, features) edges that are *physically valid*
    for the current player at this step.
    Fleet arrivals table is computed once and shared across all edges.
    """
    player      = obs.get('player', 0)
    planets_raw = obs.get('planets', []) or []
    fleets_raw  = obs.get('fleets', []) or []

    total_ships = sum(max(0, p[5]) for p in planets_raw) + \
                  sum(max(0, f[6]) for f in fleets_raw) + 1.0

    # compute once per turn — O(F) not O(F×E)
    arrivals = fleet_arrivals_table(fleets_raw, planets_raw, player, total_ships)

    my_planets = [p for p in planets_raw if p[1] == player and p[5] >= 1]

    results = []
    for src in my_planets:
        for tgt in planets_raw:
            if tgt[0] == src[0]:
                continue
            feats = build_edge_sample(obs, src[0], tgt[0], arrivals)
            if feats is not None:
                results.append((src[0], tgt[0], feats))
    return results


def ship_fraction_label(obs: dict, src_id: int, ships_sent: int) -> float:
    """
    Converts raw ships_sent into a fraction of available ships at source.
    Returns None if source not found or has 0 ships.
    """
    planets_raw = obs.get('planets', []) or []
    src = next((p for p in planets_raw if p[0] == src_id), None)
    if src is None or src[5] <= 0:
        return None
    return min(1.0, ships_sent / src[5])
