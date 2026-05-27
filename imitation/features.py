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

# ── per-planet features (relative to player) ─────────────────────────────
# These are the features used to represent each planet as a *node*.

N_PLANET_FEATS = 18

def planet_features(p, player: int, step: int,
                    total_ships: float, total_prod: float,
                    initial_by_id: dict,
                    my_total_ships: float,
                    enemy_total_ships: float) -> List[float]:
    """
    18 features for a single planet, ego-centric w.r.t. `player`.
    p is a raw planet list: [id, owner, x, y, radius, ships, production]
    """
    pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]

    # ownership encoding: 1-hot [mine, enemy, neutral]
    is_mine    = 1.0 if owner == player else 0.0
    is_enemy   = 1.0 if (owner != player and owner != -1) else 0.0
    is_neutral = 1.0 if owner == -1 else 0.0

    # position: distance from center, quadrant angle normalized
    cx = x - CENTER_X
    cy = y - CENTER_Y
    dist_center = math.hypot(cx, cy) / 50.0         # [0, ~1.4]
    angle_norm  = math.atan2(cy, cx) / math.pi      # [-1, 1]

    # orbiting vs static
    r_from_center = math.hypot(cx, cy)
    is_orbiting = 1.0 if (r_from_center + radius) < ROT_LIMIT else 0.0

    # economic
    ships_norm = min(1.0, ships / max(1.0, total_ships))
    prod_norm  = prod / 5.0                           # max prod is 5

    # relative military value
    ships_vs_mine  = min(2.0, ships / max(1.0, my_total_ships))
    ships_vs_enemy = min(2.0, ships / max(1.0, enemy_total_ships))

    # distance from sun
    dist_sun = max(0.0, r_from_center - SUN_R) / 50.0

    # time-to-produce: how many turns until this planet generates N ships
    # normalized; useful for deciding whether to capture or wait
    prod_rate_norm = prod / 5.0

    # game phase features
    phase = step / TOTAL_STEPS

    # remaining ships fraction this planet contributes to total economy
    econ_share = min(1.0, (prod / 5.0) / max(1.0, total_prod / 5.0))

    return [
        is_mine, is_enemy, is_neutral,
        dist_center, angle_norm,
        is_orbiting,
        ships_norm, prod_norm,
        ships_vs_mine, ships_vs_enemy,
        dist_sun,
        prod_rate_norm,
        phase,
        econ_share,
        min(1.0, radius / 3.5),      # size signal
        1.0 if ships < 10 else 0.0,  # "weak" flag
        1.0 if prod >= 4 else 0.0,   # "high value" flag
        1.0 if ships > 100 else 0.0, # "fortress" flag
    ]

# ── edge features: (source_planet, target_planet) ─────────────────────────
# These are appended to describe the *relationship* between a pair of planets.

N_EDGE_FEATS = 14

def edge_features(src, tgt, player: int, step: int,
                  total_ships: float) -> List[float]:
    """
    14 features for an (src, tgt) directed edge.
    src / tgt are raw planet lists.
    """
    sx, sy, sr = src[2], src[3], src[4]
    tx, ty, tr = tgt[2], tgt[3], tgt[4]

    direct_dist = _dist(sx, sy, tx, ty) / BOARD
    sun_blocked = 1.0 if _path_hits_sun(sx, sy, tx, ty) else 0.0

    # rough ETA with 50% of source ships
    mid_ships = max(1, int(src[5] * 0.5))
    raw_turns = _travel_turns(sx, sy, sr, tx, ty, tr, mid_ships)
    eta_norm   = min(1.0, (raw_turns or 999) / 100.0)

    # angle from source to target
    angle = math.atan2(ty - sy, tx - sx) / math.pi   # [-1, 1]

    # relative strength: how many ships we'd need vs what we have
    tgt_ships = max(0.0, tgt[5])
    src_ships  = max(0.0, src[5])
    strength_ratio = min(2.0, src_ships / max(1.0, tgt_ships + 1))

    # ship delta normalized
    delta_norm = (src_ships - tgt_ships) / max(1.0, total_ships)
    delta_norm = max(-1.0, min(1.0, delta_norm))

    # is this an expansion (neutral), attack (enemy), or reinforce (mine)?
    tgt_owner   = tgt[1]
    is_expand   = 1.0 if tgt_owner == -1 else 0.0
    is_attack   = 1.0 if (tgt_owner != -1 and tgt_owner != player) else 0.0
    is_reinforce = 1.0 if tgt_owner == player else 0.0

    # production-value of edge (prod * remaining turns) normalized
    remaining_turns = (TOTAL_STEPS - step)
    if raw_turns is not None:
        prod_value = (tgt[6] * max(0, remaining_turns - raw_turns)) / (5 * TOTAL_STEPS)
    else:
        prod_value = 0.0

    # orbital complexity: both orbiting = harder to aim
    def _orb(p):
        r = math.hypot(p[2] - CENTER_X, p[3] - CENTER_Y)
        return 1.0 if (r + p[4]) < ROT_LIMIT else 0.0

    both_orbiting = _orb(src) * _orb(tgt)

    return [
        direct_dist, sun_blocked, eta_norm,
        angle,
        min(1.0, strength_ratio / 2.0),
        delta_norm,
        is_expand, is_attack, is_reinforce,
        min(1.0, prod_value),
        both_orbiting,
        min(1.0, tgt_ships / max(1.0, total_ships)),
        min(1.0, src_ships / max(1.0, total_ships)),
        tgt[6] / 5.0,   # target production, normalized
    ]

# ── full sample builder ──────────────────────────────────────────────────────

N_GLOBAL_FEATS   = 31
N_SRC_FEATS      = N_PLANET_FEATS       # 18
N_TGT_FEATS      = N_PLANET_FEATS       # 18
N_EDGE_FEATS_TOT = N_EDGE_FEATS         # 14
# total per edge: 31 + 18 + 18 + 14 = 81
N_TOTAL_FEATS    = N_GLOBAL_FEATS + N_SRC_FEATS + N_TGT_FEATS + N_EDGE_FEATS_TOT

def build_edge_sample(obs: dict, src_id: int, tgt_id: int) -> Optional[List[float]]:
    """
    Build the full feature vector for an (src, tgt) action pair.
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

    gf = global_features(obs)
    sf = planet_features(src, player, step, total_ships, total_prod,
                         initial_by_id, my_total, enemy_total)
    tf = planet_features(tgt, player, step, total_ships, total_prod,
                         initial_by_id, my_total, enemy_total)
    ef = edge_features(src, tgt, player, step, total_ships)

    return gf + sf + tf + ef  # length = 81


def build_all_candidate_edges(obs: dict) -> List[Tuple[int, int, List[float]]]:
    """
    Returns all (src_id, tgt_id, features) edges that are *physically valid*
    for the current player at this step.
    src must be owned by the player and have ≥1 ship.
    tgt must not be owned by the player (attack/expand) OR be owned by the
    player but have fewer ships than a threatened planet (reinforce).
    Sun-blocked paths are included (the edge feature flags it) so the
    network can learn to avoid them; filtering is optional.
    """
    player      = obs.get('player', 0)
    planets_raw = obs.get('planets', []) or []

    my_planets  = [p for p in planets_raw if p[1] == player and p[5] >= 1]
    all_planets = planets_raw

    results = []
    for src in my_planets:
        for tgt in all_planets:
            if tgt[0] == src[0]:
                continue
            feats = build_edge_sample(obs, src[0], tgt[0])
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
