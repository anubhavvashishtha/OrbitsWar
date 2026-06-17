"""
features.py
-----------
Converts a raw Orbit Wars observation into a Spatial Grid (Image) 
for the U-Net CNN architecture.

We use a 64x64 grid to minimize the chance of two planets occupying the same pixel.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional

# ── constants ────────────────────────────────────────
GRID_SIZE    = 64
BOARD        = 100.0
CENTER_X     = 50.0
CENTER_Y     = 50.0
SUN_R        = 10.0
MAX_SPEED    = 6.0
TOTAL_STEPS  = 500
ROT_LIMIT    = 50.0

def _fleet_speed(ships):
    if ships <= 1: return 1.0
    r = math.log(max(1, ships)) / math.log(1000)
    r = max(0.0, min(1.0, r))
    return 1.0 + (MAX_SPEED - 1.0) * (r ** 1.5)

def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def get_grid_coords(x: float, y: float) -> Tuple[int, int]:
    # Maps [0, 100] to [0, GRID_SIZE-1]
    gx = int(max(0, min(GRID_SIZE - 1, (x / BOARD) * GRID_SIZE)))
    gy = int(max(0, min(GRID_SIZE - 1, (y / BOARD) * GRID_SIZE)))
    return gx, gy

# ── fleet arrival table ───────────────────────────────────────────────────
def _fleet_target_planet(fleet, planets_raw):
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
        dx, dy = px - fx, py - fy
        proj = dx * dir_x + dy * dir_y
        if proj <= 0: continue
        perp_sq = dx*dx + dy*dy - proj*proj
        if perp_sq >= pr * pr: continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, pr*pr - perp_sq)))
        turns = hit_d / speed
        if turns < best_turns:
            best_turns = turns
            best_id    = p[0]
    return best_id, best_turns

def fleet_arrivals_table(fleets_raw, planets_raw, player, total_ships):
    table = {p[0]: {'friendly_ships': 0, 'enemy_ships': 0,
                    'first_threat': 999.0, 'first_friendly': 999.0}
             for p in planets_raw}
    for f in (fleets_raw or []):
        owner, ships = f[1], int(f[6])
        if ships <= 0: continue
        target_id, turns = _fleet_target_planet(f, planets_raw)
        if target_id is None or target_id not in table: continue
        row = table[target_id]
        if owner == player:
            row['friendly_ships'] += ships
            row['first_friendly'] = min(row['first_friendly'], turns)
        else:
            row['enemy_ships'] += ships
            row['first_threat'] = min(row['first_threat'], turns)
    return table

# ── SPATIAL FEATURE BUILDER ────────────────────────────────────────

def build_spatial_features(obs: dict) -> Tuple[np.ndarray, List[int], List[Tuple[int, int]]]:
    """
    Builds the (Channels, 64, 64) spatial image for the U-Net.
    Returns:
      - img: np.ndarray of shape (C, 64, 64)
      - sorted_planet_ids: strictly deterministic list of 24 planet IDs
      - planet_coords: list of (gx, gy) tuples matching the sorted planets, 
                       used by the Matrix Head to pluck embeddings.
    """
    player      = obs.get('player', 0)
    step        = obs.get('step', 0) or 0
    planets_raw = obs.get('planets', []) or []
    fleets_raw  = obs.get('fleets', []) or []

    # Calculate global states
    total_ships = sum(max(0, p[5]) for p in planets_raw) + sum(max(0, f[6]) for f in fleets_raw) + 1.0
    arrivals = fleet_arrivals_table(fleets_raw, planets_raw, player, total_ships)

    # N_CHANNELS = 14 local planet channels + 5 global channels = 19 channels
    N_CHANNELS = 19
    img = np.zeros((N_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # 1. Paint Global Channels (broadcast across entire 64x64 map)
    my_planets_cnt = sum(1 for p in planets_raw if p[1] == player)
    enemy_planets_cnt = sum(1 for p in planets_raw if p[1] not in (-1, player))
    
    img[14, :, :] = step / TOTAL_STEPS
    img[15, :, :] = my_planets_cnt / 24.0
    img[16, :, :] = enemy_planets_cnt / 24.0
    img[17, :, :] = total_ships / 10000.0
    
    # Paint the Sun in the middle
    sun_gx, sun_gy = get_grid_coords(CENTER_X, CENTER_Y)
    sun_radius_grid = int((SUN_R / BOARD) * GRID_SIZE)
    # Simple sun mask
    for dy in range(-sun_radius_grid, sun_radius_grid+1):
        for dx in range(-sun_radius_grid, sun_radius_grid+1):
            if dx*dx + dy*dy <= sun_radius_grid*sun_radius_grid:
                nx, ny = sun_gx + dx, sun_gy + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    img[18, ny, nx] = 1.0

    # 2. Sort Planets for Deterministic Plucking
    sorted_planets = sorted(planets_raw, key=lambda p: p[0])
    planet_ids = []
    planet_coords = []

    # 3. Paint Planet Channels
    for p in sorted_planets:
        pid, owner, x, y, radius, ships, prod = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
        gx, gy = get_grid_coords(x, y)
        planet_ids.append(pid)
        planet_coords.append((gx, gy))

        # Ownership
        if owner == player:
            img[0, gy, gx] = 1.0
        elif owner != -1:
            img[1, gy, gx] = 1.0
        else:
            img[2, gy, gx] = 1.0

        # Economy & Size
        img[3, gy, gx] = ships / max(1.0, total_ships)
        img[4, gy, gx] = prod / 5.0
        img[5, gy, gx] = radius / 3.5

        # Arrival Ledger
        arr = arrivals.get(pid, {'friendly_ships': 0, 'enemy_ships': 0, 'first_threat': 999.0})
        img[6, gy, gx] = arr['friendly_ships'] / total_ships
        img[7, gy, gx] = arr['enemy_ships'] / total_ships
        img[8, gy, gx] = 1.0 if arr['first_threat'] <= 5 else 0.0

    # 4. Paint In-Flight Fleets
    for f in fleets_raw:
        owner = f[1]
        x, y = float(f[2]), float(f[3])
        ships = int(f[6])
        gx, gy = get_grid_coords(x, y)
        
        if owner == player:
            img[9, gy, gx] += ships / total_ships
            img[10, gy, gx] = 1.0  # Fleet presence mask
        elif owner != -1:
            img[11, gy, gx] += ships / total_ships
            img[12, gy, gx] = 1.0  # Enemy fleet presence
            
        # Paint the direction vector (Cosine and Sine)
        angle = float(f[4])
        img[13, gy, gx] = math.cos(angle)

    return img, planet_ids, planet_coords

