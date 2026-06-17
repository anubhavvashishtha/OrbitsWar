from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from features import global_features, planet_features, fleet_arrivals_table
from model import AutoregressiveOrbitModel, MAX_PLANETS, NUM_FRACTION_BINS, START_TOKEN, END_TOKEN

# Import the clean physics WorldModel we preserved
try:
    from physics import WorldModel
except ImportError:
    pass

_CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'
_MODEL_PATH = os.environ.get('ORBIT_MODEL_PATH', str(_CHECKPOINT_DIR / 'autoregressive_best.pt'))

_model: Optional[AutoregressiveOrbitModel] = None
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _get_model():
    global _model
    if _model is not None:
        return _model

    if not Path(_MODEL_PATH).exists():
        print(f"[agent_ar] No checkpoint found at {_MODEL_PATH}")
        return None

    try:
        _model = AutoregressiveOrbitModel(in_features=55) # 24 planet + 31 global
        _model.load_state_dict(torch.load(_MODEL_PATH, map_location=_device))
        _model.to(_device)
        _model.eval()
        return _model
    except Exception as e:
        print(f"[agent_ar] Error loading model: {e}")
        return None

def build_encoder_input(obs: dict) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    player = obs.get('player', 0)
    step = obs.get('step', 0)
    planets_raw = obs.get('planets', [])
    fleets_raw = obs.get('fleets', [])
    
    total_ships = sum(max(0, p[5]) for p in planets_raw) + sum(max(0, f[6]) for f in fleets_raw) + 1.0
    total_prod = sum(p[6] for p in planets_raw if p[1] != -1) + 1.0
    my_total = sum(max(0, p[5]) for p in planets_raw if p[1] == player) + sum(max(0, f[6]) for f in fleets_raw if f[1] == player) + 1.0
    enemy_total = sum(max(0, p[5]) for p in planets_raw if p[1] not in (-1, player)) + sum(max(0, f[6]) for f in fleets_raw if f[1] not in (-1, player)) + 1.0
    
    initial_by_id = {p[0]: p for p in (obs.get('initial_planets') or [])}
    arrivals = fleet_arrivals_table(fleets_raw, planets_raw, player, total_ships)
    
    gf = global_features(obs)
    
    feats_list = []
    pid_map = []
    
    for p in planets_raw:
        pf = planet_features(p, player, step, total_ships, total_prod, initial_by_id, my_total, enemy_total, arrivals)
        # Concatenate planet features (24) with global features (31) = 55
        feats_list.append(pf + gf)
        pid_map.append(p[0])
        
    num_planets = len(feats_list)
    padded_feats = np.zeros((MAX_PLANETS, 55), dtype=np.float32)
    mask = np.ones(MAX_PLANETS, dtype=bool)
    
    if num_planets > 0:
        actual_len = min(num_planets, MAX_PLANETS)
        padded_feats[:actual_len] = np.array(feats_list[:actual_len])
        mask[:actual_len] = False
        
    x = torch.from_numpy(padded_feats).unsqueeze(0).to(_device) # (1, P, 55)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(_device) # (1, P)
    
    return x, mask_t, pid_map

def agent(obs, config=None):
    model = _get_model()
    if model is None:
        return []

    x, mask, pid_map = build_encoder_input(obs)
    
    # We need the world model for exact intercept angles
    try:
        player = obs.get('player', 0)
        step = obs.get('step', 0)
        from physics import Planet, Fleet
        planets = [Planet(*p) for p in obs.get('planets', [])]
        fleets = [Fleet(*f) for f in obs.get('fleets', [])]
        initial_by_id = {p[0]: Planet(*p) for p in obs.get('initial_planets', [])}
        ang_vel = config.get("angular_velocity", 0.0) if config else 0.0
        comets = obs.get("comets", [])
        comet_ids = set()
        for g in comets:
            comet_ids.update(g.get("planet_ids", []))
            
        world = WorldModel(player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids)
    except Exception as e:
        print(f"[agent_ar] Error building WorldModel: {e}")
        world = None

    with torch.no_grad():
        generated = model.generate(x, max_len=60, src_mask=mask)
        
    tokens = generated[0].cpu().numpy().tolist()
    
    # Tokens follow: [START, Target, Source, Fraction, Target, Source, Fraction, ..., END]
    # START is at tokens[0]
    tokens = tokens[1:]
    
    moves = []
    spent = defaultdict(int)
    
    planet_raw_map = {p[0]: p for p in obs.get('planets', [])}
    
    for i in range(0, len(tokens) - 2, 3):
        tgt_tok = tokens[i]
        if tgt_tok == END_TOKEN:
            break
        src_tok = tokens[i+1]
        frac_tok = tokens[i+2]
        
        if tgt_tok == END_TOKEN or src_tok == END_TOKEN or frac_tok == END_TOKEN:
            break
            
        # Validate tokens
        if tgt_tok >= MAX_PLANETS or src_tok >= MAX_PLANETS:
            continue
        if frac_tok < MAX_PLANETS or frac_tok >= MAX_PLANETS + NUM_FRACTION_BINS:
            continue
            
        # The tokens map back to array indices, which map to planet IDs
        if tgt_tok >= len(pid_map) or src_tok >= len(pid_map):
            continue
            
        tgt_id = pid_map[tgt_tok]
        src_id = pid_map[src_tok]
        
        fraction_idx = frac_tok - MAX_PLANETS
        fraction = (fraction_idx + 1) / NUM_FRACTION_BINS # 0 -> 10%, 9 -> 100%
        
        src_planet = planet_raw_map.get(src_id)
        if src_planet is None or src_planet[1] != obs.get('player', 0):
            continue
            
        available = int(src_planet[5]) - spent[src_id]
        if available <= 0:
            continue
            
        send = max(1, int(available * fraction))
        
        if world is not None:
            aim = world.plan_shot(src_id, tgt_id, send)
            if aim is None:
                continue
            angle = aim[0]
        else:
            tgt_planet = planet_raw_map.get(tgt_id)
            angle = math.atan2(tgt_planet[3] - src_planet[3], tgt_planet[2] - src_planet[2])
            
        spent[src_id] += send
        moves.append([src_id, float(angle), send])

    return moves
