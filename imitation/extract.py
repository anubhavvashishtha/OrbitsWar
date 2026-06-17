import json
import math
import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Adjust imports to local path
sys.path.insert(0, str(Path(__file__).parent))
from features import build_spatial_features, NUM_FRACTION_BINS, MAX_PLANETS

NUM_FRACTION_BINS = 6  # 0 to 5

def _find_target(obs: dict, src_id: int, angle: float) -> Optional[int]:
    planets = obs.get('planets', []) or []
    src = next((p for p in planets if p[0] == src_id), None)
    if src is None:
        return None
    sx, sy = src[2], src[3]
    dx, dy = math.cos(angle), math.sin(angle)

    best_id, best_perp = None, 1e9
    for p in planets:
        if p[0] == src_id:
            continue
        tx, ty, tr = p[2], p[3], p[4]
        rx, ry = tx - sx, ty - sy
        dot = rx * dx + ry * dy
        if dot <= 0:
            continue
        perp = abs(rx * dy - ry * dx)
        if perp < tr + 1.0 and perp < best_perp:
            best_perp = perp
            best_id = p[0]
    return best_id

def extract_replay(replay_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty = (
        np.zeros((0, 19, 64, 64), dtype=np.float32),
        np.zeros((0, 24, 2), dtype=np.int16),
        np.zeros((0, 24, 24), dtype=np.int16),
        np.zeros((0, 24), dtype=bool)
    )

    try:
        with open(replay_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return empty

    rewards = data.get('rewards', []) or []
    steps = data.get('steps', []) or []

    rewards_num = [r for r in rewards if r is not None]
    if not rewards_num:
        return empty

    max_r = max(rewards_num)
    if rewards_num.count(max_r) > 1 or max_r <= 0:
        return empty

    winner = rewards.index(max_r)
    
    # Filter by Top Players (Elite behavioral cloning)
    TOP_PLAYERS = {
        "Isaiah @ Tufa Labs", "Vadasz", "bowwowforeach", "typeIIIfairy", 
        "3Comets", "Jake Will", "saharan", "Maruichi01", "Audun Ljone Henriksen", 
        "skalermo", "寿!", "Vincent Schuler", "Harm Buisman", "Kh0a", "Viltrum Empire"
    }
    
    team_names = data.get("info", {}).get("TeamNames", [])
    if team_names and winner < len(team_names):
        if team_names[winner] not in TOP_PLAYERS:
            return empty

    X_list = []
    Coords_list = []
    Y_matrix_list = []
    Y_mask_list = []

    for step_idx, step in enumerate(steps[1:], 1):
        if winner >= len(step):
            continue
        player_step = step[winner]
        obs = player_step.get('observation', {})
        actions = player_step.get('action') or []

        if not obs or not obs.get('planets'):
            continue

        # 1. Build Spatial Features
        img, planet_ids, planet_coords = build_spatial_features(obs)
        if len(planet_ids) != 24:
            continue  # Must be exactly 24 planets for standard Orbit Wars
            
        # 2. Build Y matrix and Mask
        y_mat = np.zeros((24, 24), dtype=np.int16)
        y_mask = np.zeros(24, dtype=bool)
        
        # Populate Mask (True if planet belongs to winner)
        planets_raw = obs.get('planets', [])
        for p in planets_raw:
            if p[1] == winner:
                idx = planet_ids.index(p[0])
                y_mask[idx] = True

        # Populate Actions into Y matrix
        for act in actions:
            src_id, angle, ships_sent = act[0], act[1], act[2]
            tgt_id = _find_target(obs, src_id, angle)
            if tgt_id is None:
                continue
                
            src_planet = next((p for p in planets_raw if p[0] == src_id), None)
            if src_planet is None or src_planet[5] <= 0:
                continue
                
            fraction = min(1.0, ships_sent / src_planet[5])
            bin_idx = int(fraction * (NUM_FRACTION_BINS - 1))
            
            src_idx = planet_ids.index(src_id)
            tgt_idx = planet_ids.index(tgt_id)
            y_mat[src_idx, tgt_idx] = bin_idx
            
        X_list.append(img)
        Coords_list.append(planet_coords)
        Y_matrix_list.append(y_mat)
        Y_mask_list.append(y_mask)

    if not X_list:
        return empty

    X = np.stack(X_list, axis=0)
    Coords = np.array(Coords_list, dtype=np.int16)
    Y = np.stack(Y_matrix_list, axis=0)
    Mask = np.stack(Y_mask_list, axis=0)

    return X, Coords, Y, Mask

if __name__ == "__main__":
    # Determine Environment Paths
    if os.path.exists("/kaggle/input"):
        print("Kaggle Environment Detected!")
        input_dir = Path("/kaggle/input")
        output_dir = Path("/kaggle/working")
    else:
        print("Local Environment Detected!")
        input_dir = Path(__file__).parent.parent / "records"
        output_dir = Path(__file__).parent

    replays = list(input_dir.rglob("*.json"))
    print(f"Found {len(replays)} replays for extraction.")

    BATCH_SIZE = 500
    chunk_idx = 0
    X_acc, C_acc, Y_acc, M_acc = [], [], [], []

    for i, path in enumerate(replays):
        x, c, y, m = extract_replay(str(path))
        if x.shape[0] > 0:
            X_acc.append(x)
            C_acc.append(c)
            Y_acc.append(y)
            M_acc.append(m)

        if len(X_acc) >= BATCH_SIZE or i == len(replays) - 1:
            if not X_acc:
                continue
                
            X_batch = np.concatenate(X_acc, axis=0)
            C_batch = np.concatenate(C_acc, axis=0)
            Y_batch = np.concatenate(Y_acc, axis=0)
            M_batch = np.concatenate(M_acc, axis=0)
            
            chunk_path = output_dir / f"dataset_chunk_{chunk_idx:04d}.npz"
            np.savez_compressed(chunk_path, X=X_batch, Coords=C_batch, Y=Y_batch, Mask=M_batch)
            print(f"Saved {chunk_path} with {len(X_batch)} steps.")
            
            chunk_idx += 1
            X_acc, C_acc, Y_acc, M_acc = [], [], [], []
            gc.collect()

    print("Extraction Complete.")
