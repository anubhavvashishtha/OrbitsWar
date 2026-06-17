import json
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# Adjust imports to local path
sys.path.insert(0, str(Path(__file__).parent))
from features import global_features, planet_features, fleet_arrivals_table, ship_fraction_label

# Model Constants
MAX_PLANETS = 200
NUM_FRACTION_BINS = 10
VOCAB_SIZE = MAX_PLANETS + NUM_FRACTION_BINS + 2
END_TOKEN = VOCAB_SIZE - 2
START_TOKEN = VOCAB_SIZE - 1
MAX_SEQ_LEN = 60

RECORDS_DIR = Path(__file__).parent.parent / "records"
DATASET_PATH = Path(__file__).parent / "dataset.npz"

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
        np.zeros((0, MAX_PLANETS, 55), dtype=np.float32),
        np.zeros((0, MAX_PLANETS), dtype=bool),
        np.zeros((0, MAX_SEQ_LEN), dtype=np.int16),
        np.zeros((0, MAX_PLANETS), dtype=np.int16)
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

    X_list = []
    Mask_list = []
    Y_list = []
    PidMap_list = []

    for step_idx, step in enumerate(steps[1:], 1):
        if winner >= len(step):
            continue
        player_step = step[winner]
        obs = player_step.get('observation', {})
        actions = player_step.get('action') or []

        if not obs or not obs.get('planets'):
            continue

        player = obs.get('player', 0)
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
            pf = planet_features(p, player, step_idx, total_ships, total_prod, initial_by_id, my_total, enemy_total, arrivals)
            feats_list.append(pf + gf)
            pid_map.append(p[0])
            
        num_planets = len(feats_list)
        if num_planets == 0:
            continue
            
        padded_feats = np.zeros((MAX_PLANETS, 55), dtype=np.float32)
        mask = np.ones(MAX_PLANETS, dtype=bool)
        padded_pid_map = np.zeros(MAX_PLANETS, dtype=np.int16) - 1
        
        actual_len = min(num_planets, MAX_PLANETS)
        padded_feats[:actual_len] = np.array(feats_list[:actual_len])
        mask[:actual_len] = False
        padded_pid_map[:actual_len] = np.array(pid_map[:actual_len])
        
        # Build action sequence
        # actions is list of [src_id, angle, ships]
        valid_actions = []
        for act in actions:
            if len(act) != 3: continue
            src_id, angle, ships = act
            tgt_id = _find_target(obs, src_id, angle)
            if tgt_id is None: continue
            valid_actions.append((src_id, tgt_id, ships))
            
        # Sort actions by ships sent (descending)
        valid_actions.sort(key=lambda x: -x[2])
        
        seq = [START_TOKEN]
        for src_id, tgt_id, ships in valid_actions:
            if len(seq) + 3 >= MAX_SEQ_LEN:
                break
            
            # Find index in pid_map
            try:
                src_tok = pid_map.index(src_id)
                tgt_tok = pid_map.index(tgt_id)
            except ValueError:
                continue
                
            fraction = ship_fraction_label(obs, src_id, ships)
            if fraction is None: continue
            
            frac_bin = min(NUM_FRACTION_BINS - 1, int(fraction * NUM_FRACTION_BINS))
            frac_tok = MAX_PLANETS + frac_bin
            
            seq.extend([tgt_tok, src_tok, frac_tok])
            
        seq.append(END_TOKEN)
        
        padded_seq = np.full(MAX_SEQ_LEN, END_TOKEN, dtype=np.int16)
        padded_seq[:len(seq)] = seq
        
        X_list.append(padded_feats)
        Mask_list.append(mask)
        Y_list.append(padded_seq)
        PidMap_list.append(padded_pid_map)
        
    if not X_list:
        return empty
        
    return (
        np.array(X_list, dtype=np.float32),
        np.array(Mask_list, dtype=bool),
        np.array(Y_list, dtype=np.int16),
        np.array(PidMap_list, dtype=np.int16)
    )

if __name__ == "__main__":
    import shutil
    
    replays = list(RECORDS_DIR.glob("*.json"))
    print(f"Found {len(replays)} total replays in directory.")
    
    META_PATH = Path(__file__).parent / "dataset_meta.json"
    CHUNKS_DIR = Path(__file__).parent / "_chunks"
    CHUNKS_DIR.mkdir(exist_ok=True)
    
    processed = []
    if META_PATH.exists():
        with open(META_PATH, 'r') as f:
            processed = json.load(f)
    processed_set = set(processed)
    
    unprocessed = [p for p in replays if p.name not in processed_set]
    print(f"Already processed {len(processed)}. Remaining: {len(unprocessed)}")
    
    BATCH_SIZE = 500
    
    all_X = []
    all_Mask = []
    all_Y = []
    all_PidMap = []
    current_batch_count = 0
    
    for i, p in enumerate(unprocessed):
        x, m, y, pm = extract_replay(str(p))
        if len(x) > 0:
            all_X.append(x)
            all_Mask.append(m)
            all_Y.append(y)
            all_PidMap.append(pm)
            
        processed.append(p.name)
        current_batch_count += 1
        
        if (i+1) % 50 == 0:
            print(f"Processed {i+1} un-processed replays...")
            
        if current_batch_count >= BATCH_SIZE or i == len(unprocessed) - 1:
            if all_X:
                chunk_id = len(list(CHUNKS_DIR.glob("*.npz")))
                chunk_path = CHUNKS_DIR / f"chunk_{chunk_id:04d}.npz"
                np.savez_compressed(
                    chunk_path,
                    X=np.concatenate(all_X, axis=0),
                    Mask=np.concatenate(all_Mask, axis=0),
                    Y=np.concatenate(all_Y, axis=0),
                    PidMap=np.concatenate(all_PidMap, axis=0)
                )
                print(f"Saved {chunk_path} with data from {len(all_X)} valid replays.")
                
            with open(META_PATH, 'w') as f:
                json.dump(processed, f)
                
            all_X, all_Mask, all_Y, all_PidMap = [], [], [], []
            current_batch_count = 0
            
    print("All replays processed. Merging chunks...")
    chunks = sorted(list(CHUNKS_DIR.glob("*.npz")))
    if not chunks:
        print("No chunks to merge.")
    else:
        merged_X, merged_Mask, merged_Y, merged_PidMap = [], [], [], []
        for c in chunks:
            data = np.load(c)
            merged_X.append(data['X'])
            merged_Mask.append(data['Mask'])
            merged_Y.append(data['Y'])
            merged_PidMap.append(data['PidMap'])
            
        final_X = np.concatenate(merged_X, axis=0)
        final_Mask = np.concatenate(merged_Mask, axis=0)
        final_Y = np.concatenate(merged_Y, axis=0)
        final_PidMap = np.concatenate(merged_PidMap, axis=0)
        
        print(f"Saving final dataset with {len(final_X)} steps to {DATASET_PATH}")
        np.savez_compressed(
            DATASET_PATH,
            X=final_X,
            Mask=final_Mask,
            Y=final_Y,
            PidMap=final_PidMap
        )
        
        # Cleanup chunks
        shutil.rmtree(CHUNKS_DIR)
        print("Cleanup complete. Dataset ready!")
