import json

nb = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

def add_code(source):
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

add_code("""# CELL 1 — Install orbit_wars environment
%pip install --upgrade "kaggle-environments>=1.28.0"
from kaggle_environments import make
env = make('orbit_wars', configuration={'episodeSteps': 5})
print('orbit_wars OK')""")

add_code("""# CELL 2 — Dataset paths
from pathlib import Path

KAGGLE_USER       = "anubhav123412"
SRC_DATASET       = "orbitswar-src"
RL_DATASET        = "orbitswar-rl"

BASE = f"/kaggle/input/datasets/{KAGGLE_USER}"
SRC_DIR  = Path(f"{BASE}/{SRC_DATASET}")
RL_DIR   = Path(f"{BASE}/{RL_DATASET}")

print(f"SRC  exists: {SRC_DIR.exists()}")
print(f"RL   exists: {RL_DIR.exists()}")""")

add_code("""# CELL 3 — Copy source tree to /kaggle/working/
import os, sys, shutil
WORK = Path('/kaggle/working')

for item in SRC_DIR.iterdir():
    dst = WORK / item.name
    if item.is_dir():
        if dst.exists(): shutil.rmtree(dst)
        shutil.copytree(item, dst)
    else:
        shutil.copy2(item, dst)

rl_dst = WORK / 'rl'
rl_dst.mkdir(parents=True, exist_ok=True)
if RL_DIR.exists():
    for name in ['transformer_best.pt', 'transformer_history.json']:
        if (RL_DIR / name).exists():
            shutil.copy2(RL_DIR / name, rl_dst / name)

for p in [str(WORK), str(WORK/'rl'), str(WORK/'imitation')]:
    if p not in sys.path:
        sys.path.insert(0, p)
""")

add_code("""# CELL 4 — 4v4 Combinations Evaluation
import itertools
import random
import torch
from kaggle_environments import make
import sys

# 1. Load your agent
import rl.transformer_trainer as TT
from imitation.model import EdgeTransformerNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
actor = EdgeTransformerNet().to(device)
ckpt_path = WORK / 'rl' / 'transformer_best.pt'
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt['actor'])
    print(f"Loaded {ckpt_path.name} (Round {ckpt.get('round', '?')})")
actor.eval()
my_agent = TT.create_snapshot_agent(actor, device)

# 2. Find all 15 enemies
enemy_dir = WORK / 'enemy agents'
enemy_paths = list(enemy_dir.glob('*/main.py'))
enemies = {p.parent.name: str(p) for p in enemy_paths}
enemy_names = sorted(list(enemies.keys()))

print(f"Found {len(enemy_names)} enemies:")
print(enemy_names)

# 3. Generate 15C3 combinations
combs = list(itertools.combinations(enemy_names, 3))
print(f"Total unique 3-enemy combinations: {len(combs)}")

import csv
import os
results = []
GAMES_PER_COMB = 5

already_evaluated = set()
if os.path.exists('4v4_results.csv'):
    with open('4v4_results.csv', 'r') as f:
        reader = csv.reader(f)
        try:
            next(reader) # skip header
        except StopIteration:
            pass
        for row in reader:
            if len(row) >= 3:
                already_evaluated.add(tuple(sorted([row[0], row[1], row[2]])))

print(f"Found {len(already_evaluated)} combinations already evaluated in CSV.")

mode = 'a' if os.path.exists('4v4_results.csv') else 'w'
csv_file = open('4v4_results.csv', mode, newline='')
csv_writer = csv.writer(csv_file)
if mode == 'w':
    csv_writer.writerow(['Enemy_1', 'Enemy_2', 'Enemy_3', 'Wins', 'Total_Games', 'Win_Rate'])

env = make('orbit_wars')

for idx, comb in enumerate(combs):
    if tuple(sorted(comb)) in already_evaluated:
        print(f"[{idx+1}/{len(combs)}] Skipping {comb[0]} + {comb[1]} + {comb[2]} (already done)")
        continue
        
    wins = 0
    for g in range(GAMES_PER_COMB):
        # Build agents list: [my_agent, e1, e2, e3]
        agents = [my_agent, enemies[comb[0]], enemies[comb[1]], enemies[comb[2]]]
        
        # We shuffle the seat order to be fair
        seats = list(range(4))
        random.shuffle(seats)
        shuffled_agents = [None] * 4
        for orig_idx, new_idx in enumerate(seats):
            shuffled_agents[new_idx] = agents[orig_idx]
        
        my_seat = seats[0]
        
        steps = env.run(shuffled_agents)
        final_state = steps[-1]
        
        # Check if we won
        rewards = [float(s.reward or 0) for s in final_state]
        max_r = max(rewards)
        if max_r > 0 and rewards[my_seat] == max_r:
            wins += 1
            
    win_rate = wins / GAMES_PER_COMB
    results.append({
        'combination': comb,
        'wins': wins,
        'win_rate': win_rate
    })
    csv_writer.writerow([comb[0], comb[1], comb[2], wins, GAMES_PER_COMB, f"{win_rate:.4f}"])
    csv_file.flush()
    print(f"[{idx+1}/{len(combs)}] vs {comb[0]} + {comb[1]} + {comb[2]} -> Won {wins}/{GAMES_PER_COMB} ({win_rate*100:.0f}%)")

csv_file.close()

# 4. Print Summary
print("\\n\\n=== EVALUATION COMPLETE ===")
results.sort(key=lambda x: x['win_rate'], reverse=True)

print("\\nTop 10 easiest combinations:")
for r in results[:10]:
    print(f"{r['combination']}: {r['wins']}/5 ({r['win_rate']*100:.1f}%)")

print("\\nTop 10 hardest combinations:")
for r in results[-10:]:
    print(f"{r['combination']}: {r['wins']}/5 ({r['win_rate']*100:.1f}%)")
""")

with open('check_enemy_4v4.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
