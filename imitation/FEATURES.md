# Imitation Learning — Feature Reference

> **Dataset produced by:** `imitation/extract_dataset.py`  
> **Features computed by:** `imitation/features.py`  
> **Total feature dimensions:** **81** per (source-planet, target-planet) edge

---

## 1. Vector Layout

```
┌─────────────────────────────────────────────────────────────┐
│  [0 : 31]   Global board state        (31 dims)             │
│  [31 : 49]  Source planet features    (18 dims)             │
│  [49 : 67]  Target planet features    (18 dims)             │
│  [67 : 81]  Edge / relationship       (14 dims)             │
└─────────────────────────────────────────────────────────────┘
```

Everything is ego-centric (from the **winner's** perspective) and normalized to
`[0, 1]` unless noted.

---

## 2. Global Board State (indices 0–30)

These 31 features summarize the whole board from the perspective of the querying
player. They match the 30-dim input used by `classifier/value_fn_1v1.py` plus
one extra raw `n_players` hint at the end.

| Idx | Name | Range | Description |
|-----|------|-------|-------------|
| 0 | `step_norm` | [0, 1] | Current turn / 500 |
| 1 | `np2_frac` | {0.5, 1.0} | n_players / 4 (0.5 for 2-player, 1.0 for 4-player) |
| 2 | `my_planet_frac` | [0, 1] | My planets / total planets |
| 3 | `enemy_planet_frac` | [0, 1] | Best enemy's planets / total planets |
| 4 | `my_ship_frac` | [0, 1] | My planet ships / all planet ships |
| 5 | `enemy_ship_frac` | [0, 1] | Best enemy's ships / all ships |
| 6 | `my_prod_frac` | [0, 1] | My total production / total production |
| 7 | `enemy_prod_frac` | [0, 1] | Best enemy production / total production |
| 8 | `my_centrality` | [0, 1] | My avg centrality score / 60 |
| 9 | `enemy_centrality` | [0, 1] | Best enemy avg centrality / 60 |
| 10 | `ship_advantage` | [-1, 1] | (my_ships − enemy_ships) / total_ships |
| 11 | `planet_advantage` | [-1, 1] | (my_planets − enemy_planets) / total_planets |
| 12 | `prod_advantage` | [-1, 1] | (my_prod − enemy_prod) / (total_prod+1) |
| **13** | **`is_1v1`** | **{0, 1}** | **1 if n_players == 2 (1v1 game)** |
| **14** | **`is_4v4`** | **{0, 1}** | **1 if n_players == 4 (4-player game)** |
| 15 | `my_ship_ratio` | [0, ∞] | my_ships / (enemy_ships+1), capped at 2 |
| 16 | `my_max_ship_frac` | [0, 1] | Strongest single planet I own / total ships |
| 17 | `enemy_max_ship_frac` | [0, 1] | Strongest single enemy planet / total ships |
| 18 | `neutral_frac` | [0, 1] | Neutral planet count / total planets |
| 19 | `my_top3_prod_frac` | [0, 1] | Sum of my top-3 production planets / total prod |
| 20 | `enemy_top3_prod_frac` | [0, 1] | Sum of best enemy top-3 production / total prod |
| 21 | `min_enemy_dist` | [0, 1] | Min distance between my and enemy planets / 100 |
| 22 | `my_fleet_ship_frac` | [0, 1] | My in-flight ships / total army |
| 23 | `enemy_fleet_ship_frac` | [0, 1] | Enemy in-flight ships / total army |
| 24 | `my_fleet_count` | [0, 1] | My fleet count / 8 |
| 25 | `enemy_fleet_count` | [0, 1] | Enemy fleet count / 8 |
| 26 | `my_fleet_economy` | [0, 1] | My fleet ships / (my planet ships + 1) |
| 27 | `enemy_fleet_economy` | [0, 1] | Enemy fleet ships / (enemy planet ships + 1) |
| 28 | `my_prod_per_planet` | [0, 1] | My avg production / 5 |
| 29 | `phase_ship_dominance` | [0, 1] | (step/500) × (my_ship_frac) — late-game pressure |
| 30 | `raw_n_players` | {2.0, 4.0} | Raw player count (useful as a continuous signal) |

---

## 3. Source Planet Features (indices 31–48)

18 features describing the planet **the winner is launching from**.
All relative to the current player.

| Idx | Name | Range | Description |
|-----|------|-------|-------------|
| 31 | `is_mine` | {0,1} | 1 if I own this planet |
| 32 | `is_enemy` | {0,1} | 1 if an enemy owns it |
| 33 | `is_neutral` | {0,1} | 1 if neutral |
| 34 | `dist_center` | [0, ~1.4] | Distance from board center / 50 |
| 35 | `angle_norm` | [-1, 1] | Angle from center: atan2(y-50, x-50) / π |
| 36 | `is_orbiting` | {0,1} | 1 if planet orbits the sun |
| 37 | `ships_norm` | [0, 1] | Planet ships / total ships |
| 38 | `prod_norm` | [0, 1] | Production / 5 |
| 39 | `ships_vs_mine` | [0, 2] | Planet ships / my total ships |
| 40 | `ships_vs_enemy` | [0, 2] | Planet ships / best enemy total ships |
| 41 | `dist_sun` | [0, 1] | (dist_from_center − SUN_R) / 50 |
| 42 | `prod_rate_norm` | [0, 1] | Production / 5 (same as prod_norm; economy signal) |
| 43 | `phase` | [0, 1] | Step / 500 (repeated for per-planet context) |
| 44 | `econ_share` | [0, 1] | (prod/5) / (total_prod/5+1) |
| 45 | `radius_norm` | [0, 1] | Planet radius / 3.5 |
| 46 | `weak_flag` | {0,1} | 1 if ships < 10 |
| 47 | `high_prod_flag` | {0,1} | 1 if production ≥ 4 |
| 48 | `fortress_flag` | {0,1} | 1 if ships > 100 |

---

## 4. Target Planet Features (indices 49–66)

Same 18-dim structure as source planet (above), applied to the **destination planet**.
Ownership flags now reflect whether the target is mine / enemy / neutral.

| Idx | Name | Range | Description |
|-----|------|-------|-------------|
| 49 | `is_mine` | {0,1} | 1 if I already own the target |
| 50 | `is_enemy` | {0,1} | 1 if enemy owns it |
| 51 | `is_neutral` | {0,1} | 1 if neutral |
| 52 | `dist_center` | [0, ~1.4] | Target distance from board center / 50 |
| 53 | `angle_norm` | [-1, 1] | Target angle from center |
| 54 | `is_orbiting` | {0,1} | 1 if target orbits the sun |
| 55 | `ships_norm` | [0, 1] | Target ships / total ships |
| 56 | `prod_norm` | [0, 1] | Target production / 5 |
| 57 | `ships_vs_mine` | [0, 2] | Target ships / my total ships |
| 58 | `ships_vs_enemy` | [0, 2] | Target ships / best enemy total ships |
| 59 | `dist_sun` | [0, 1] | Target distance from sun (safety margin) |
| 60 | `prod_rate_norm` | [0, 1] | Target production / 5 |
| 61 | `phase` | [0, 1] | Step / 500 |
| 62 | `econ_share` | [0, 1] | Target's share of total economy |
| 63 | `radius_norm` | [0, 1] | Target radius / 3.5 |
| 64 | `weak_flag` | {0,1} | 1 if target has < 10 ships |
| 65 | `high_prod_flag` | {0,1} | 1 if target production ≥ 4 |
| 66 | `fortress_flag` | {0,1} | 1 if target has > 100 ships |

---

## 5. Edge / Relationship Features (indices 67–80)

14 features that describe the **relationship between source and target**.

| Idx | Name | Range | Description |
|-----|------|-------|-------------|
| 67 | `direct_dist_norm` | [0, 1] | Euclidean distance / 100 |
| 68 | `sun_blocked` | {0,1} | 1 if the straight-line path passes through the sun |
| 69 | `eta_norm` | [0, 1] | Estimated travel turns (at 50% ships) / 100 |
| 70 | `angle_to_target` | [-1, 1] | atan2(ty−sy, tx−sx) / π |
| 71 | `strength_ratio` | [0, 1] | src_ships / (tgt_ships+1), capped at 2 then halved |
| 72 | `delta_ships_norm` | [-1, 1] | (src_ships − tgt_ships) / total_ships |
| 73 | `is_expand` | {0,1} | 1 if target is neutral (colonisation) |
| 74 | `is_attack` | {0,1} | 1 if target is enemy (combat) |
| 75 | `is_reinforce` | {0,1} | 1 if target is mine (reinforcement) |
| 76 | `prod_value_norm` | [0, 1] | tgt_prod × remaining_turns / (5 × 500) |
| 77 | `both_orbiting` | {0,1} | 1 if both src and tgt are orbiting planets |
| 78 | `tgt_ships_frac` | [0, 1] | Target ships / total ships |
| 79 | `src_ships_frac` | [0, 1] | Source ships / total ships |
| 80 | `tgt_prod_norm` | [0, 1] | Target production / 5 |

---

## 6. How 1v1 vs 4v4 Is Incorporated

### Auto-detection in `features.py`

The number of players is **never passed as an argument**. Instead it is inferred
live from the observation:

```python
np2 = max(2, 1 + len(ed))   # count distinct enemy player IDs seen
np2 = 2 if np2 <= 2 else 4  # clamp: only 1v1 or 4-player FFA exist in the dataset
```

This means the feature extractor works identically at training time and at
game runtime without any mode flags.

### Encoding (indices 13 & 14)

```
is_1v1 = 1   is_4v4 = 0   → 2-player game
is_1v1 = 0   is_4v4 = 1   → 4-player FFA
```

The dual 1-hot lets the neural network learn completely separate response
surfaces for each game mode while sharing all other weights.

### "Best Enemy" aggregation in 4-player games

In 4-player FFA, there are up to 3 enemies. All enemy aggregates (ships,
planets, production, centrality) are taken from the **most threatening single
opponent**:

```python
best_ow = max(enemies, key=lambda e: e['ships'] + 100 * e['planets'])
```

This keeps the feature space fixed at 31 globals regardless of the number of
live opponents, and teaches the network to focus on the largest threat.

### Dataset metadata

The saved `.npz` includes a `n_players` array (2 or 4) so you can:
- Train separate models per mode
- Analyse accuracy broken down by game type
- Apply different class-balance strategies (1v1: strict 1:1; 4-player: keep
  raw 1:3 ratio since only one of four players wins)

---

## 7. Winner-Context Extraction

### Why only the winner?

Imitation learning trains on the actions of the **best available teacher**.
Using losers would teach suboptimal play; using all players creates
contradictory labels (same state, opposite actions).

### Extraction algorithm (`extract_dataset.py`)

```
For each replay:
  1. Find winner = player index with the highest final reward
     • Ties (multiple players share max reward) → skip replay
     • All non-positive rewards → skip replay

  2. For each game step t where the winner was ACTIVE:
     a. Resolve each action [src_id, angle, ships_sent] → (src_id, tgt_id)
        via perpendicular-distance collision test (matches game physics)
     b. Compute ship fraction = ships_sent / src_planet_ships

  3. Candidate edges = all (my_planet, any_other_planet) pairs with ≥1 ship

  4. Label:
     • (src, tgt) the winner took   → y_action=1, y_fraction=frac  [positive]
     • (src, tgt) winner skipped    → y_action=0, y_fraction=NaN   [negative]
     • Negatives sampled at 1:neg_ratio (default 3:1)
```

### Tie handling for winner detection

```python
max_r = max(rewards)
if rewards.count(max_r) > 1 or max_r <= 0:
    skip  # ambiguous — no unambiguous winner to imitate
winner = rewards.index(max_r)
```

This is identical to the logic in `classifier/train_from_replays.py`, ensuring
consistent winner labelling across the whole project.

---

## 8. Dataset Arrays

| Array | dtype | Shape | Description |
|-------|-------|-------|-------------|
| `X` | float32 | (N, 81) | Feature vectors (all 0 if NaN/Inf replaced) |
| `y_action` | int8 | (N,) | 1 = winner took this edge, 0 = skipped |
| `y_fraction` | float32 | (N,) | Ship fraction (NaN for negatives) |
| `step` | int16 | (N,) | Game turn this sample came from (0–500) |
| `replay` | int32 | (N,) | Replay index (use for grouped CV splits) |
| `n_players` | int8 | (N,) | Game mode: 2 = 1v1, 4 = 4-player FFA |
| `winner` | int8 | (N,) | Player index of the winner (0–3) |

**Quick inspection:**
```python
import numpy as np
d = np.load('imitation/dataset.npz')
print(d['n_players'][:20])   # 2 or 4
print(d['winner'][:20])      # 0–3
print(d['y_action'].mean())  # ~0.25 with neg_ratio=3
```

---

## 9. Training Notes

| Aspect | Setting | Reason |
|--------|---------|--------|
| **Action loss** | BCE with `pos_weight=2.0` | 3:1 neg:pos imbalance; upweight true positives |
| **Fraction loss** | MSE on positives only | Negatives have no meaningful "how many ships" |
| **Fraction weight** | 0.5 × MSE | Secondary signal; shouldn't dominate |
| **Sampler** | `WeightedRandomSampler` | Maintains ~1:1 pos:neg inside each mini-batch |
| **CV split** | By `replay` index | Prevents data leakage across steps of the same game |
| **1v1 balance** | Strict 50/50 win/loss | Equal difficulty both directions |
| **4-player balance** | Raw ~25% win / 75% loss | Reflects real odds (1 winner, 3 losers) |

---

## 10. Runtime Usage (agent_nn.py)

At inference time `build_all_candidate_edges(obs)` generates all valid
(src, tgt) pairs, the model scores them in a single batched forward pass,
and the top-K edges above the confidence threshold are committed as moves.
The exact launch angle is then computed by `main.py`'s `plan_shot()` physics
solver — the network never needs to output angles directly.

```
Observation dict
  └─► build_all_candidate_edges()  → list[(src_id, tgt_id, 81-dim vector)]
  └─► model.forward(batch)         → action_prob, fraction
  └─► filter top-K + threshold
  └─► main.py plan_shot()          → exact angle
  └─► return [[src, angle, ships], ...]
```
