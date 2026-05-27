# Orbit Wars — Imitation Learning System

Replaces the rule-based mission scorer in `main.py` with a neural network
trained from top-10 replay data. Physics (ballistics, intercept, combat sim)
from `main.py` are kept 100% intact.

---

## Files

| File | Purpose |
|------|---------|
| `features.py` | Feature extraction — converts raw obs to 81-dim edge vectors |
| `extract_dataset.py` | Parses replay JSONs → labeled `dataset.npz` |
| `model.py` | Neural net (ResNet MLP), training loop, checkpointing |
| `agent_nn.py` | Drop-in agent that uses the trained model + main.py physics |

---

## Architecture — EdgeTransformerNet

```
Per turn:
  WorldModel (main.py)              ← keeps all physics/sim
       ↓
  build_all_candidate_edges()       ← every (my_planet, any_planet) pair
       ↓                               returns list of (src_id, tgt_id, 81-dim vector)
  EdgeTransformerNet                ← scores ALL edges JOINTLY
    ├── Per-component projectors
    │     global(31)→64  src(18)→64  tgt(18)→64  edge(14)→64
    │     fuse 256 → d_model(128)
    ├── TransformerEncoder (3 layers, 4 heads, Pre-LN)
    │     every edge (A→X) attends to every other edge (B→Y)
    ├── action_head  → "should I take this edge?"   (BCE loss)
    └── fraction_head→ "how many ships to send?"    (MSE loss, positives only)
       ↓
  Top-K edges by action_prob (above threshold 0.35)
       ↓
  world.plan_shot()                 ← exact angle from main.py physics
       ↓
  [src_id, angle, ships]
```

**Why joint scoring matters:** edge (A→X) can see that (A→Y) is also being
considered — the transformer learns to not double-commit from planet A, and
to coordinate which fleet goes where in a single turn.

### Feature vector (81 dims per edge)
- `[0:31]`  Global board state (31 dims, includes `is_1v1` and `is_4v4` 1-hot)
- `[31:49]` Source planet features (18 dims, ego-centric + normalized)
- `[49:67]` Target planet features (18 dims)
- `[67:81]` Edge features (14 dims: distance, ETA, strength ratio, sun block, etc.)

See `FEATURES.md` for the full per-index breakdown.

---

## Step 1 — Extract dataset from replays

```bash
# Test on one replay
python extract_dataset.py --replay_file /path/to/replay.json

# Full extraction (5000 replays)
python extract_dataset.py \
    --replay_dir  /path/to/replays/ \
    --out_path    dataset.npz \
    --neg_ratio   3            # 3 negative samples per positive
    --workers     1
```

**What it produces:**
- Each winner action → `(src, tgt)` **positive edge** (label=1)
- For each positive, samples `neg_ratio` negative edges the winner
  *could have taken* but didn't (label=0)
- Ship fraction = `ships_sent / available_ships` stored for regression

**Expected output from 5000 replays:**
```
Total samples: ~8,000,000
Positives:     ~2,000,000 (25%)
Negatives:     ~6,000,000 (75%)
```

---

## Step 2 — Train

```bash
# Default (recommended — runs on GPU or CPU):
python model.py \
    --dataset    dataset.npz \
    --out_dir    checkpoints/ \
    --epochs     40

# Larger model (GPU with >8 GB VRAM):
python model.py \
    --dataset    dataset.npz \
    --out_dir    checkpoints/ \
    --epochs     40 \
    --d_model    256 \
    --nhead      8  \
    --num_layers 4  \
    --batch_size 64
```

**Key args (EdgeTransformerNet):**

| Arg | Default | Meaning |
|-----|---------|---------|
| `--d_model` | 128 | Transformer hidden size |
| `--nhead` | 4 | Attention heads |
| `--num_layers` | 3 | Transformer layers |
| `--batch_size` | 32 | **Turns** per batch (not edges) |
| `--lr` | 3e-4 | Learning rate |
| `--epochs` | 40 | Training epochs |

Training time (~7M samples, 32 turns/batch):
- CPU: ~20–30 min/epoch → **~16 h total** (not recommended)
- GPU T4 (Kaggle): ~3–5 min/epoch → **~2–3 h total**

**Watch for:**
- F1 should reach 0.55–0.70 on val by epoch 20
- If val_loss diverges from tr_loss after epoch 15 → increase dropout or reduce n_blocks
- Precision > Recall is normal (network is conservative about attacking)

---

## Step 3 — Run the agent

```python
# In your Kaggle submission or local test:
import os
os.environ['ORBIT_MODEL_PATH'] = '/path/to/checkpoints/best_model.pt'

from agent_nn import agent

# The agent signature is identical to main.py:
moves = agent(obs, config)
```

The agent:
1. Loads the model once (cached singleton)
2. Scores all candidate edges with the neural net
3. Uses `world.plan_shot()` for exact physics
4. Falls back to `main.py`'s rule-based agent if model fails to load

---

## Tuning

### If precision is too low (agent attacks randomly):
```python
ACTION_PROB_THRESHOLD = 0.45   # raise threshold in agent_nn.py
TOP_K_EDGES = 3                # commit to fewer edges per turn
```

### If agent is too passive:
```python
ACTION_PROB_THRESHOLD = 0.28
TOP_K_EDGES = 7
```

### If ship counts are wrong:
- Check `y_fraction` distribution in dataset: `np.nanpercentile(y_frac, [10,50,90])`
- Increase `fraction_loss_weight` in training (default 0.5)

### For 4-player games:
The global features encode `n_players` (feature index 14). The model
learns different behavior in 4p vs 2p from the data automatically.
No separate model needed.

---

## What the model learns (that the rule-based agent can't)

| Behavior | Rule-based problem | NN solution |
|---|---|---|
| Coordinated swarm | Scores each source independently | Sees all sources as context in global feats |
| Defensive priority | Fixed `DEFENSE_FRONTIER_SCORE_MULT` | Learns when to defend from game outcomes |
| Ship count | Fixed margin constants | Learns distribution of winning fractions |
| 4-player diplomacy | No "attack the leader" logic beyond a 1.2× mult | Learns leader-targeting from winner patterns |
| Late-game efficiency | LATE_IMMEDIATE_SHIP_VALUE constant | Learns step-dependent behavior from phase feature |

---

## Why 800 Elo was the ceiling

The old system scored each mission with `value / (ships + turns * weight)` 
where `value` was computed by multiplying ~15 constants together. The problem:

1. **Local scoring**: each (src, tgt) pair scored without knowing what
   other pairs were committed that turn. The model sees global state.

2. **Constants tuned against old opponents**: `HOSTILE_TARGET_VALUE_MULT=1.85`
   means "always slightly prefer attacking enemy planets". Top players
   exploit this pattern.

3. **Value function couldn't fix it**: the GBT boosted individual mission
   scores by ±35%, but if the *ranking* of missions was wrong, a 35% boost
   on the wrong winner doesn't help.

The neural net scores edges in the context of the full board state
(global features) and has learned the joint distribution of what
the winner did across all turns — not what a single constant says.
