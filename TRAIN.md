# OrbitsWar — Full Training Guide

End-to-end walkthrough from raw replays to a Kaggle-ready submission agent.

---

## Overview

```
records/*.json
      │
      ▼  extract_dataset.py
imitation/dataset.npz                          ← ~7M edge samples
      │
      ▼  Kaggle: train_imitation_kaggle.ipynb  (~3h, GPU T4)
imitation/checkpoints/best_model.pt            ← EdgeTransformerNet actor
      │
      ▼  Kaggle: train_transformer_ppo.ipynb   (~9h, GPU T4)
rl/transformer_best.pt                         ← PPO fine-tuned actor
      │
      ▼  imitation/agent_nn.py
Kaggle submission
```

---

## Prerequisites

- Python 3.10+ with `.venv` activated
- `pip install numpy torch kaggle-environments`
- All replay JSON files in `records/`
- Kaggle account with GPU quota

---

## Stage 0 — Verify the environment

```powershell
python -c "import numpy, torch; print('numpy', numpy.__version__, '| torch', torch.__version__)"
python -c "from kaggle_environments import make; make('orbit_wars'); print('env OK')"
```

---

## Stage 1 — Extract the imitation dataset

### 1a. Run extraction (uses all 8 physical cores)

```powershell
python imitation/extract_dataset.py --workers 8
```

**Expected output (healthy run):**
```
Found 5109 replay files in records/
  [100/5109] ok=100 fail=0 1v1=0  4p=100  samples=133,860
  ...
  [3600/5109] ok=3600 fail=0 1v1=34  4p=3566  ...   ← 1v1 files start ~here
  ...
Done. Replays: 5106 ok, 3 failed
Total samples: 6,969,820
  Positives: 1,742,455  (25.0%)
  Feature dim: 81
Saved to imitation/dataset.npz  (XXX MB)
```

**Red flags to watch for:**
- `fail` count > 50 → something wrong with replay format
- `Feature dim` ≠ 81 → features.py mismatch, stop and investigate
- Positives % outside 20–30% → neg_ratio wrong

**Time:** ~30–40 min with 8 workers on Ryzen 7 7435HS.

### 1b. Quick sanity check

```powershell
python -c "
import numpy as np
d = np.load('imitation/dataset.npz')
print('Samples   :', len(d['X']))
print('Feature dim:', d['X'].shape[1])
print('Positives  :', d['y_action'].sum(), f'({d[\"y_action\"].mean()*100:.1f}%)')
print('1v1 samples:', (d['n_players']==2).sum())
print('4p  samples:', (d['n_players']==4).sum())
print('NaN in X   :', (~np.isfinite(d['X'])).sum())
"
```

Expected: Feature dim=81, ~25% positives, 0 NaN.

---

## Stage 2 — Build Kaggle upload zips

### 2a. Build source + dataset zips

```powershell
python zip_kaggle_dataset.py --src --dataset
```

Output in `kaggle_upload/`:
```
orbitswar-src.zip        ~X MB   [source code + notebooks]
orbitswar-dataset.zip    ~X MB   [imitation/dataset.npz]
```

**Time:** ~1–2 min (compression).

---

## Stage 3 — Imitation training on Kaggle

### 3a. Upload datasets to Kaggle

Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset** for each:

| Local file | Dataset title | Slug (exact) |
|---|---|---|
| `kaggle_upload/orbitswar-src.zip` | OrbitsWar Source | `orbitswar-src` |
| `kaggle_upload/orbitswar-dataset.zip` | OrbitsWar Dataset | `orbitswar-dataset` |

> Slug must match exactly — the notebook uses these names to find the files.

### 3b. Open `train_imitation_kaggle.ipynb` on Kaggle

1. Upload or open the notebook on Kaggle
2. **Settings → Accelerator → GPU T4 x2**
3. **Add Data** → search `orbitswar-src` → Add
4. **Add Data** → search `orbitswar-dataset` → Add
5. Verify Cell 2 shows both paths as `exists=True`
6. **Run All**

### 3c. What to watch in the logs

Each epoch prints:
```
Epoch  1/40  tr=0.6823  val=0.6791  ent=0.6820  P=0.312 R=0.401 F1=0.351  lr=3.00e-04
```

| Metric | Early (epoch 1–5) | Good (epoch 15–25) | Converged |
|---|---|---|---|
| `val` loss | ~0.65–0.70 | ~0.48–0.55 | <0.45 |
| `ent` entropy | ~0.65–0.69 | ~0.30–0.45 | <0.25 |
| `F1` | ~0.30–0.40 | ~0.55–0.65 | >0.65 |

**Red flags:**
- `ent` stays at 0.693 after epoch 5 → model stuck outputting 0.5 everywhere (random), check pos_weight
- `val` loss goes UP after epoch 10 → overfitting, early stopping will save best checkpoint
- `F1` < 0.40 at epoch 20 → learning rate may be too high

**Time:** ~3–4h on T4 GPU (40 epochs × ~5 min/epoch).

### 3d. Download outputs

From the Kaggle **Output tab**, download:
- `checkpoints/best_model.pt`
- `checkpoints/training_history.json`
- `imitation_training.log`  ← full training log for review

Place `best_model.pt`:
```powershell
# Create dir if needed
mkdir imitation\checkpoints -Force
# Copy downloaded file here:
# imitation/checkpoints/best_model.pt
```

---

## Stage 4 — Build model zips

```powershell
python zip_kaggle_dataset.py --models
```

Output in `kaggle_upload/`:
```
orbitswar-imitation.zip   ~X MB   [best_model.pt from imitation training]
orbitswar-rl.zip          ~X MB   [last.pt — critic warm-start for PPO]
```

> `orbitswar-rl.zip` uses your existing `rl/last.pt` (OrbitsNet PPO weights).
> It warm-starts the ValueNet critic so PPO doesn't start from random.

---

## Stage 5 — PPO fine-tuning on Kaggle

### 5a. Upload model datasets to Kaggle

| Local file | Dataset title | Slug (exact) |
|---|---|---|
| `kaggle_upload/orbitswar-imitation.zip` | OrbitsWar Imitation | `orbitswar-imitation` |
| `kaggle_upload/orbitswar-rl.zip` | OrbitsWar RL | `orbitswar-rl` |

### 5b. Open `train_transformer_ppo.ipynb` on Kaggle

1. Upload or open the notebook
2. **Settings → Accelerator → GPU T4 x2**
3. **Add Data** → attach all 4 datasets:
   - `orbitswar-src`
   - `orbitswar-imitation`
   - `orbitswar-rl`
   *(orbitswar-dataset is not needed for PPO)*
4. **Run All**

### 5c. What to watch in the logs

Each round prints:
```
Round 1  game 8/8 [4p]  W:3 L:4 D:1
  -> WR=37.5%  loss=0.1823  ent=0.9821  alpha=0.0512  [target=1.00]
```

| Metric | Early (round 1–10) | Good (round 50+) | Notes |
|---|---|---|---|
| `WR` | 20–40% | >50% | Win rate vs enemies |
| `ent` | 0.5–1.5 | ~0.9–1.1 | Should hover near target=1.0 |
| `alpha` | ~0.05 | 0.001–0.20 | Auto-adjusts via SAC |
| `loss` | varies | decreasing | PPO clip loss |

**Red flags:**
- `ent` → 0.0 after 20 rounds → entropy collapse, agent plays only one move
- `ent` stays at 1.7+ → policy stuck at random, actor not learning
- `WR` never exceeds 25% over 50 rounds → actor warm-start failed, check best_model.pt loaded correctly
- `alpha` hits ceiling (0.20) and stays there → TARGET_ENTROPY too high for this enemy set

**Time:** ~9h on T4 GPU (200 rounds × ~2.5–3 min/round).

### 5d. Download outputs

From the Kaggle **Output tab**, download:
- `rl/transformer_best.pt`   ← use this for submission
- `rl/transformer_last.pt`   ← last checkpoint (backup)
- `ppo_training.log`         ← full PPO log for review

---

## Stage 6 — Prepare local submission agent

```powershell
# Place the PPO-finetuned model where agent_nn.py looks for it
copy transformer_best.pt imitation\checkpoints\best_model.pt
```

The agent resolves checkpoints in this order:
1. `$ORBIT_MODEL_PATH` env variable
2. `imitation/checkpoints/best_model.pt`   ← this is where to put it
3. `imitation/checkpoints/latest_model.pt`

### Quick local eval

```powershell
python eval_full.py --mode nn --n1v1 1 --n4v4 5
```

Expected: NeuralNet row in the summary table with WR > 50% in 1v1.

---

## Stage 7 — Kaggle submission

1. In your submission notebook, replace the agent import with:
   ```python
   from imitation.agent_nn import agent
   ```
   Or copy `imitation/agent_nn.py` + `imitation/features.py` + `imitation/model.py`
   alongside `main.py` in the submission zip.

2. Set `$ORBIT_MODEL_PATH` or ensure `best_model.pt` is at the correct relative path.

3. Submit.

---

## Quick reference — all commands

```powershell
# Stage 1 — extract dataset
python imitation/extract_dataset.py --workers 8

# Stage 2 — build upload zips
python zip_kaggle_dataset.py --src --dataset       # before imitation training
python zip_kaggle_dataset.py --models              # after imitation training

# Stage 6 — local eval after downloading PPO model
copy transformer_best.pt imitation\checkpoints\best_model.pt
python eval_full.py --mode nn --n1v1 1 --n4v4 5

# Inspect dataset
python imitation/extract_dataset.py --replay_file records/77189904.json  # single file test
```

---

## File map

```
OrbitsWar/
├── records/                        raw replay JSONs (input)
├── imitation/
│   ├── extract_dataset.py          Stage 1: replay → dataset.npz
│   ├── features.py                 81-dim feature builder (shared by all stages)
│   ├── model.py                    EdgeTransformerNet + training loop
│   ├── agent_nn.py                 submission agent (loads best_model.pt)
│   ├── dataset.npz                 extracted dataset (created by Stage 1)
│   └── checkpoints/
│       ├── best_model.pt           best imitation checkpoint (from Kaggle Stage 3)
│       └── latest_model.pt         last epoch checkpoint
├── rl/
│   ├── transformer_trainer.py      PPO fine-tuning loop
│   ├── self_play_trainer.py        older OrbitsNet PPO (provides helpers)
│   ├── last.pt                     OrbitsNet weights → critic warm-start
│   └── transformer_best.pt         PPO fine-tuned model (from Kaggle Stage 5)
├── classifier/
│   └── train_from_replays.py       extract_features() used by PPO critic
├── train_imitation_kaggle.ipynb    Kaggle: Stage 3
├── train_transformer_ppo.ipynb     Kaggle: Stage 5
├── zip_kaggle_dataset.py           builds kaggle_upload/*.zip
├── eval_full.py                    local evaluation vs all enemies
└── kaggle_upload/                  zips ready to upload (created by Stage 2/4)
    ├── orbitswar-src.zip
    ├── orbitswar-dataset.zip
    ├── orbitswar-imitation.zip
    └── orbitswar-rl.zip
```
