# Training Your RL Agent on Kaggle GPU (Free)

Kaggle gives you **30 hours/week of free GPU** (T4 × 2). Training 1M steps
takes about **3–4 hours** on a T4. Follow this guide exactly.

---

## Step 1 — Upload Your Code as a Kaggle Dataset

You need to get your local project files onto Kaggle. The easiest way is to
zip the relevant files and upload them as a **Dataset**.

### Files to include in the upload zip

```
orbit-wars-rl/
├── rl/
│   ├── __init__.py
│   ├── features.py
│   ├── model.py
│   ├── env_wrapper.py
│   ├── ppo.py
│   └── train_rl.py
├── main.py                   ← your new RL submission agent
├── fallback.py               ← your current heuristic agent (used as fallback)
└── enemy agents/
    ├── main_1.py
    ├── main_2.py
    └── main_3.py
```

### How to zip and upload

**Option A — Kaggle CLI (fastest)**

```bash
# From your project root (c:\projects\OrbitsWar)
# Create the zip
Compress-Archive -Path rl, main.py, fallback.py, "enemy agents" -DestinationPath orbit-wars-rl.zip -Force

# Upload as a dataset (run once)
kaggle datasets create -p . --dir-mode zip -t "orbit-wars-rl"
```

**Option B — Web UI**

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"**
3. Upload `orbit-wars-rl.zip`
4. Name it **orbit-wars-rl** and make it Private
5. Click **"Create"**

---

## Step 2 — Create a Kaggle Notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **"New Notebook"**
2. In the notebook settings (right panel):
   - **Accelerator**: GPU T4 x2
   - **Language**: Python
   - **Internet**: ON (needed to install packages)
3. Click **"Add data"** → search for your **orbit-wars-rl** dataset → add it

Your dataset will be mounted at `/kaggle/input/orbit-wars-rl/`.

---

## Step 3 — Paste This Training Script Into the Notebook

Copy each cell below into your Kaggle notebook and run them in order.

### Cell 1 — Install dependencies

```python
!pip install -q kaggle-environments torch
```

### Cell 2 — Copy your code to the working directory

```python
import shutil, os

# Copy uploaded dataset files to the working directory so imports work
src = "/kaggle/input/orbit-wars-rl"
dst = "/kaggle/working"

for item in os.listdir(src):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

os.makedirs("/kaggle/working/rl/checkpoints", exist_ok=True)
print("Files ready:", os.listdir("/kaggle/working"))
```

### Cell 3 — Run Phase 1 training (vs Random — 200K steps, ~45 min)

```python
import subprocess, sys

result = subprocess.run([
    sys.executable, "-m", "rl.train_rl",
    "--steps", "200000",
    "--n_steps", "1024",
], cwd="/kaggle/working", capture_output=False)
```

### Cell 4 — Run Phase 2 + 3 training (vs Enemies + Self-Play — 800K steps, ~3 hrs)

```python
# Find the latest checkpoint to resume from
import glob, os

checkpoints = sorted(glob.glob("/kaggle/working/rl/checkpoints/policy_step_*.pt"))
latest = checkpoints[-1] if checkpoints else None
print("Resuming from:", latest)

result = subprocess.run([
    sys.executable, "-m", "rl.train_rl",
    "--steps", "800000",
    "--resume", latest or "",
], cwd="/kaggle/working", capture_output=False)
```

> **Tip:** Kaggle notebooks timeout after 9 hours. If training takes longer,
> split into multiple runs using `--steps` and `--resume`.

### Cell 5 — Download the trained checkpoint

```python
# The best checkpoint is saved here:
print("Best policy:", "/kaggle/working/rl/checkpoints/best_policy.pt")

# All checkpoints:
for f in sorted(glob.glob("/kaggle/working/rl/checkpoints/*.pt")):
    size = os.path.getsize(f) / 1024
    print(f"  {os.path.basename(f)} ({size:.0f} KB)")
```

After the run, go to the **Output** tab of your notebook and download
`rl/checkpoints/best_policy.pt` (it's about 2–5 MB).

---

## Step 4 — Use the Trained Model Locally

1. Copy `best_policy.pt` you downloaded into `c:\projects\OrbitsWar\rl\checkpoints\`
2. Test it locally:

```bash
# Quick smoke test — 1 game vs random
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
final = env.steps[-1]
print('RL agent reward:', final[0].reward)
print('Random reward:  ', final[1].reward)
"
```

3. Run it against your enemies:

```bash
# Edit run.py or just test directly:
python -c "
from kaggle_environments import make
from main import agent as rl_agent
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location('m1', r'enemy agents/main_1.py')
m = module_from_spec(spec); spec.loader.exec_module(m)

env = make('orbit_wars', debug=True)
env.run([rl_agent, m.agent])
print(env.steps[-1][0].reward, 'vs', env.steps[-1][1].reward)
"
```

---

## Step 5 — Submit to Kaggle Competition

Bundle `main.py`, `fallback.py`, the `rl/` folder, and `best_policy.pt` together:

```bash
# From c:\projects\OrbitsWar
mkdir submission_rl
copy main.py submission_rl\main.py
copy fallback.py submission_rl\fallback.py
xcopy /E /I rl submission_rl\rl

# Compress
Compress-Archive -Path submission_rl\* -DestinationPath submission_rl.zip -Force

# Submit
kaggle competitions submit orbit-wars -f submission_rl.zip -m "PPO RL agent v1"
```

---

## Understanding the Training Log

After training, `rl/training_log.json` contains per-update stats:

```json
{
  "step": 50000,
  "phase": "Phase 1: vs Random",
  "win_rate": 0.65,
  "mean_reward": 0.12,
  "pg_loss": 0.034,
  "v_loss": 0.21,
  "ent_loss": -2.1
}
```

| Field | What to look for |
|---|---|
| `win_rate` | Should rise above 0.5 by 200K steps vs random |
| `pg_loss` | Should stay near 0 (big spikes = learning rate too high) |
| `v_loss` | Should decrease steadily |
| `ent_loss` | Should stay negative (positive = collapsed policy) |

---

## Recommended Training Schedule

The training loop automatically flips a coin every match to decide if it will run a 1v1 game or a 1v1v1v1 game. 
For 4-player games, it randomly samples from the pool of available opponents independently for each slot (meaning it can fight `main_1.py`, `main_3.py`, and `random` simultaneously).

| Run | Steps | Time (T4) | Opponent Pool |
|---|---|---|---|
| Run 1 | 200K | ~45 min | Random |
| Run 2 | 400K | ~1.5 hrs | Enemies + Random |
| Run 3 | 400K | ~1.5 hrs | Enemies + Random + Self-play (Past Checkpoints) |
| **Total** | **1M** | **~4 hrs** | |

For a stronger agent, repeat Run 3 several times (resume from latest checkpoint).

---

## Frequently Asked Questions

**Q: Can I run training locally without a GPU?**
Yes — it just takes ~4–6× longer. The code auto-detects CPU vs GPU.
```bash
python -m rl.train_rl --steps 50000   # quick local test
```

**Q: Does the RL agent use the heuristic rules from fallback.py?**
No. It learns entirely from experience. The neural network figures out its
own strategy by playing thousands of games. It may discover strategies you
never programmed.

**Q: How do I continue improving the agent?**
Keep running `--resume` from the latest checkpoint. The self-play phase
(after 600K steps) is where the biggest improvements happen — the agent
starts playing against past versions of itself.

**Q: What if win rate doesn't improve?**
Try lowering the learning rate: `--lr 1e-4`, or check that reward shaping
is on (default). You can also increase `--n_steps 2048` for more stable updates.
