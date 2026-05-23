# Continuous PPO on Kaggle: Setup & Execution Guide

Training the Reinforcement Learning agent continuously requires moving your codebase to a Kaggle Notebook so it can run 24/7 on their servers.

Here is exactly how to do it:

## 1. Zip Your Codebase
First, you need to package your entire `OrbitsWar` folder so it can be uploaded to Kaggle as a "Dataset".
Open PowerShell on your computer and run:
```powershell
Compress-Archive -Path C:\projects\OrbitsWar\* -DestinationPath C:\projects\OrbitsWar_Codebase.zip -Force
```

## 2. Create a Kaggle Dataset
1. Go to **Kaggle > Datasets > New Dataset**.
2. Name the dataset something like `orbitswar-codebase`.
3. Upload the `OrbitsWar_Codebase.zip` you just created.
4. Click **Create Dataset**.

## 3. Create the Kaggle Notebook
1. Go to your new Dataset page.
2. Click **New Notebook** (this automatically mounts your codebase into the notebook).
3. In the Kaggle Notebook menu, click **File > Import Notebook**.
4. Upload the `train_ppo_kaggle.ipynb` file I just generated for you.

## 4. Run the Pipeline!
1. Turn on the **GPU** (or keep it on CPU if you prefer, but GPU accelerates PyTorch). To do this, look at the right sidebar under **Accelerator** and select a GPU (e.g., T4 x2).
2. Look at the very first cell. Ensure `INPUT_DIR` matches the path to your dataset. If you named your dataset `orbitswar-codebase`, then `INPUT_DIR = "/kaggle/input/orbitswar-codebase"`.
3. Click **Run All**.

### What to expect:
- **Cell 1:** Will copy your codebase into the `/kaggle/working/OrbitsWar` directory so we can write checkpoint files.
- **Cell 2:** Will run Phase A & Phase B pretraining (using the oversampled replays we discussed previously) to calibrate the Value Head.
- **Cell 3:** Will launch into an infinite Self-Play loop. Every game it plays, it will overwrite `/kaggle/working/OrbitsWar/rl/last.pt`. If it hits a new high score win rate, it saves to `/kaggle/working/OrbitsWar/rl/best.pt`.

## 5. Exporting Weights
After leaving it running for a few hours (or overnight), you'll want to get the weights out.
You have two options:
1. **Download directly from Kaggle:** In the right sidebar, under **Output**, you can navigate into `/kaggle/working/OrbitsWar/rl/` and directly download `last.pt` or `best.pt`.
2. **Export to `.py` right in Kaggle:** Stop the notebook execution manually, add a new Code Cell at the bottom, and run:
   ```python
   !python rl/export_weights.py
   ```
   This will read your `best.pt` file and generate `rl_value_fn.py`, which you can then download and drop right back into your local codebase.
