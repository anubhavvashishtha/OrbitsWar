"""
zip_kaggle_dataset.py
======================
Creates the dataset zip to upload to Kaggle for training.

    python zip_kaggle_dataset.py

Output: OrbitsWar_TrainingDataset.zip

What it includes
----------------
  records/          — Kaggle replay JSON files (training data)
  enemy agents/     — Enemy bots for vs-bots / RL self-play
  classifier/       — 1v1 GBC training scripts
  heuristic_4v4/    — 4v4 GBC training scripts + agent
  regressor/        — 1v1 GBR training scripts
  regressor_4v4/    — 4v4 GBR training scripts
  rl/               — PPO actor-critic training scripts (net, trainer, exporter)
  main.py           — Required by vs-bots scripts and RL self-play

What it excludes
----------------
  .pkl replay caches — Rebuilt automatically on first run (model.pkl IS included)
  .pyc / __pycache__
  submissions/
  .venv/
  enemy agents/*/weights/ — enemy checkpoint files (not needed for training)

Upload instructions
-------------------
1. Run this script to create OrbitsWar_TrainingDataset.zip
2. Go to kaggle.com/datasets → New Dataset
3. Upload the zip
4. Note the dataset slug (e.g. username/orbitswar-training)
5. In train_models_kaggle.ipynb, update DATASET_SLUG to match
"""
import os
import zipfile
from pathlib import Path

OUTPUT_ZIP = "OrbitsWar_TrainingDataset.zip"

# Folders/files to include
INCLUDE = [
    "records",
    "enemy agents",
    "classifier",
    "heuristic_4v4",
    "regressor",
    "regressor_4v4",
    "rl",
    "main.py",
]

# Extensions to always skip
SKIP_EXTS = {".pyc", ".zip"}

# Directory names to skip entirely
SKIP_DIRS = {"__pycache__", ".venv", "submissions", ".git", "weights"}

# Filenames to skip — replay caches (large, rebuilt on first run); model.pkl IS included
SKIP_FILES = {
    "replay_cache_v2.pkl",
    "replay_cache_4v4.pkl",
    "replay_td_cache.pkl",
    "replay_td_cache_4v4.pkl",
}


def should_skip(path: str) -> bool:
    p = Path(path)
    if p.suffix in SKIP_EXTS:
        return True
    if p.name in SKIP_FILES:
        return True
    for part in p.parts:
        if part in SKIP_DIRS:
            return True
    return False


def main():
    print(f"Creating {OUTPUT_ZIP} ...")
    print("=" * 50)

    total_files = 0
    total_bytes = 0

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for inc in INCLUDE:
            if not os.path.exists(inc):
                print(f"  [SKIP] Not found: {inc}")
                continue

            if os.path.isfile(inc):
                if should_skip(inc):
                    continue
                zf.write(inc)
                sz = os.path.getsize(inc)
                total_bytes += sz
                total_files += 1
                print(f"  + {inc}  ({sz/1024:.1f} KB)")

            else:
                n_dir = 0
                for root, dirs, files in os.walk(inc):
                    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
                    for fname in sorted(files):
                        fpath = os.path.join(root, fname)
                        if should_skip(fpath):
                            continue
                        zf.write(fpath)
                        sz = os.path.getsize(fpath)
                        total_bytes += sz
                        total_files += 1
                        n_dir += 1
                        if not fname.endswith(".json"):
                            print(f"  + {fpath}  ({sz/1024:.1f} KB)")
                        else:
                            pass  # suppress per-replay noise

                json_count = sum(
                    1 for _, _, fs in os.walk(inc)
                    for f in fs if f.endswith(".json")
                )
                if json_count:
                    print(f"  + {inc}/*.json  ({json_count} replay files)")

    zip_mb = os.path.getsize(OUTPUT_ZIP) / 1024 / 1024
    print("=" * 50)
    print(f"Done: {total_files} files  →  {OUTPUT_ZIP}  ({zip_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("  1. Upload OrbitsWar_TrainingDataset.zip to kaggle.com/datasets")
    print("  2. Update DATASET_SLUG in train_models_kaggle.ipynb")
    print("  3. Attach the dataset to your notebook and run all cells")


if __name__ == "__main__":
    main()
