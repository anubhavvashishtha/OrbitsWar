"""
convert_dataset_f16.py
----------------------
Converts dataset.npz (float32, compressed) to dataset_f16.npz (float16, UNCOMPRESSED).

Why uncompressed?
  model.py uses mmap_mode='r' so X is memory-mapped from disk — no RAM spike.
  mmap only works for UNCOMPRESSED arrays. Compressed npz forces full
  decompression into RAM (15 GB peak) before any mmap can happen → OOM.

Why float16?
  Features are normalized to [-1, 1] or [0, 1] — float16 (3 decimal digits)
  is more than sufficient. Halves disk size: 15 GB → 7.7 GB.

Result:
  dataset_f16.npz  ~7.7 GB on disk (uncompressed float16)
  On Kaggle: np.load('dataset_f16.npz', mmap_mode='r') → zero RAM for X.
  Rows are paged in from disk only as needed per training batch.

Usage:
    python imitation/convert_dataset_f16.py
"""

import numpy as np
from pathlib import Path
import time

SRC = Path(__file__).parent / "dataset.npz"
DST = Path(__file__).parent / "dataset_f16.npz"

print(f"Source : {SRC}  ({SRC.stat().st_size/1e9:.1f} GB compressed)")
print("Loading (this will use ~15 GB RAM — run locally, not on Kaggle) ...")
t0 = time.time()

d = np.load(SRC)

X          = d['X']
y_action   = d['y_action']
y_fraction = d['y_fraction']
replay     = d['replay']
step       = d['step']
n_players  = d['n_players'] if 'n_players' in d else None
winner     = d['winner']    if 'winner'    in d else None

print(f"  Loaded {len(X):,} samples in {time.time()-t0:.0f}s")
print(f"  X shape : {X.shape}  ({X.nbytes/1e9:.1f} GB float32)")
print(f"  X range : min={float(X.min()):.4f}  max={float(X.max()):.4f}")

if float(X.max()) > 60 or float(X.min()) < -60:
    print("  WARNING: large values detected — check feature normalization")

X_f16 = X.astype(np.float16)
print(f"  X_f16   : {X_f16.nbytes/1e9:.1f} GB float16")

save_kwargs = dict(
    X          = X_f16,
    y_action   = y_action,
    y_fraction = y_fraction,
    replay     = replay,
    step       = step,
)
if n_players is not None: save_kwargs['n_players'] = n_players
if winner    is not None: save_kwargs['winner']    = winner

print(f"\nSaving UNCOMPRESSED to {DST} ...")
print("(uncompressed = mmap works on Kaggle — no RAM spike when loading)")
t1 = time.time()

# np.savez (NOT savez_compressed) — required for mmap_mode to work
np.savez(DST, **save_kwargs)

elapsed = time.time() - t1
size_gb = Path(DST).stat().st_size / 1e9
print(f"  Saved in {elapsed:.0f}s  →  {size_gb:.1f} GB")
print()
print("Next steps:")
print("  On Kaggle — zip and upload without downloading:")
print("    import zipfile, time")
print("    with zipfile.ZipFile('/kaggle/working/dataset_f16.zip', 'w',")
print("                         zipfile.ZIP_DEFLATED, compresslevel=1) as zf:")
print("        zf.write('/kaggle/working/dataset_f16.npz', 'dataset_f16.npz')")
print("  Then download dataset_f16.zip and upload to Kaggle as 'orbitswar-dataset'.")
print()
print("Done.")
