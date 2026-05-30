"""
zip_kaggle_dataset.py
======================
Creates the Kaggle dataset zips needed to run train_transformer_ppo.ipynb.

Usage
-----
    python zip_kaggle_dataset.py            # all three zips
    python zip_kaggle_dataset.py --src      # source code only
    python zip_kaggle_dataset.py --models   # checkpoints only

Outputs  (written to kaggle_upload/)
-------------------------------------
  orbitswar-src.zip        upload as Kaggle dataset  "orbitswar-src"
  orbitswar-imitation.zip  upload as Kaggle dataset  "orbitswar-imitation"
  orbitswar-rl.zip         upload as Kaggle dataset  "orbitswar-rl"  (optional)

Upload order
------------
  1. orbitswar-src.zip        (required)
  2. orbitswar-imitation.zip  (required — train imitation model first)
  3. orbitswar-rl.zip         (optional — warm-starts the PPO critic)

Then attach all datasets to train_transformer_ppo.ipynb and run all cells.
"""

import argparse
import os
import zipfile
from pathlib import Path

ROOT       = Path(__file__).parent
OUT_DIR    = ROOT / "kaggle_upload"

# ── Skip rules (shared across all zips) ──────────────────────────────────────
SKIP_EXTS  = {".pyc", ".zip", ".pkl", ".npz"}
SKIP_DIRS  = {"__pycache__", ".venv", "submissions", ".git", "weights",
              "kaggle_upload", ".mypy_cache"}
SKIP_FILES = {
    "replay_cache_v2.pkl", "replay_cache_4v4.pkl",
    "replay_td_cache.pkl", "replay_td_cache_4v4.pkl",
    # trained GBC/GBR model serialised as Python — huge and not needed for PPO
    "value_fn_1v1.py", "value_fn_4v4.py",
}

# Files larger than this (MB) are skipped in the source zip
# (avoids accidentally including serialised models / embedded weights)
MAX_SRC_FILE_MB = 5.0


def _should_skip(path: Path, max_mb: float = None) -> bool:
    if path.suffix in SKIP_EXTS:
        return True
    if path.name in SKIP_FILES:
        return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    if max_mb is not None and path.exists():
        if path.stat().st_size > max_mb * 1e6:
            print(f"    [SKIP large] {path.name}  ({path.stat().st_size/1e6:.1f} MB > {max_mb} MB limit)")
            return True
    return False


def _add_file(zf: zipfile.ZipFile, src: Path, arcname: str,
              stats: dict, quiet: bool = False):
    """Add one file to zip and update stats."""
    zf.write(src, arcname=arcname)
    sz = src.stat().st_size
    stats["files"] += 1
    stats["bytes"] += sz
    if not quiet:
        print(f"    + {arcname}  ({sz/1024:.1f} KB)")


def _add_dir(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str,
             stats: dict, quiet_ext: set = None, max_mb: float = None,
             skip_exts: set = None):
    """Recursively add a directory to zip under arc_prefix."""
    quiet_ext  = quiet_ext  or set()
    skip_exts  = skip_exts  or set()
    n = 0
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fname in sorted(files):
            fpath = Path(root) / fname
            if fpath.suffix in skip_exts:
                continue
            if _should_skip(fpath, max_mb=max_mb):
                continue
            rel   = fpath.relative_to(src_dir)
            arc   = f"{arc_prefix}/{rel}".replace("\\", "/")
            quiet = fpath.suffix in quiet_ext
            _add_file(zf, fpath, arc, stats, quiet=quiet)
            n += 1
    return n


# ══════════════════════════════════════════════════════════════════════════════
# ZIP 1 — orbitswar-src   (source code + notebook)
# ══════════════════════════════════════════════════════════════════════════════

def build_src_zip(out_path: Path):
    """
    Source code + both Kaggle notebooks.

    Layout inside zip (mirrors project root so /kaggle/working/ copy works):
      main.py
      train_imitation_kaggle.ipynb
      train_transformer_ppo.ipynb
      imitation/
      rl/
      classifier/
      enemy agents/
    """
    stats = {"files": 0, "bytes": 0}
    print(f"\n{'='*60}")
    print(f"  Building {out_path.name}")
    print(f"{'='*60}")

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── main.py ───────────────────────────────────────────────────────────
        mp = ROOT / "main.py"
        if mp.exists():
            _add_file(zf, mp, "main.py", stats)
        else:
            print("  [WARN] main.py not found!")

        # ── Notebooks ─────────────────────────────────────────────────────────
        for nb_name in ["train_imitation_kaggle.ipynb",
                        "train_transformer_ppo.ipynb"]:
            nb = ROOT / nb_name
            if nb.exists():
                _add_file(zf, nb, nb_name, stats)
            else:
                print(f"  [WARN] {nb_name} not found!")

        # ── imitation/ ────────────────────────────────────────────────────────
        imit = ROOT / "imitation"
        if imit.exists():
            print(f"\n  imitation/")
            # include code files but NOT the dataset.npz (too large)
            for fname in sorted(imit.iterdir()):
                if fname.is_file() and not _should_skip(fname):
                    _add_file(zf, fname, f"imitation/{fname.name}", stats)
            # include sub-dirs (e.g. checkpoints dir exists but might be empty)
            for sub in sorted(imit.iterdir()):
                if sub.is_dir() and sub.name not in SKIP_DIRS:
                    n = _add_dir(zf, sub, f"imitation/{sub.name}", stats)
                    if n == 0:
                        print(f"    (imitation/{sub.name}/ — empty, skipped)")
        else:
            print("  [WARN] imitation/ not found!")

        # ── rl/ ───────────────────────────────────────────────────────────────
        # Exclude .pt checkpoints — they go in orbitswar-rl.zip separately
        rl = ROOT / "rl"
        if rl.exists():
            print(f"\n  rl/  (code only, .pt checkpoints -> orbitswar-rl.zip)")
            _add_dir(zf, rl, "rl", stats, skip_exts={".pt"})
        else:
            print("  [WARN] rl/ not found!")

        # ── classifier/ ───────────────────────────────────────────────────────
        # Only need train_from_replays.py (extract_features used by PPO critic)
        # Skip value_fn_1v1.py (9 MB serialised GBC model, not needed for PPO)
        clf = ROOT / "classifier"
        if clf.exists():
            print(f"\n  classifier/  (skipping value_fn*.py)")
            _add_dir(zf, clf, "classifier", stats, max_mb=MAX_SRC_FILE_MB)
        else:
            print("  [WARN] classifier/ not found!")

        # ── enemy agents/ ─────────────────────────────────────────────────────
        # Skip files > MAX_SRC_FILE_MB (embedded-model submission bots are fine
        # to run as enemies on Kaggle but are too large for the src zip)
        ea = ROOT / "enemy agents"
        if ea.exists():
            print(f"\n  enemy agents/  (files > {MAX_SRC_FILE_MB} MB skipped)")
            _add_dir(zf, ea, "enemy agents", stats, max_mb=MAX_SRC_FILE_MB)
        else:
            print("  [WARN] 'enemy agents/' not found -- notebook will use main.py as self-play bot.")

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Done: {stats['files']} files  ->  {mb:.1f} MB")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# ZIP 2 — orbitswar-imitation   (trained EdgeTransformerNet checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

def build_imitation_zip(out_path: Path):
    """
    Contains best_model.pt from imitation/checkpoints/.
    Run AFTER imitation training completes.
    """
    stats = {"files": 0, "bytes": 0}
    print(f"\n{'='*60}")
    print(f"  Building {out_path.name}")
    print(f"{'='*60}")

    ckpt = ROOT / "imitation" / "checkpoints" / "best_model.pt"
    latest = ROOT / "imitation" / "checkpoints" / "latest_model.pt"

    if not ckpt.exists():
        print(f"  [SKIP] {ckpt} not found.")
        print("  Train the imitation model first:")
        print("    python imitation/model.py --dataset imitation/dataset.npz --out_dir imitation/checkpoints")
        print("  Then re-run this script.")
        return None

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _add_file(zf, ckpt, "best_model.pt", stats)
        if latest.exists():
            _add_file(zf, latest, "latest_model.pt", stats)
        # Include training history if present
        hist = ROOT / "imitation" / "checkpoints" / "training_history.json"
        if hist.exists():
            _add_file(zf, hist, "training_history.json", stats)

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Done: {stats['files']} files  ->  {mb:.1f} MB")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# ZIP 3 — orbitswar-rl   (OrbitsNet critic warm-start, optional)
# ══════════════════════════════════════════════════════════════════════════════

def build_rl_zip(out_path: Path):
    """
    Contains EdgeTransformerNet PPO checkpoints for resuming training.

      transformer_last.pt  — latest checkpoint from previous session
      transformer_best.pt  — best win-rate checkpoint

    Upload this zip as 'orbitswar-rl' and attach it to train_transformer_ppo.ipynb
    to resume from a previous Kaggle session instead of starting fresh.
    """
    stats = {"files": 0, "bytes": 0}
    print(f"\n{'='*60}")
    print(f"  Building {out_path.name}")
    print(f"{'='*60}")

    candidates = [
        (ROOT / "rl" / "transformer_last.pt", "transformer_last.pt"),
        (ROOT / "rl" / "transformer_best.pt", "transformer_best.pt"),
    ]
    found = [(src, arc) for src, arc in candidates if src.exists()]

    if not found:
        print("  [SKIP] No transformer checkpoints found in rl/.")
        print("  Download transformer_last.pt from the Kaggle Output tab first,")
        print("  place it in rl/, then re-run:  python zip_kaggle_dataset.py --models")
        return None

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in found:
            _add_file(zf, src, arc, stats)

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Done: {stats['files']} files  ->  {mb:.1f} MB")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# ZIP 4 — orbitswar-dataset   (dataset.npz for imitation training)
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_zip(out_path: Path):
    """
    Zips imitation/dataset.npz.
    Run after extract_dataset.py finishes.
    """
    stats = {"files": 0, "bytes": 0}
    print(f"\n{'='*60}")
    print(f"  Building {out_path.name}")
    print(f"{'='*60}")

    npz = ROOT / "imitation" / "dataset.npz"
    if not npz.exists():
        print(f"  [SKIP] imitation/dataset.npz not found.")
        print("  Wait for extract_dataset.py to finish, then re-run.")
        return None

    sz_mb = npz.stat().st_size / 1e6
    print(f"  Zipping dataset.npz  ({sz_mb:.1f} MB) ...")
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        _add_file(zf, npz, "dataset.npz", stats, quiet=True)

    mb = out_path.stat().st_size / 1e6
    ratio = sz_mb / mb if mb > 0 else 0
    print(f"  Done: {sz_mb:.1f} MB -> {mb:.1f} MB compressed  (ratio {ratio:.1f}x)")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# ZIP 5 — submission_imitation   (Kaggle competition submission)
# ══════════════════════════════════════════════════════════════════════════════

def build_submission_zip(out_path: Path):
    """
    Creates a submission zip ready to upload to the orbit_wars competition.

    Flat structure (all files at root so imports work inside Kaggle sandbox):
      agent_nn.py          ← entry point — defines agent(obs, config)
      features.py          ← edge feature builder
      model.py             ← EdgeTransformerNet + TransformerCritic
      main.py              ← physics layer (build_world, plan_shot)
      checkpoints/
        best_model.pt      ← imitation model weights

    The agent loads the PPO checkpoint first (rl/transformer_best.pt) if it
    exists, then falls back to checkpoints/best_model.pt automatically.
    For an imitation-only submission, only the imitation checkpoint is included.
    """
    stats = {"files": 0, "bytes": 0}
    print(f"\n{'='*60}")
    print(f"  Building {out_path.name}")
    print(f"{'='*60}")

    # Required files
    files = [
        (ROOT / "imitation" / "agent_nn.py",  "agent_nn.py"),
        (ROOT / "imitation" / "features.py",  "features.py"),
        (ROOT / "imitation" / "model.py",     "model.py"),
        (ROOT / "main.py",                    "main.py"),
    ]
    ckpt = ROOT / "imitation" / "checkpoints" / "best_model.pt"

    # Validate everything exists before writing
    missing = [src for src, _ in files if not src.exists()]
    if missing:
        for m in missing:
            print(f"  [SKIP] Missing: {m}")
        return None
    if not ckpt.exists():
        print(f"  [SKIP] Model not found: {ckpt}")
        print("  Train imitation model first:")
        print("    python imitation/model.py --dataset imitation/dataset.npz")
        return None

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            _add_file(zf, src, arc, stats)
        _add_file(zf, ckpt, "checkpoints/best_model.pt", stats)

    mb = out_path.stat().st_size / 1e6
    ckpt_mb = ckpt.stat().st_size / 1e6
    print(f"\n  Done: {stats['files']} files  ->  {mb:.1f} MB")
    print(f"  Model weights: {ckpt_mb:.1f} MB")
    print(f"\n  Submit at: kaggle.com/competitions/orbit-wars -> Submit Predictions")
    print(f"  Upload   : {out_path.name}")
    return stats




def build_submission_ppo_zip(out_path: Path):
    """
    Competition submission with PPO-finetuned EdgeTransformerNet.

    Same flat layout as submission_imitation, plus:
      checkpoints/transformer_best.pt  (from rl/ after PPO training)
    agent_nn.py loads PPO checkpoint first, then imitation fallback.
    """
    stats = {"files": 0, "bytes": 0}
    print(f"\n{'='*60}")
    print(f"  Building {out_path.name}")
    print(f"{'='*60}")

    files = [
        (ROOT / "imitation" / "agent_nn.py",  "agent_nn.py"),
        (ROOT / "imitation" / "features.py",  "features.py"),
        (ROOT / "imitation" / "model.py",     "model.py"),
        (ROOT / "main.py",                    "main.py"),
    ]
    ppo_candidates = [
        (ROOT / "rl" / "transformer_best.pt", "checkpoints/transformer_best.pt"),
    ]
    ppo_found = [(src, arc) for src, arc in ppo_candidates if src.exists()]

    missing = [src for src, _ in files if not src.exists()]
    if missing:
        for m in missing:
            print(f"  [SKIP] Missing: {m}")
        return None
    if not ppo_found:
        print("  [SKIP] No PPO checkpoint in rl/.")
        print("  Download transformer_best.pt from Kaggle Output -> rl/")
        return None

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            _add_file(zf, src, arc, stats)
        for src, arc in ppo_found:
            _add_file(zf, src, arc, stats)
        imit = ROOT / "imitation" / "checkpoints" / "best_model.pt"
        if imit.exists():
            _add_file(zf, imit, "checkpoints/best_model.pt", stats)

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Done: {stats['files']} files  ->  {mb:.1f} MB")
    print("  Submit at: kaggle.com/competitions/orbit-wars")
    return stats

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Build Kaggle dataset zips.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow
--------
  Step 1 (now):      python zip_kaggle_dataset.py --src
                     Upload orbitswar-src.zip as dataset 'orbitswar-src'

  Step 2 (after extraction finishes):
                     python zip_kaggle_dataset.py --dataset
                     Upload orbitswar-dataset.zip as dataset 'orbitswar-dataset'
                     Run train_imitation_kaggle.ipynb on Kaggle GPU (~3h)

  Step 3 (after imitation training):
                     Download best_model.pt -> imitation/checkpoints/
                     python zip_kaggle_dataset.py --models
                     Upload orbitswar-imitation.zip + orbitswar-rl.zip
                     Run train_transformer_ppo.ipynb on Kaggle GPU
        """)
    parser.add_argument("--src",        action="store_true", help="Source code + notebooks")
    parser.add_argument("--dataset",    action="store_true", help="dataset.npz (after extraction)")
    parser.add_argument("--models",     action="store_true", help="Model checkpoints (after training)")
    parser.add_argument("--submission", action="store_true", help="Imitation-only competition zip")
    parser.add_argument("--submission-ppo", action="store_true", help="PPO competition zip")
    parser.add_argument("--all",        action="store_true", help="Build everything available")
    args = parser.parse_args()

    # default: build everything if no flag given
    build_all    = args.all or not (args.src or args.dataset or args.models or args.submission or args.submission_ppo)
    do_src        = args.src        or build_all
    do_dataset    = args.dataset    or build_all
    do_models     = args.models     or build_all
    do_submission = args.submission or build_all
    do_submission_ppo = args.submission_ppo or build_all

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    built = []

    if do_src:
        p = OUT_DIR / "orbitswar-src.zip"
        r = build_src_zip(p)
        if r:
            built.append(("orbitswar-src",     p, "required - upload first"))

    if do_dataset:
        p = OUT_DIR / "orbitswar-dataset.zip"
        r = build_dataset_zip(p)
        if r:
            built.append(("orbitswar-dataset", p, "required for imitation training"))

    if do_models:
        p = OUT_DIR / "orbitswar-imitation.zip"
        r = build_imitation_zip(p)
        if r:
            built.append(("orbitswar-imitation", p, "required for PPO training"))

        p = OUT_DIR / "orbitswar-rl.zip"
        r = build_rl_zip(p)
        if r:
            built.append(("orbitswar-rl",         p, "optional — critic warm-start"))

    if do_submission:
        p = OUT_DIR / "submission_imitation.zip"
        r = build_submission_zip(p)
        if r:
            built.append(("submission_imitation", p, "upload to competition"))

    if do_submission_ppo:
        p = OUT_DIR / "submission_ppo.zip"
        r = build_submission_ppo_zip(p)
        if r:
            built.append(("submission_ppo", p, "upload to competition (PPO)"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    if not built:
        print("  Nothing was built. Check warnings above.")
        return

    for slug, path, note in built:
        mb = path.stat().st_size / 1e6
        print(f"  {path.name:<35} {mb:>7.1f} MB   [{note}]")

    print(f"\n  All files saved to: {OUT_DIR}/")
    print()
    print("  Upload steps:")
    print("  " + "-"*57)
    for i, (slug, path, note) in enumerate(built, 1):
        print(f"  {i}. kaggle.com/datasets -> New Dataset")
        print(f"     Upload : {path.name}")
        print(f"     Title  : {slug}")
        print(f"     Slug   : {slug}   ({note})")
        print()
    print("  Then in train_transformer_ppo.ipynb:")
    print("    Data -> Add Data -> search each slug and attach")
    print("    Verify dataset names match CELL 3 of the notebook:")
    print("      SRC_DATASET       = 'orbitswar-src'")
    print("      IMITATION_DATASET = 'orbitswar-imitation'")
    print("      RL_DATASET        = 'orbitswar-rl'")


if __name__ == "__main__":
    main()
