"""
extract_dataset.py
------------------
Parses a directory of Orbit Wars replay JSON files and produces a labeled
dataset of (edge_features, label, ship_fraction) for training.

Each positive sample = an (src, tgt) edge the winner actually took.
Each negative sample = an (src, tgt) edge the winner COULD have taken
                       but didn't, sampled to keep class balance.

Usage:
    # Full extraction (defaults to records/ → imitation/dataset.npz):
    python imitation/extract_dataset.py

    # Custom batch size (fewer replays in RAM at once):
    python imitation/extract_dataset.py --batch_size 500

    # Parallel workers:
    python imitation/extract_dataset.py --batch_size 500 --workers 4

    # Quick single-file test:
    python imitation/extract_dataset.py --replay_file /path/to/replay.json

Batch mode
----------
Replays are processed in chunks of --batch_size.  Each chunk is saved to
imitation/_chunks/chunk_NNNN.npz immediately, then the batch is freed from RAM.
If the script is interrupted, re-running skips completed chunks automatically.
After all chunks are done they are merged into the final dataset.npz and the
_chunks/ directory is deleted.  dataset_meta.json is updated after every batch.

Output (dataset.npz):
    X            float32  (N, 96)  — feature vectors
    y_action     int8     (N,)     — 1 = winner took this edge, 0 = didn't
    y_fraction   float32  (N,)     — ship fraction sent (NaN for negatives)
    step         int16    (N,)     — game step
    replay       int32    (N,)     — replay index (for grouped CV splits)
    n_players    int8     (N,)     — 2 (1v1) or 4 (4-player FFA)
    winner       int8     (N,)     — player index of the winner (0–3)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# local
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    build_edge_sample, build_all_candidate_edges,
    ship_fraction_label, N_TOTAL_FEATS
)

# ── project-level defaults ────────────────────────────────────────────────────
RECORDS_DIR  = Path(__file__).parent.parent / "records"   # C:\projects\OrbitsWar\records
DATASET_PATH = Path(__file__).parent / "dataset.npz"      # imitation/dataset.npz

# ── target resolution ──────────────────────────────────────────────────────

def _find_target(obs: dict, src_id: int, angle: float) -> Optional[int]:
    """
    Given a launch angle, find which planet it was aimed at using the
    perpendicular-distance test (same as the env's continuous collision).
    Returns planet_id or None.
    """
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

# ── single-replay extractor ────────────────────────────────────────────────

def extract_replay(replay_path: str, neg_ratio: int = 3,
                   rng: np.random.Generator = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Returns arrays (X, y_action, y_fraction, step_arr, n_agents, winner_id)
    for one replay.  Returns empty arrays + (0, -1) on failure.

    n_agents  : 2 for 1v1, 4 for 4-player FFA
    winner_id : player index of the winner (0–3), or -1 on failure
    """
    empty = (np.zeros((0, N_TOTAL_FEATS), dtype=np.float32),
             np.zeros(0, dtype=np.int8),
             np.full(0, np.nan, dtype=np.float32),
             np.zeros(0, dtype=np.int16),
             0,    # n_agents
             -1)   # winner_id

    if rng is None:
        rng = np.random.default_rng()

    try:
        with open(replay_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return empty

    rewards = data.get('rewards', []) or []
    steps   = data.get('steps',   []) or []

    if not rewards or not steps:
        return empty

    # ── robust winner detection (mirrors classifier/train_from_replays.py) ──
    # Guard against None entries — some replays record None for crashed agents
    rewards_num = [r for r in rewards if r is not None]
    if not rewards_num:
        return empty

    max_r = max(rewards_num)
    if rewards_num.count(max_r) > 1 or max_r <= 0:
        return empty          # tie or all-loss — no unambiguous winner

    winner   = rewards.index(max_r)   # index in original list → correct player
    n_agents = len(steps[0])  # 2 = 1v1, 4 = 4-player FFA

    X_rows, y_act_rows, y_frac_rows, step_rows = [], [], [], []

    for step_idx, step in enumerate(steps[1:], 1):  # skip step 0 (empty)
        if winner >= len(step):
            continue
        player_step = step[winner]
        if player_step.get('status') not in ('ACTIVE', 'DONE'):
            continue

        obs     = player_step.get('observation', {})
        actions = player_step.get('action', []) or []
        step_val = obs.get('step', step_idx)

        # ── resolve which (src, tgt) pairs the winner actually took ────────
        taken: dict[Tuple[int,int], float] = {}   # (src,tgt) -> ship_frac
        for a in actions:
            if len(a) < 3:
                continue
            src_id, angle, ships_sent = a[0], a[1], a[2]
            tgt_id = _find_target(obs, src_id, angle)
            if tgt_id is None:
                continue
            frac = ship_fraction_label(obs, src_id, ships_sent)
            if frac is None:
                continue
            # keep the highest-fraction action if same edge appears twice
            key = (src_id, tgt_id)
            if key not in taken or frac > taken[key]:
                taken[key] = frac

        if not taken and not actions:
            # winner chose to do nothing — skip (no positive edges to label)
            continue

        # ── build all candidate edges for this step ────────────────────────
        candidates = build_all_candidate_edges(obs)
        if not candidates:
            continue

        # ── separate positives and negatives ──────────────────────────────
        positives = []
        negatives = []
        for src_id, tgt_id, feats in candidates:
            key = (src_id, tgt_id)
            if key in taken:
                positives.append((feats, taken[key]))
            else:
                negatives.append(feats)

        if not positives:
            continue

        # ── add positives ──────────────────────────────────────────────────
        for feats, frac in positives:
            X_rows.append(feats)
            y_act_rows.append(1)
            y_frac_rows.append(frac)
            step_rows.append(step_val)

        # ── sample negatives (neg_ratio * |positives|) ────────────────────
        n_neg = min(len(negatives), neg_ratio * len(positives))
        if n_neg > 0:
            idxs = rng.choice(len(negatives), size=n_neg, replace=False)
            for i in idxs:
                X_rows.append(negatives[i])
                y_act_rows.append(0)
                y_frac_rows.append(float('nan'))
                step_rows.append(step_val)

    if not X_rows:
        return empty

    X         = np.array(X_rows, dtype=np.float32)
    y_action  = np.array(y_act_rows, dtype=np.int8)
    y_fraction = np.array(y_frac_rows, dtype=np.float32)
    step_arr  = np.array(step_rows, dtype=np.int16)

    return X, y_action, y_fraction, step_arr, n_agents, winner


# ── parallel worker (must be module-level for Windows pickling) ───────────

def _worker(args):
    """Called in a subprocess. Each worker gets its own rng seeded by index."""
    fpath, neg_ratio, file_idx = args
    rng = np.random.default_rng(file_idx * 31337 + 42)
    try:
        result = extract_replay(str(fpath), neg_ratio, rng)
        return file_idx, result, None
    except Exception as e:
        return file_idx, None, str(e)


# ── helpers ───────────────────────────────────────────────────────────────

META_PATH = Path(__file__).parent / "dataset_meta.json"

def _load_meta():
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except Exception:
            pass
    return {}

def _save_meta(meta: dict):
    META_PATH.write_text(json.dumps(meta, indent=2))


def _process_batch(batch_files, neg_ratio, global_offset, workers):
    """Process a list of replay files, return assembled arrays."""
    results = {}
    ok = fail = 0
    n2 = n4 = 0

    if workers <= 1:
        rng = np.random.default_rng(global_offset)
        for local_i, fpath in enumerate(batch_files):
            try:
                result = extract_replay(str(fpath), neg_ratio, rng)
                results[local_i] = result
                if result[0].shape[0] == 0:
                    fail += 1
                else:
                    ok += 1
                    if result[4] == 2: n2 += 1
                    else:              n4 += 1
            except Exception as e:
                fail += 1
    else:
        args_list = [(fpath, neg_ratio, global_offset + i)
                     for i, fpath in enumerate(batch_files)]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_worker, a): a[2] for a in args_list}
            for fut in as_completed(futures):
                file_idx, result, err = fut.result()
                local_i = file_idx - global_offset
                if err or result is None or result[0].shape[0] == 0:
                    fail += 1
                else:
                    results[local_i] = result
                    ok += 1
                    if result[4] == 2: n2 += 1
                    else:              n4 += 1

    # assemble in order
    all_X, all_y_act, all_y_frac = [], [], []
    all_step, all_replay, all_np, all_win = [], [], [], []

    for local_i in sorted(results):
        X_r, y_act, y_frac, step_arr, n_ag, winner_id = results[local_i]
        if X_r.shape[0] == 0:
            continue
        global_i = global_offset + local_i
        n = len(y_act)
        all_X.append(X_r)
        all_y_act.append(y_act)
        all_y_frac.append(y_frac)
        all_step.append(step_arr)
        all_replay.append(np.full(n, global_i,   dtype=np.int32))
        all_np.append(np.full(n, n_ag,           dtype=np.int8))
        all_win.append(np.full(n, winner_id,     dtype=np.int8))

    if not all_X:
        return None, ok, fail, n2, n4

    return {
        'X':          np.concatenate(all_X),
        'y_action':   np.concatenate(all_y_act),
        'y_fraction': np.concatenate(all_y_frac),
        'step':       np.concatenate(all_step),
        'replay':     np.concatenate(all_replay),
        'n_players':  np.concatenate(all_np),
        'winner':     np.concatenate(all_win),
    }, ok, fail, n2, n4


# ── batched extractor ─────────────────────────────────────────────────────

def extract_dataset(replay_dir: str, out_path: str,
                    neg_ratio: int = 3,
                    max_replays: Optional[int] = None,
                    workers: int = 1,
                    batch_size: int = 500):
    """
    Process replays in batches of batch_size.
    Each batch is saved as _chunks/chunk_NNNN.npz immediately → frees RAM.
    Existing chunks are skipped (resumable after crash).
    dataset_meta.json is updated after every batch.
    Final step: merge all chunks → dataset.npz → delete _chunks/.
    """
    replay_dir = Path(replay_dir)
    out_path   = Path(out_path)
    chunk_dir  = out_path.parent / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(replay_dir.glob('*.json'))
    if max_replays:
        files = files[:max_replays]

    n_total  = len(files)
    n_batches = math.ceil(n_total / batch_size)

    print(f"Found {n_total} replay files in {replay_dir}")
    print(f"Batch size  : {batch_size} replays  →  {n_batches} batches")
    print(f"Feature dim : {N_TOTAL_FEATS}")
    print(f"Chunks dir  : {chunk_dir}")
    print()

    meta = _load_meta()
    total_ok = meta.get('total_ok', 0)
    total_fail = meta.get('total_fail', 0)
    total_samples = meta.get('total_samples', 0)
    n2_total = meta.get('n2_games', 0)
    n4_total = meta.get('n4_games', 0)

    chunk_paths = []

    for batch_idx in range(n_batches):
        chunk_path = chunk_dir / f"chunk_{batch_idx:04d}.npz"
        chunk_paths.append(chunk_path)

        if chunk_path.exists():
            # already done — recover stats from meta if available
            prev = meta.get('chunks', {}).get(str(batch_idx), {})
            print(f"  Batch {batch_idx+1}/{n_batches} — chunk exists, skipping  "
                  f"({prev.get('samples', '?')} samples)")
            continue

        batch_start = batch_idx * batch_size
        batch_files = files[batch_start : batch_start + batch_size]

        print(f"  Batch {batch_idx+1}/{n_batches}: replays {batch_start}–"
              f"{batch_start + len(batch_files) - 1} ...", end=' ', flush=True)

        arrays, ok, fail, n2, n4 = _process_batch(
            batch_files, neg_ratio, batch_start, workers)

        if arrays is None:
            print(f"SKIP (no valid samples)  fail={fail}")
            # save empty marker so we don't retry
            np.savez_compressed(chunk_path, X=np.zeros((0, N_TOTAL_FEATS), dtype=np.float32))
        else:
            bad = ~np.isfinite(arrays['X'])
            if bad.any():
                arrays['X'][bad] = 0.0

            np.savez_compressed(chunk_path, **arrays)
            n_samp = arrays['X'].shape[0]
            print(f"ok={ok} fail={fail} 1v1={n2} 4p={n4} samples={n_samp:,}")

            total_ok      += ok
            total_fail    += fail
            total_samples += n_samp
            n2_total      += n2
            n4_total      += n4

            # update meta after every batch
            if 'chunks' not in meta:
                meta['chunks'] = {}
            meta['chunks'][str(batch_idx)] = {'samples': n_samp, 'ok': ok, 'fail': fail}

        meta.update({
            'total_replays':     n_total,
            'batches_done':      batch_idx + 1,
            'total_batches':     n_batches,
            'total_ok':          total_ok,
            'total_fail':        total_fail,
            'total_samples':     total_samples,
            'n2_games':          n2_total,
            'n4_games':          n4_total,
            'feature_dim':       N_TOTAL_FEATS,
            'neg_ratio':         neg_ratio,
            'status':            'in_progress',
        })
        _save_meta(meta)

        # free batch memory immediately
        del arrays

    # ── merge all chunks into final dataset.npz ───────────────────────────
    print(f"\nMerging {len(chunk_paths)} chunks → {out_path} ...")

    all_parts = {k: [] for k in
                 ('X', 'y_action', 'y_fraction', 'step', 'replay', 'n_players', 'winner')}

    for cp in chunk_paths:
        if not cp.exists():
            continue
        data = np.load(cp)
        if data['X'].shape[0] == 0:
            continue
        for k in all_parts:
            all_parts[k].append(data[k])
        del data   # free each chunk after reading

    if not any(all_parts['X']):
        print("ERROR: no valid data in any chunk.")
        return

    final = {k: np.concatenate(v) for k, v in all_parts.items() if v}
    del all_parts

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **final)
    size_mb = out_path.stat().st_size / 1e6

    print(f"Saved → {out_path}  ({size_mb:.1f} MB)")
    print(f"Total samples : {final['X'].shape[0]:,}")
    print(f"  Positives   : {final['y_action'].sum():,}")
    print(f"  Feature dim : {final['X'].shape[1]}")

    # update meta to done
    meta['status'] = 'done'
    meta['output_path'] = str(out_path)
    meta['output_mb']   = round(size_mb, 1)
    _save_meta(meta)

    # clean up chunks
    shutil.rmtree(chunk_dir)
    print(f"Chunks cleaned up.")
    print(f"dataset_meta.json updated.")


# ── quick single-file test ────────────────────────────────────────────────

def _test_single(replay_path: str):
    print(f"Testing: {replay_path}")
    X, y_act, y_frac, step_arr, n_agents, winner_id = \
        extract_replay(replay_path, neg_ratio=3)
    if X.shape[0] == 0:
        print("  No samples extracted — tie / no winner / parse error.")
        return
    print(f"  Game mode : {'1v1' if n_agents == 2 else f'{n_agents}-player FFA'}")
    print(f"  Winner    : player {winner_id}")
    print(f"  Samples   : {X.shape[0]}")
    print(f"  Positives : {y_act.sum()}  ({y_act.mean()*100:.1f}%)")
    print(f"  Feature dim: {X.shape[1]}")
    print(f"  Ship fracs (positives): min={np.nanmin(y_frac):.3f} "
          f"mean={np.nanmean(y_frac):.3f} max={np.nanmax(y_frac):.3f}")
    print(f"  Steps range: {step_arr.min()}–{step_arr.max()}")
    print(f"  Sample X[0]:\n  {X[0]}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract imitation learning dataset from Orbit Wars replays."
    )
    parser.add_argument(
        '--replay_dir',  type=str, default=str(RECORDS_DIR),
        help=f'Directory of replay JSON files (default: {RECORDS_DIR})')
    parser.add_argument(
        '--replay_file', type=str, default=None,
        help='Test on a single replay file instead of a full directory')
    parser.add_argument(
        '--out_path',    type=str, default=str(DATASET_PATH),
        help=f'Output .npz path (default: {DATASET_PATH})')
    parser.add_argument('--neg_ratio',   type=int, default=3)
    parser.add_argument('--max_replays', type=int, default=None)
    parser.add_argument('--workers',     type=int, default=1)
    parser.add_argument('--batch_size',  type=int, default=500,
                        help='Replays per batch (lower = less RAM, default 500)')
    args = parser.parse_args()

    if args.replay_file:
        _test_single(args.replay_file)
    else:
        extract_dataset(args.replay_dir, args.out_path,
                        args.neg_ratio, args.max_replays,
                        args.workers, args.batch_size)
