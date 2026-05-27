"""
agent_nn.py
-----------
Drop-in agent using EdgeTransformerNet for joint edge scoring.

All candidate edges for a turn are scored TOGETHER through a Transformer
encoder — edge (A→X) attends to (B→Y), (C→X), etc. before producing its
score.  Only physics (angles, orbits) comes from main.py.

Model path resolution (first found wins):
  1. $ORBIT_MODEL_PATH environment variable
  2. checkpoints/best_model.pt
  3. checkpoints/latest_model.pt

Usage (Kaggle submission):
    from agent_nn import agent
"""

from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# ── physics layer (main.py) ───────────────────────────────────────────────
try:
    from main import build_world, WorldModel, plan_shot
    _MAIN_AVAILABLE = True
except ImportError:
    _MAIN_AVAILABLE = False

# ── local imports ─────────────────────────────────────────────────────────
from features import build_all_candidate_edges
from model import EdgeTransformerNet, load_model

# ── configuration ──────────────────────────────────────────────────────────
_ENV_PATH       = os.environ.get('ORBIT_MODEL_PATH', '')
_CHECKPOINT_DIR = Path(__file__).parent / 'checkpoints'

_CANDIDATE_CHECKPOINTS = [p for p in [
    Path(_ENV_PATH) if _ENV_PATH else None,
    _CHECKPOINT_DIR / 'best_model.pt',
    _CHECKPOINT_DIR / 'latest_model.pt',
] if p is not None]

TOP_K_EDGES          = 5      # max edges committed per turn
ACTION_PROB_THRESHOLD = 0.35  # min action probability
MIN_FRACTION          = 0.08  # min ship fraction (below → skip)
MAX_FRACTION_PER_PLANET = 0.90

# ── model singleton ────────────────────────────────────────────────────────
_model: Optional[EdgeTransformerNet] = None
_device = 'cpu'


def _get_model() -> Optional[EdgeTransformerNet]:
    global _model
    if _model is not None:
        return _model
    for ckpt in _CANDIDATE_CHECKPOINTS:
        if not ckpt.exists():
            continue
        try:
            _model = load_model(str(ckpt), device=_device)
            _model.eval()
            return _model
        except Exception as e:
            print(f"[agent_nn] Could not load {ckpt.name}: {e}")
    print("[agent_nn] No checkpoint found — falling back to rule-based agent.")
    return None


# ── edge scoring ───────────────────────────────────────────────────────────

def score_all_edges(
        obs: dict,
        model: EdgeTransformerNet,
) -> List[Tuple[float, float, int, int]]:
    """
    Score all candidate edges JOINTLY.

    All (src, tgt) pairs for this turn are packed into a (1, E, 81)
    tensor and processed through the Transformer in a single forward pass.
    Every edge attends to every other before producing its score.

    Returns list of (action_prob, fraction, src_id, tgt_id) sorted by
    action_prob descending.
    """
    candidates = build_all_candidate_edges(obs)
    if not candidates:
        return []

    edge_list   = [(src_id, tgt_id) for src_id, tgt_id, _ in candidates]
    feat_matrix = np.array([f for _, _, f in candidates], dtype=np.float32)

    # (1, E, 81) — single turn, no padding needed
    X = torch.from_numpy(feat_matrix).unsqueeze(0)

    with torch.no_grad():
        al, fl = model(X, pad_mask=None)
        act_prob = torch.sigmoid(al.squeeze(0).squeeze(-1)).cpu().numpy()  # (E,)
        fraction = torch.sigmoid(fl.squeeze(0).squeeze(-1)).cpu().numpy()  # (E,)

    results = [
        (float(act_prob[i]), float(fraction[i]), src_id, tgt_id)
        for i, (src_id, tgt_id) in enumerate(edge_list)
    ]
    results.sort(key=lambda x: -x[0])
    return results


# ── sun-block helper (no main.py dependency) ──────────────────────────────

def _path_hits_sun(sx, sy, tx, ty, safety=1.5):
    dx, dy = tx - sx, ty - sy
    seg_sq = dx*dx + dy*dy
    if seg_sq < 1e-9:
        return math.hypot(50.0 - sx, 50.0 - sy) < 10.0 + safety
    t = ((50.0 - sx)*dx + (50.0 - sy)*dy) / seg_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(50.0 - (sx + t*dx), 50.0 - (sy + t*dy)) < 10.0 + safety


# ── move builder ───────────────────────────────────────────────────────────

def build_moves_from_scores(
        scored_edges: List[Tuple[float, float, int, int]],
        world: 'WorldModel',
        top_k:              int   = TOP_K_EDGES,
        prob_threshold:     float = ACTION_PROB_THRESHOLD,
        min_fraction:       float = MIN_FRACTION,
        max_frac_per_planet: float = MAX_FRACTION_PER_PLANET,
) -> List[List]:
    """
    Convert (action_prob, fraction, src, tgt) into actual move list.
    Uses world.plan_shot() for exact angles.  Respects ship budgets.
    """
    planet_map = {p.id: p for p in world.planets}
    spent      = defaultdict(int)
    moves: List[List] = []
    committed  = 0

    for act_prob, fraction, src_id, tgt_id in scored_edges:
        if committed >= top_k:
            break
        if act_prob < prob_threshold:
            break

        src = planet_map.get(src_id)
        tgt = planet_map.get(tgt_id)
        if src is None or tgt is None or src.owner != world.player:
            continue

        available = int(src.ships) - spent[src_id]
        if available <= 0 or fraction < min_fraction:
            continue

        send = max(1, int(available * min(max_frac_per_planet, fraction)))

        if _MAIN_AVAILABLE:
            shot = world.plan_shot(src_id, tgt_id, send)
            if shot is None:
                continue
            angle = shot[0]
        else:
            angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
            if _path_hits_sun(src.x, src.y, tgt.x, tgt.y):
                continue

        spent[src_id] += send
        moves.append([src_id, float(angle), send])
        committed += 1

    return moves


# ── agent entry point ──────────────────────────────────────────────────────

def agent(obs, config=None):
    """
    Drop-in replacement for main.py's agent().
    Falls back to rule-based agent if model is unavailable.
    """
    model = _get_model()
    if model is None or not _MAIN_AVAILABLE:
        if _MAIN_AVAILABLE:
            from main import agent as _fallback
            return _fallback(obs, config)
        return []
    try:
        return _agent_impl(obs, model)
    except Exception:
        if _MAIN_AVAILABLE:
            try:
                from main import agent as _fallback
                return _fallback(obs, config)
            except Exception:
                pass
        return []


def _agent_impl(obs, model: EdgeTransformerNet) -> List[List]:
    world = build_world(obs)
    if not world.my_planets:
        return []

    scored = score_all_edges(obs, model)
    if not scored:
        from main import _agent_impl as _main_impl
        return _main_impl(obs, None)

    return build_moves_from_scores(scored, world)


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import json, argparse

    parser = argparse.ArgumentParser(
        description='Inspect joint edge scores for a replay step.')
    parser.add_argument('replay', help='Replay JSON file')
    parser.add_argument('--step',  type=int, default=50)
    parser.add_argument('--model', type=str, default='')
    args = parser.parse_args()

    if args.model:
        os.environ['ORBIT_MODEL_PATH'] = args.model

    with open(args.replay) as f:
        data = json.load(f)

    rewards = data.get('rewards', [])
    max_r   = max(rewards) if rewards else -1
    if rewards.count(max_r) > 1 or max_r <= 0:
        print("No clear winner."); sys.exit(1)

    winner   = rewards.index(max_r)
    n_agents = len(data['steps'][0])
    obs      = data['steps'][args.step][winner]['observation']
    actual   = data['steps'][args.step][winner].get('action', [])

    print(f"Step {args.step}  winner=player{winner}  "
          f"{'1v1' if n_agents == 2 else f'{n_agents}p-FFA'}")
    print(f"Winner actual actions: {actual}")

    m = _get_model()
    if m is None:
        print("No model found."); sys.exit(1)

    scored = score_all_edges(obs, m)
    print(f"\nTotal candidate edges: {len(scored)}")
    print("Top-15 (joint-scored):")
    pmap = {p[0]: p for p in obs['planets']}
    for i, (ap, frac, src, tgt) in enumerate(scored[:15]):
        s = pmap.get(src, [0]*7)
        t = pmap.get(tgt, [0]*7)
        own = {-1: 'N', winner: 'ME'}.get(t[1], f'E{t[1]}')
        print(f"  {i+1:2d}. P={ap:.3f} frac={frac:.2f}  "
              f"src={src}({s[5]:.0f}sh) -> {tgt}({own},{t[5]:.0f}sh,p{t[6]})")
