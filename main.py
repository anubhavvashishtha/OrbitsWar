"""
main_rl.py
==========
Submission-ready Orbit Wars agent that uses the trained PPO policy.

Drop-in replacement — the competition only needs:
  • this file (main.py)
  • rl/features.py
  • rl/model.py
  • rl/checkpoints/best_policy.pt

Falls back to a safe "do nothing" move-set if the model can't be loaded,
so the submission never crashes on Kaggle.
"""

import os
import sys
import math

# ── Lazy imports so the agent loads fast on Kaggle ───────────────────────────
_MODEL      = None
_TORCH_OK   = False
_FEATURES   = None

_CKPT_PATH  = os.path.join(os.path.dirname(__file__), "rl", "checkpoints", "best_policy.pt")


def _ensure_loaded():
    global _MODEL, _TORCH_OK, _FEATURES

    if _MODEL is not None:
        return

    # Insert project root so rl.* imports work
    _root = os.path.dirname(os.path.abspath(__file__))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    try:
        import torch
        _TORCH_OK = True
    except ImportError:
        print("[RL Agent] torch not available — returning empty moves", flush=True)
        return

    try:
        from rl.model    import ActorCritic
        from rl.features import encode_obs, actions_to_moves

        model = ActorCritic()
        ckpt  = torch.load(_CKPT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        model.eval()

        _MODEL    = model
        _FEATURES = (encode_obs, actions_to_moves)
        print(f"[RL Agent] Policy loaded from {_CKPT_PATH} "
              f"(trained {ckpt.get('total_steps', '?'):,} steps)", flush=True)

    except Exception as exc:
        print(f"[RL Agent] Could not load model: {exc}", flush=True)


# ── Public agent function ────────────────────────────────────────────────────
def agent(obs):
    """
    Called once per turn by the Orbit Wars environment.

    Parameters
    ----------
    obs : observation dict (or Namespace) with fields:
          player, planets, fleets, angular_velocity, step, ...

    Returns
    -------
    moves : list of [from_planet_id, angle_radians, num_ships]
    """
    _ensure_loaded()

    if _MODEL is None:
        try:
            import fallback  # fallback to your original heuristic agent
            return fallback.agent(obs)
        except Exception:
            return []   # ultimate safe fallback

    try:
        import torch
        encode_obs, actions_to_moves = _FEATURES

        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player

        pt, ft, gt, my_mask, vm, pids, pships = encode_obs(obs, player)

        pt_t = torch.FloatTensor(pt).unsqueeze(0)
        ft_t = torch.FloatTensor(ft).unsqueeze(0)
        gt_t = torch.FloatTensor(gt).unsqueeze(0)
        vm_t = torch.BoolTensor(vm).unsqueeze(0)
        mm_t = torch.BoolTensor(my_mask).unsqueeze(0)

        with torch.no_grad():
            send_a, angle_a, frac_a, _, _ = _MODEL.act(
                pt_t, ft_t, gt_t, vm_t, mm_t,
                deterministic=True,   # greedy at test time
            )

        return actions_to_moves(send_a, angle_a, frac_a, pids, pships)

    except Exception as exc:
        print(f"[RL Agent] step error: {exc}", flush=True)
        return []
