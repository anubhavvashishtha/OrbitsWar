"""
rl/train_rl.py
==============
Main PPO training entry point.

Training curriculum (auto-advances by step count)
--------------------------------------------------
  Phase 1   0 – 200 K steps  →  vs "random" agent only
  Phase 2 200K – 600 K steps  →  random + main_1/2/3
  Phase 3 600K+               →  all above + self-play vs past checkpoints

Usage
-----
  python -m rl.train_rl                        # 1 M steps, fresh start
  python -m rl.train_rl --steps 500000         # custom step budget
  python -m rl.train_rl --resume rl/checkpoints/policy_step_200000.pt
"""

import os
import sys
import glob
import json
import random
import argparse
import importlib.util
import numpy as np
import torch
from datetime import datetime

# ── Allow running as "python -m rl.train_rl" from project root ───────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.features     import encode_obs, actions_to_moves
from rl.model        import ActorCritic
from rl.env_wrapper  import OrbitWarsEnv
from rl.ppo          import RolloutBuffer, PPO

# ── Directories ───────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_FILE       = os.path.join(os.path.dirname(__file__), "training_log.json")
ENEMY_DIR      = os.path.join(os.path.dirname(os.path.dirname(__file__)), "enemy agents")

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_STEPS        = 1024    # steps collected per PPO update
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
SAVE_EVERY     = 50_000  # steps between checkpoint saves


# ─────────────────────────────────────────────────────────────────────────────
def load_agent_file(filepath):
    """Dynamically import an agent function from a .py file."""
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    dir_path    = os.path.dirname(os.path.abspath(filepath))
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)
    spec   = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = os.path.abspath(filepath)
    spec.loader.exec_module(module)
    return module.agent


def make_selfplay_agent(ckpt_path, device):
    """Load a past checkpoint as a callable opponent agent."""
    from rl.model    import ActorCritic
    from rl.features import encode_obs, actions_to_moves

    sp_model = ActorCritic().to(device)
    ckpt     = torch.load(ckpt_path, map_location=device, weights_only=True)
    sp_model.load_state_dict(ckpt["model"])
    sp_model.eval()

    def _agent(obs):
        if isinstance(obs, dict):
            player = obs.get("player", 0)
        else:
            player = getattr(obs, "player", 0)
        pt, ft, gt, my_mask, vm, pids, pships = encode_obs(obs, player)
        pt_t  = torch.FloatTensor(pt).unsqueeze(0).to(device)
        ft_t  = torch.FloatTensor(ft).unsqueeze(0).to(device)
        gt_t  = torch.FloatTensor(gt).unsqueeze(0).to(device)
        vm_t  = torch.BoolTensor(vm).unsqueeze(0).to(device)
        mm_t  = torch.BoolTensor(my_mask).unsqueeze(0).to(device)
        send_a, angle_a, frac_a, _, _ = sp_model.act(pt_t, ft_t, gt_t, vm_t, mm_t,
                                                       deterministic=True)
        return actions_to_moves(send_a, angle_a, frac_a, pids, pships)

    return _agent


def choose_opponents(total_steps, enemy_agents, past_checkpoints, device, num_players):
    """
    Curriculum-based opponent selection.
    Returns a list of callable agents or the string "random".
    """
    if total_steps < 200_000:
        pool = ["random"]
    else:
        pool = ["random"] + enemy_agents
        if total_steps >= 600_000 and past_checkpoints:
            # Add self-play opponents from the last 5 checkpoints
            for ckpt in past_checkpoints[-5:]:
                pool.append(make_selfplay_agent(ckpt, device))

    return [random.choice(pool) for _ in range(num_players - 1)]


def get_phase(total_steps):
    if total_steps < 200_000:
        return "Phase 1: vs Random"
    elif total_steps < 600_000:
        return "Phase 2: vs Enemies"
    else:
        return "Phase 3: Self-Play"


# ─────────────────────────────────────────────────────────────────────────────
def collect_rollout(env: OrbitWarsEnv, model, buffer: RolloutBuffer,
                    n_steps: int, device):
    """
    Play n_steps of game experience, store in buffer.

    Returns
    -------
    last_value   : float  (bootstrap value for GAE)
    ep_rewards   : list[float]
    ep_wins      : int
    ep_count     : int
    """
    obs            = env.reset()
    ep_rewards     = []
    ep_reward      = 0.0
    ep_wins        = 0
    ep_count       = 0
    player         = (obs.get("player", 0) if isinstance(obs, dict)
                      else getattr(obs, "player", 0))

    for _ in range(n_steps):
        pt, ft, gt, my_mask, vm, pids, pships = encode_obs(obs, player)

        pt_t  = torch.FloatTensor(pt).unsqueeze(0).to(device)
        ft_t  = torch.FloatTensor(ft).unsqueeze(0).to(device)
        gt_t  = torch.FloatTensor(gt).unsqueeze(0).to(device)
        vm_t  = torch.BoolTensor(vm).unsqueeze(0).to(device)
        mm_t  = torch.BoolTensor(my_mask).unsqueeze(0).to(device)

        send_a, angle_a, frac_a, log_prob, value = model.act(
            pt_t, ft_t, gt_t, vm_t, mm_t
        )

        moves = actions_to_moves(send_a, angle_a, frac_a, pids, pships)

        obs, reward, done, _ = env.step(moves)
        ep_reward += reward

        buffer.add(
            pt, ft, gt, vm, my_mask,
            send_a, angle_a, frac_a,
            log_prob, value,
            reward, done,
        )

        if done:
            ep_rewards.append(ep_reward)
            if reward > 0:
                ep_wins += 1
            ep_count  += 1
            ep_reward  = 0.0
            obs        = env.reset()
            player     = (obs.get("player", 0) if isinstance(obs, dict)
                          else getattr(obs, "player", 0))

    # Bootstrap value for the last (possibly unfinished) episode
    pt, ft, gt, _, vm, _, _ = encode_obs(obs, player)
    pt_t = torch.FloatTensor(pt).unsqueeze(0).to(device)
    ft_t = torch.FloatTensor(ft).unsqueeze(0).to(device)
    gt_t = torch.FloatTensor(gt).unsqueeze(0).to(device)
    vm_t = torch.BoolTensor(vm).unsqueeze(0).to(device)

    with torch.no_grad():
        last_value = model.get_value(pt_t, ft_t, gt_t, vm_t).item()

    return last_value, ep_rewards, ep_wins, ep_count


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Orbit Wars PPO Trainer")
    parser.add_argument("--steps",    type=int,   default=1_000_000,
                        help="Total environment steps to train for")
    parser.add_argument("--n_steps",  type=int,   default=N_STEPS,
                        help="Steps per PPO rollout")
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--resume",   type=str,   default=None,
                        help="Path to a checkpoint .pt file to resume training")
    parser.add_argument("--no_shape", action="store_true",
                        help="Disable reward shaping (sparse reward only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Load enemy agents ─────────────────────────────────────────────────────
    enemy_files  = sorted(glob.glob(os.path.join(ENEMY_DIR, "*.py")))
    enemy_files  = [f for f in enemy_files if not os.path.basename(f).startswith("__")]
    enemy_agents = []
    for f in enemy_files:
        try:
            enemy_agents.append(load_agent_file(f))
            print(f"  [OK] Loaded enemy: {os.path.basename(f)}")
        except Exception as e:
            print(f"  [WARN] Could not load {os.path.basename(f)}: {e}")

    # ── Initialise model & PPO ────────────────────────────────────────────────
    model = ActorCritic().to(device)
    ppo   = PPO(model, lr=args.lr, device=device)

    total_steps = 0
    logs        = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            ppo.optimizer.load_state_dict(ckpt["optimizer"])
        total_steps = ckpt.get("total_steps", 0)
        print(f"[Resume] from {args.resume}  (step {total_steps:,})")

    target_steps   = total_steps + args.steps
    update_count   = 0

    print("=" * 65)
    print(f"[START] Orbit Wars PPO -- training for {args.steps:,} steps")
    print(f"   Rollout size : {args.n_steps} steps / update")
    print(f"   Reward shape : {'ON' if not args.no_shape else 'OFF'}")
    print("=" * 65)

    while total_steps < target_steps:
        past_checkpoints = sorted(
            glob.glob(os.path.join(CHECKPOINT_DIR, "policy_step_*.pt"))
        )

        # 50% chance of 4-player game to learn both strategies simultaneously
        num_players = 4 if random.random() < 0.5 else 2

        opps  = choose_opponents(total_steps, enemy_agents, past_checkpoints, device, num_players)
        phase = get_phase(total_steps)

        env    = OrbitWarsEnv(opponents=opps, reward_shaping=not args.no_shape)
        buffer = RolloutBuffer(args.n_steps)

        last_val, ep_rewards, ep_wins, ep_count = collect_rollout(
            env, model, buffer, args.n_steps, device
        )

        stats        = ppo.update(buffer, last_val, GAMMA, GAE_LAMBDA)
        total_steps += args.n_steps
        update_count += 1

        win_rate    = ep_wins / max(ep_count, 1)
        mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0

        opp_names = [(o if isinstance(o, str) else getattr(o, "__name__", "checkpoint")) for o in opps]
        opp_name_str = " + ".join(opp_names) if len(opp_names) <= 3 else f"mixed_x{len(opp_names)}"

        log_entry = {
            "step"       : total_steps,
            "update"     : update_count,
            "phase"      : phase,
            "mode"       : f"{num_players}P",
            "opponent"   : opp_name_str,
            "episodes"   : ep_count,
            "win_rate"   : round(win_rate, 3),
            "mean_reward": round(mean_reward, 4),
            **{k: round(v, 5) for k, v in stats.items()},
            "time"       : datetime.now().isoformat(),
        }
        logs.append(log_entry)

        print(f"[{total_steps:>9,}] {phase:<20} | {num_players}P | "
              f"eps={ep_count:>3} win={win_rate:.0%} "
              f"r={mean_reward:+.4f} | "
              f"pg={stats['pg_loss']:.4f} "
              f"v={stats['v_loss']:.4f} "
              f"ent={stats['ent_loss']:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if total_steps % SAVE_EVERY < args.n_steps:
            ckpt_path = os.path.join(CHECKPOINT_DIR,
                                     f"policy_step_{total_steps}.pt")
            torch.save({
                "model"      : model.state_dict(),
                "optimizer"  : ppo.optimizer.state_dict(),
                "total_steps": total_steps,
            }, ckpt_path)

            # Save as best if win rate is good
            if win_rate >= 0.5:
                torch.save({"model": model.state_dict(),
                            "total_steps": total_steps},
                           os.path.join(CHECKPOINT_DIR, "best_policy.pt"))

            print(f"  [SAVE] Checkpoint --> {ckpt_path}")

        # ── Log file ──────────────────────────────────────────────────────────
        with open(LOG_FILE, "w") as f:
            json.dump(logs, f, indent=2)

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINT_DIR,
                              f"policy_step_{total_steps}_final.pt")
    torch.save({"model": model.state_dict(), "total_steps": total_steps},
               final_path)
    # Always save as best at the end of training
    torch.save({"model": model.state_dict(), "total_steps": total_steps},
               os.path.join(CHECKPOINT_DIR, "best_policy.pt"))

    print("\n" + "=" * 65)
    print(f"[DONE] Training complete!  Total steps: {total_steps:,}")
    print(f"   Final model  --> {final_path}")
    print(f"   Best policy  --> {CHECKPOINT_DIR}/best_policy.pt")
    print(f"   Training log --> {LOG_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
