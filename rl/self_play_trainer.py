"""
rl/self_play_trainer.py
=======================
Fine-tune OrbitsNet via PPO self-play against enemy agents.

Architecture: actor-critic (shared trunk, value head + policy head).
The policy head picks one of 6 strategic intents each turn.
The chosen intent soft-adjusts scoring multipliers in main.py by
monkey-patching the loaded module's constants before each agent call,
then restoring them — no changes to main.py required.

Intent table
------------
  0  EXPAND        — prioritise safe / contested neutral planets
  1  ATTACK_WEAK   — focus on the enemy with fewest ships
  2  ATTACK_LEADER — FFA: focus resources on the current leader
  3  DEFEND        — prioritise rescue and reinforcement of own planets
  4  CONSOLIDATE   — pull ships to rear, minimal aggression
  5  SWARM         — multi-source coordinated attacks & crash exploits

Each round:
  1. Play GAMES_PER_ROUND FFA games — player 0 uses PPO policy,
     enemies use their fixed heuristics.
  2. For every step, sample intent from policy, patch main module
     constants, call agent, store (state, action, log_prob, reward,
     done, value) in RolloutBuffer.
  3. PPO clip loss update over the collected rollout.
  4. Save checkpoint to rl/model.pt.

Run AFTER replay_trainer.py for best results (that pretrains the value
head on 1732 Kaggle replays before PPO starts).

Usage
-----
    python rl/self_play_trainer.py
    python rl/self_play_trainer.py --rounds 20 --games 12
"""
import os, sys, glob, random, importlib.util, argparse, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classifier"))
sys.path.insert(0, str(PROJECT_ROOT))

from buffer import RolloutBuffer

MODEL_PATH      = ROOT / "last.pt"
BEST_MODEL_PATH = ROOT / "best.pt"
ENEMY_DIR       = PROJECT_ROOT / "enemy agents"
MAIN_PY         = PROJECT_ROOT / "main.py"

# ── Hyper-parameters ─────────────────────────────────────────────────────────
GAMES_PER_ROUND = 8
N_ROUNDS        = 100000  # Continuous loop
LR              = 3e-4
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_EPS        = 0.2
VAL_COEF        = 0.5
PPO_EPOCHS      = 4
BATCH_SIZE      = 256
SNAPSHOT_EVERY  = 1      # capture every turn (TODO #4 — dense transitions)

# SAC-style adaptive entropy — replaces fixed ENT_COEF
# log(N_ACTIONS) * 0.4 ≈ 0.72 nats keeps reasonable intent diversity
TARGET_ENTROPY  = math.log(6) * 0.4
ALPHA_LR        = 3e-4

# ── Intent names (for logging) ────────────────────────────────────────────────
INTENT_NAMES = [
    "EXPAND", "ATTACK_WEAK", "ATTACK_LEADER",
    "DEFEND", "CONSOLIDATE", "SWARM",
]

# Default multiplier values from main.py (restored after each call)
_DEFAULTS = {
    "SAFE_NEUTRAL_VALUE_MULT":      1.2,
    "CONTESTED_NEUTRAL_VALUE_MULT": 0.7,
    "HOSTILE_TARGET_VALUE_MULT":    1.85,
    "REINFORCE_VALUE_MULT":         1.35,
    "SWARM_VALUE_MULT":             1.05,
    "SWARM_SCORE_MULT":             1.06,
    "CRASH_EXPLOIT_VALUE_MULT":     1.18,
}

# Per-intent multiplier overrides — soft adjustments, not hard gates
INTENT_MULTS = [
    # 0 EXPAND: neutral-first
    {"SAFE_NEUTRAL_VALUE_MULT": 2.0, "CONTESTED_NEUTRAL_VALUE_MULT": 1.1,
     "HOSTILE_TARGET_VALUE_MULT": 0.75},
    # 1 ATTACK_WEAK: aggressive vs weakest enemy
    {"HOSTILE_TARGET_VALUE_MULT": 2.5, "SAFE_NEUTRAL_VALUE_MULT": 0.7,
     "CONTESTED_NEUTRAL_VALUE_MULT": 0.5},
    # 2 ATTACK_LEADER: FFA focus on leader (main.py already has 1.20× for leader planets)
    {"HOSTILE_TARGET_VALUE_MULT": 2.2, "SAFE_NEUTRAL_VALUE_MULT": 0.8,
     "CONTESTED_NEUTRAL_VALUE_MULT": 0.6},
    # 3 DEFEND: reinforce and rescue priority
    {"REINFORCE_VALUE_MULT": 2.5, "HOSTILE_TARGET_VALUE_MULT": 0.5,
     "SAFE_NEUTRAL_VALUE_MULT": 0.6, "CONTESTED_NEUTRAL_VALUE_MULT": 0.4},
    # 4 CONSOLIDATE: pull ships back, minimal aggression
    {"REINFORCE_VALUE_MULT": 2.2, "HOSTILE_TARGET_VALUE_MULT": 0.4,
     "SAFE_NEUTRAL_VALUE_MULT": 0.4, "CONTESTED_NEUTRAL_VALUE_MULT": 0.3},
    # 5 SWARM: coordinated multi-source attacks
    {"SWARM_VALUE_MULT": 1.9, "SWARM_SCORE_MULT": 1.5,
     "CRASH_EXPLOIT_VALUE_MULT": 1.8, "HOSTILE_TARGET_VALUE_MULT": 1.65},
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_agent(path, force=False):
    name = Path(path).stem
    d    = str(Path(path).parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    if force:
        for k in [name, "value_fn", "main"]:
            sys.modules.pop(k, None)
    spec = importlib.util.spec_from_file_location(name, str(Path(path).resolve()))
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.agent, name


def snap(raw_p, comet_ids=None):
    cids = comet_ids or set()
    return [{"id": p[0], "owner": p[1], "ships": p[5],
             "production": p[6], "x": p[2], "y": p[3]}
            for p in raw_p if p[0] not in cids]


def _read(obs, key, default=None):
    if isinstance(obs, dict): return obs.get(key, default)
    return getattr(obs, key, default)


def _patch_main(main_mod, intent: int):
    """Apply intent multipliers to the main module; returns original values."""
    orig = {}
    for k, v in INTENT_MULTS[intent].items():
        orig[k] = getattr(main_mod, k, _DEFAULTS.get(k))
        setattr(main_mod, k, v)
    return orig


def _restore_main(main_mod, orig: dict):
    for k, v in orig.items():
        if v is not None:
            setattr(main_mod, k, v)


# ── Reward shaping (event-based + small momentum) ─────────────────────────────
#
# In-game rewards (per-turn, signed; cumulative sum over 500 turns is roughly
# the same order of magnitude as the terminal win/loss=1/0 signal):
#
#   +CAPTURE_BASE * production         on capturing a planet (any prior owner)
#   +CAPTURE_FROM_ENEMY * production   extra bonus when prior owner was an enemy
#                                       (capturing from an enemy is worth more
#                                        than grabbing a neutral)
#   -PLANET_LOSS * production          on losing one of my planets
#   +ELIMINATION                       per opponent eliminated this turn
#                                       (they have 0 planets + 0 fleets left)
#   +MOMENTUM_SCALE * Δshares          small continuous signal so flat-but-
#                                       improving turns also get some credit
#
# These constants are tuned so a typical winning trajectory accumulates ~1.0
# total shaped reward (matching the terminal scale), which means the early
# game is no longer credit-starved.
CAPTURE_BASE        = 0.06   # per planet captured, scaled by its production
CAPTURE_FROM_ENEMY  = 0.04   # extra on top when prior owner was an enemy
PLANET_LOSS         = 0.06   # symmetric penalty for losing a planet
ELIMINATION_BONUS   = 0.30   # per opponent fully knocked out this turn
MOMENTUM_SCALE      = 0.02   # weight on continuous share-delta term


def _game_stats(sd, player):
    """Sum my/total ships, production, planet count from a snap list."""
    my_ships = my_prod = my_planets = total_ships = total_prod = 0.0
    total_planets = float(len(sd))
    for p in sd:
        sh = float(p.get("ships", 0))
        pr = float(p.get("production", 0))
        total_ships += sh
        total_prod  += pr
        if p.get("owner") == player:
            my_ships   += sh
            my_prod    += pr
            my_planets += 1.0
    return (my_ships, my_prod, my_planets, total_ships, total_prod, total_planets)


def _alive_enemies(sd, raw_fleets, player):
    """Owners that still have any planet or any in-flight fleet."""
    alive = set()
    for p in sd:
        ow = p.get("owner", -1)
        if ow != -1 and ow != player:
            alive.add(ow)
    for f in (raw_fleets or []):
        fow = f[1] if isinstance(f, (list, tuple)) else f.get("owner", -1)
        if fow != -1 and fow != player:
            alive.add(fow)
    return alive


def _event_reward(prev_sd, curr_sd, prev_alive, curr_alive, player):
    """Capture/loss/elimination events between two consecutive snapshots."""
    if prev_sd is None:
        return 0.0
    prev_owner = {p["id"]: p.get("owner", -1)        for p in prev_sd}
    prev_prod  = {p["id"]: float(p.get("production", 0)) for p in prev_sd}
    r = 0.0
    for cp in curr_sd:
        pid = cp["id"]
        if pid not in prev_owner:
            continue
        po = prev_owner[pid]
        co = cp.get("owner", -1)
        if co == po:
            continue
        prod = max(prev_prod.get(pid, 1.0), float(cp.get("production", 0)))
        if co == player and po != player:
            r += CAPTURE_BASE * prod
            if po != -1:
                r += CAPTURE_FROM_ENEMY * prod
        elif po == player and co != player:
            r -= PLANET_LOSS * prod
    eliminated = prev_alive - curr_alive
    r += ELIMINATION_BONUS * len(eliminated)
    return r


def _shaped_reward(curr_stats, prev_stats, prev_sd, curr_sd,
                   prev_alive, curr_alive, player):
    """Event-based dense reward + small momentum term."""
    if prev_stats is None or prev_sd is None:
        return 0.0
    event_r = _event_reward(prev_sd, curr_sd, prev_alive, curr_alive, player)

    my_s, my_pr, my_pl, tot_s, tot_pr, tot_pl = curr_stats
    ps,   ppr,   ppl,   _,     _,      _      = prev_stats
    d_ships   = (my_s  - ps)  / max(1.0, tot_s)
    d_prod    = (my_pr - ppr) / max(1.0, tot_pr)
    d_planets = (my_pl - ppl) / max(1.0, tot_pl)
    momentum = MOMENTUM_SCALE * (0.4 * d_ships + 0.4 * d_prod + 0.2 * d_planets)
    return event_r + momentum


# ── Game runner ───────────────────────────────────────────────────────────────
def play_game_collect(model, agent_fns, seed: int = 0):
    """
    Play one FFA game. Player 0 uses the PPO policy to pick intents.
    Returns (RolloutBuffer for player 0, outcome string).
    """
    from train_from_replays import extract_features
    from kaggle_environments import make

    n       = len(agent_fns)
    env     = make("orbit_wars",
                   configuration={"seed": seed, "randomSeed": seed},
                   debug=False)
    env.reset(num_agents=n)

    buf        = RolloutBuffer(gamma=GAMMA, lam=GAE_LAMBDA)
    main_mod   = sys.modules.get("main")
    prev_stats = None   # for shaped reward computation
    prev_sd    = None
    prev_alive = set()

    states = env.step([[]] * n)
    step   = 0

    while True:
        st0    = states[0]
        status = str(_read(st0, "status", "ACTIVE"))
        if status != "ACTIVE":
            break

        obss = [_read(s, "observation") for s in states]

        # ── Feature extraction & intent sampling for player 0 ──────────────
        intent       = 0
        log_prob_val = 0.0
        value_val    = 0.5
        feats        = None
        shaped_r     = 0.0

        if obss[0] is not None and step % SNAPSHOT_EVERY == 0:
            rp = _read(obss[0], "planets", []) or []
            ci = set(_read(obss[0], "comet_planet_ids", []) or [])
            rf = _read(obss[0], "fleets", []) or []
            sd = snap(rp, ci)
            if sd:
                curr_stats = _game_stats(sd, 0)
                curr_alive = _alive_enemies(sd, rf, 0)
                shaped_r = _shaped_reward(
                    curr_stats, prev_stats,
                    prev_sd, sd,
                    prev_alive, curr_alive,
                    player=0,
                )
                prev_stats = curr_stats
                prev_sd    = sd
                prev_alive = curr_alive

                n_p = max(2, len({p[1] for p in rp if p[1] != -1}))
                feats = extract_features(sd, 0, step, n_p, fleets=rf)[:30]
                with torch.no_grad():
                    model.eval()
                    ft               = torch.FloatTensor(feats).unsqueeze(0)
                    value_t, logits_t = model(ft)
                    dist             = torch.distributions.Categorical(logits=logits_t)
                    intent_t         = dist.sample()
                    intent           = int(intent_t.item())
                    log_prob_val     = float(dist.log_prob(intent_t).item())
                    value_val        = float(value_t.item())

        # ── Patch main module constants for chosen intent ──────────────────
        orig = {}
        if main_mod is not None:
            orig = _patch_main(main_mod, intent)

        # ── Execute all agents ─────────────────────────────────────────────
        actions = []
        for i, obs in enumerate(obss):
            try:    a = agent_fns[i](obs)
            except: a = []
            actions.append(a or [])

        if main_mod is not None:
            _restore_main(main_mod, orig)

        # ── Store transition with shaped reward ────────────────────────────
        if feats is not None:
            buf.add(feats, intent, log_prob_val, shaped_r, False, value_val)

        states = env.step(actions)
        step  += 1

    # ── Override terminal step reward with win/loss ────────────────────────
    rewards = [float(_read(s, "reward", 0) or 0) for s in states]
    max_r   = max(rewards)
    winner  = -1 if max_r <= 0 else rewards.index(max_r)

    if buf.rewards:
        buf.rewards[-1] = 1.0 if winner == 0 else 0.0
        buf.dones[-1]   = True

    outcome = "WIN" if winner == 0 else ("DRAW" if winner < 0 else "LOSS")
    return buf, outcome


# ── PPO update ────────────────────────────────────────────────────────────────
def ppo_update(model, optimizer, buf: RolloutBuffer,
               log_alpha=None, alpha_optimizer=None):
    """
    PPO clip loss over one rollout buffer (one game's worth of data).
    log_alpha / alpha_optimizer: SAC-style adaptive entropy temperature.
    """
    if len(buf) < 2:
        return 0.0, 0.0

    states, actions, old_log_probs, returns, advantages = buf.as_arrays()

    X    = torch.FloatTensor(states)
    A    = torch.LongTensor(actions)
    oldp = torch.FloatTensor(old_log_probs)
    R    = torch.FloatTensor(returns)
    Adv  = torch.FloatTensor(advantages)
    # Normalise advantages for stable training
    Adv  = (Adv - Adv.mean()) / (Adv.std() + 1e-8)

    idx_list      = list(range(len(X)))
    total_loss    = 0.0
    total_entropy = 0.0
    n_updates     = 0

    model.train()
    for _ in range(PPO_EPOCHS):
        random.shuffle(idx_list)
        for start in range(0, len(idx_list), BATCH_SIZE):
            batch = idx_list[start:start + BATCH_SIZE]
            bx    = X[batch]

            values, logits = model(bx)
            dist       = torch.distributions.Categorical(logits=logits)
            new_log_p  = dist.log_prob(A[batch])
            entropy    = dist.entropy().mean()

            # Current temperature — detach so alpha grad is separate
            alpha = log_alpha.exp().detach() if log_alpha is not None else 0.05

            ratio   = torch.exp(new_log_p - oldp[batch])
            adv_b   = Adv[batch]
            L_clip  = torch.min(
                ratio * adv_b,
                torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b
            ).mean()
            L_val   = F.mse_loss(values, R[batch])
            loss    = -L_clip + VAL_COEF * L_val - alpha * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            # Alpha update: push entropy toward TARGET_ENTROPY
            if log_alpha is not None and alpha_optimizer is not None:
                alpha_loss = -(log_alpha * (entropy.detach() - TARGET_ENTROPY))
                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()

            total_loss    += loss.item()
            total_entropy += entropy.item()
            n_updates     += 1

    return total_loss / max(1, n_updates), total_entropy / max(1, n_updates)


# ── Main loop ─────────────────────────────────────────────────────────────────
def train(n_rounds: int = N_ROUNDS, games_per_round: int = GAMES_PER_ROUND):
    from net import OrbitsNet

    model = OrbitsNet()
    def load_ckpt(path, model):
        ckpt = torch.load(path, weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
            return ckpt.get("best_wr", 0.0), ckpt.get("epoch", 0)
        else:
            model.load_state_dict(ckpt)
            return 0.0, 0

    best_wr = 0.0
    start_rnd = 1

    load_from = MODEL_PATH if MODEL_PATH.exists() else BEST_MODEL_PATH
    if load_from.exists():
        try:
            best_wr, start_rnd = load_ckpt(load_from, model)
            print(f"Loaded model from {load_from} (Epoch {start_rnd}, Best WR: {best_wr:.1%})")
        except Exception as e:
            print(f"Architecture mismatch ({e}) — starting from random weights.")
    else:
        # Fallback to model.pt if pretraining just finished
        legacy_path = ROOT / "model.pt"
        if legacy_path.exists():
            try:
                load_ckpt(legacy_path, model)
                print(f"Loaded pretrained model from {legacy_path}")
            except Exception as e:
                pass
        else:
            print("No last.pt or model.pt found — starting from random weights.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"OrbitsNet params: {total_params:,}")

    optimizer      = optim.Adam(model.parameters(), lr=LR)

    # SAC-style adaptive entropy temperature
    log_alpha      = nn.Parameter(torch.tensor(math.log(0.05)))
    alpha_optimizer = optim.Adam([log_alpha], lr=ALPHA_LR)

    # Restore log_alpha from checkpoint if available
    _la_path = MODEL_PATH if MODEL_PATH.exists() else BEST_MODEL_PATH
    if _la_path.exists():
        try:
            _ckpt = torch.load(_la_path, weights_only=False)
            if isinstance(_ckpt, dict) and "log_alpha" in _ckpt:
                log_alpha.data.fill_(_ckpt["log_alpha"])
                print(f"  Restored alpha={log_alpha.exp().item():.4f} from checkpoint")
        except Exception:
            pass

    my_fn, my_name = load_agent(str(MAIN_PY), force=True)

    _std_paths = [p for p in glob.glob(str(ENEMY_DIR / "*" / "main.py"))
                  if Path(p).parent.name != "my"]
    _my_paths  = [p for p in glob.glob(str(ENEMY_DIR / "my" / "*.py"))
                  if not Path(p).name.startswith("__")]
    enemies = []
    for ef in sorted(_std_paths + _my_paths):
        p    = Path(ef)
        label = p.stem if p.parent.name == "my" else p.parent.name
        try:
            fn, name = load_agent(ef)
            enemies.append((label, fn))
            print(f"  Loaded enemy: {label}")
        except Exception as e:
            print(f"  WARN {label}: {e}")

    if not enemies:
        print(f"No enemy agents found in {ENEMY_DIR}"); return

    print(f"\nPPO self-play: {n_rounds} rounds × {games_per_round} games")
    print(f"Intents: {', '.join(f'{i}={n}' for i, n in enumerate(INTENT_NAMES))}")
    print(f"Adaptive entropy: target={TARGET_ENTROPY:.3f} nats, init_alpha={log_alpha.exp().item():.4f}")
    print("=" * 60)

    no_improve = 0
    rnd = start_rnd - 1  # guard: defined even if loop body never executes

    for rnd in range(start_rnd, start_rnd + n_rounds):
        wins = losses = draws = 0
        round_loss = 0.0
        round_entropy = 0.0
        intent_counts = [0] * len(INTENT_NAMES)

        for g in range(games_per_round):
            fillers = random.choices(enemies, k=min(3, len(enemies)))
            fns     = [my_fn] + [fn for _, fn in fillers]
            seed    = 42 + rnd * 1000 + g

            try:
                buf, outcome = play_game_collect(model, fns, seed=seed)
            except Exception as e:
                print(f"\n  game error: {e}"); continue

            # tally intent distribution for this game
            for a in buf.actions:
                intent_counts[a] += 1

            loss, entropy = ppo_update(model, optimizer, buf, log_alpha, alpha_optimizer)
            round_loss += loss
            round_entropy += entropy
            buf.clear()

            if   outcome == "WIN":  wins   += 1
            elif outcome == "LOSS": losses += 1
            else:                   draws  += 1

            sys.stdout.write(
                f"\r  Round {rnd}/{n_rounds}  game {g+1}/{games_per_round}"
                f"  W:{wins} L:{losses} D:{draws}"
                f"  ppo_loss={round_loss/(g+1):.4f}  "
            )
            sys.stdout.flush()

        wr       = wins / games_per_round
        avg_loss = round_loss / games_per_round

        avg_entropy = round_entropy / games_per_round

        # Intent distribution summary
        total_steps = sum(intent_counts)
        if total_steps > 0:
            intent_pct = "  ".join(
                f"{INTENT_NAMES[i][0]}:{100*c/total_steps:.0f}%"
                for i, c in enumerate(intent_counts)
            )
        else:
            intent_pct = "no steps"

        alpha_val = log_alpha.exp().item()
        checkpoint_data = {
            "epoch": rnd,
            "best_wr": max(wr, best_wr),
            "loss": avg_loss,
            "entropy": avg_entropy,
            "log_alpha": log_alpha.item(),
            "state_dict": model.state_dict()
        }
        
        torch.save(checkpoint_data, MODEL_PATH)

        if wr > best_wr:
            best_wr    = wr
            no_improve = 0
            torch.save(checkpoint_data, BEST_MODEL_PATH)
            tag = f"  *** best so far ({wr:.1%}) → best.pt"
        else:
            no_improve += 1
            tag = f"  (no improvement, best={best_wr:.1%}, streak={no_improve})"
            # We explicitly do NOT reload the best model here, as it traps the RL 
            # agent in high-variance local minima and prevents exploration.

        print(f"\n  -> WR={wr:.1%}  avg_ppo_loss={avg_loss:.4f}  avg_ent={avg_entropy:.4f}  alpha={alpha_val:.4f}{tag}")
        print(f"     intents: {intent_pct}")

    print(f"\nPPO self-play complete.  Best WR={best_wr:.1%}")
    if BEST_MODEL_PATH.exists():
        load_ckpt(BEST_MODEL_PATH, model)
        torch.save({
            "epoch": rnd,
            "best_wr": best_wr,
            "loss": avg_loss,
            "entropy": avg_entropy,
            "log_alpha": log_alpha.item(),
            "state_dict": model.state_dict()
        }, MODEL_PATH)
        print(f"  Final last.pt ← best checkpoint ({best_wr:.1%})")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=N_ROUNDS,        help="Training rounds")
    p.add_argument("--games",  type=int, default=GAMES_PER_ROUND, help="Games per round")
    args = p.parse_args()
    train(n_rounds=args.rounds, games_per_round=args.games)
