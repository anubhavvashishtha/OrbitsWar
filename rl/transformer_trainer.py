"""
rl/transformer_trainer.py
=========================
PPO fine-tuning for EdgeTransformerNet (actor) + TransformerCritic (critic).

Unlike self_play_trainer.py (which picks one of 6 abstract intents via a
small MLP), this trainer lets the EdgeTransformerNet directly score every
(src, tgt) edge and sample one as the "primary" action for PPO credit
assignment.  The same model then executes all edges above a threshold for
the actual game moves.

Actor  : EdgeTransformerNet   (imitation/model.py)
         Input : (E, 81) edge features for the current turn
         Output: per-edge logits → Categorical → sample primary edge
         Pretrained from: imitation/checkpoints/best_model.pt

Critic : TransformerCritic     (lightweight transformer on edge features)
         Input : (E, 81) edge feature matrix — same as actor
         Output: scalar V(s)  (no Sigmoid — unbounded for proper GAE)
         Warm-started from: rl/last.pt OrbitsNet trunk weights (if available)

PPO credit assignment
---------------------
  Primary edge = the single edge sampled from Categorical(softmax(logits)).
  This edge's log_prob is used for the PPO clipped ratio.
  Agent execution = top-K edges above ACTION_THRESHOLD (the actor still
  moves multiple fleets per turn — PPO just learns via the primary edge).

Usage
-----
    python rl/transformer_trainer.py
    python rl/transformer_trainer.py --rounds 50 --games 8
    python rl/transformer_trainer.py --actor  imitation/checkpoints/best_model.pt
    python rl/transformer_trainer.py --critic rl/last.pt
"""

import sys, glob, os, random, argparse, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ROOT          = Path(__file__).parent
PROJECT_ROOT  = ROOT.parent
IMITATION_DIR = PROJECT_ROOT / "imitation"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classifier"))
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(IMITATION_DIR))

# ── imports from imitation pipeline ──────────────────────────────────────
from model    import EdgeTransformerNet, load_model as load_actor
from features import build_all_candidate_edges

# ── imports from RL helpers ───────────────────────────────────────────────
from self_play_trainer import (
    load_agent, snap, _read,
    _shaped_reward, _game_stats, _alive_enemies,
    ENEMY_DIR, MAIN_PY,
    GAMMA, GAE_LAMBDA,
    CAPTURE_BASE, CAPTURE_FROM_ENEMY, PLANET_LOSS,
    ELIMINATION_BONUS, MOMENTUM_SCALE,
)

# ── paths ─────────────────────────────────────────────────────────────────
DEFAULT_ACTOR_PATH  = IMITATION_DIR / "checkpoints" / "best_model.pt"
DEFAULT_CRITIC_PATH = ROOT / "last.pt"      # OrbitsNet checkpoint (for warm-start)
SAVE_PATH           = ROOT / "transformer_last.pt"
BEST_PATH           = ROOT / "transformer_best.pt"

# ── hyper-parameters ──────────────────────────────────────────────────────
N_ROUNDS        = 100_000
GAMES_PER_ROUND = 16
LR_ACTOR        = 5e-5    # low — actor is pretrained; fine-tune carefully
LR_CRITIC       = 3e-4
CLIP_EPS        = 0.2
VAL_COEF        = 0.5
PPO_EPOCHS      = 4
MINI_BATCH      = 64      # turns per mini-batch (variable E handled via loop)
ACTION_THRESHOLD = 0.35   # execute edges above this; only primary used for PPO
TOP_K_EDGES      = 5

ENT_COEF = 0.01  # fixed entropy bonus — adaptive alpha kept hitting its ceiling

# ── Critic warm-up ─────────────────────────────────────────────────────────
# Fresh critic gives noisy value estimates → noisy advantages → bad PPO.
# For the first CRITIC_WARMUP_ROUNDS rounds the actor is frozen: only the
# critic is updated so it can calibrate V(s) before PPO fine-tunes the actor.
CRITIC_WARMUP_ROUNDS = 20


# ════════════════════════════════════════════════════════════════════════════
# CRITIC  — Transformer on the same (E, 81) edge features as the actor
# ════════════════════════════════════════════════════════════════════════════

class TransformerCritic(nn.Module):
    """
    Lightweight transformer critic that consumes the same (E, 81) edge
    feature matrix the actor already builds each turn.

    Architecture  (d_model=64, 2 encoder layers):
      4 component projectors  31/18/18/14 → 64 each
      fuse                    256 → 64
      TransformerEncoder      2 × (MHA + FFN)  d=64, heads=4, ffn=256
      mean-pool over edges    → (B, 64)
      value head              Linear(64,32) → ReLU → Linear(32,1)

    No Sigmoid — V(s) is unbounded so GAE can go negative.
    """
    _D = 64

    def __init__(self):
        super().__init__()
        D = self._D
        self.global_proj = nn.Sequential(nn.Linear(31, D), nn.LayerNorm(D), nn.GELU())
        self.src_proj    = nn.Sequential(nn.Linear(18, D), nn.LayerNorm(D), nn.GELU())
        self.tgt_proj    = nn.Sequential(nn.Linear(18, D), nn.LayerNorm(D), nn.GELU())
        self.edge_proj   = nn.Sequential(nn.Linear(14, D), nn.LayerNorm(D), nn.GELU())
        self.fuse        = nn.Sequential(nn.Linear(4 * D, D), nn.LayerNorm(D), nn.GELU())

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=4, dim_feedforward=D * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=2, enable_nested_tensor=False)

        self.value_head = nn.Sequential(
            nn.Linear(D, D // 2), nn.ReLU(), nn.Linear(D // 2, 1)
        )

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        """
        x        : (B, E, 81)  or (1, E, 81) for a single turn
        pad_mask : (B, E) bool — True = padding, excluded from mean-pool
        Returns  : (B,) scalar values
        """
        g = self.global_proj(x[..., :31])
        s = self.src_proj(x[..., 31:49])
        t = self.tgt_proj(x[..., 49:67])
        e = self.edge_proj(x[..., 67:81])
        h = self.fuse(torch.cat([g, s, t, e], dim=-1))          # (B, E, D)
        h = self.transformer(h, src_key_padding_mask=pad_mask)   # (B, E, D)

        if pad_mask is not None:
            real  = (~pad_mask).unsqueeze(-1).float()            # (B, E, 1)
            h_mean = (h * real).sum(1) / real.sum(1).clamp(min=1)
        else:
            h_mean = h.mean(1)                                   # (B, D)

        return self.value_head(h_mean).squeeze(-1)               # (B,)


# ════════════════════════════════════════════════════════════════════════════
# ROLLOUT BUFFER  (stores per-turn edge feature sets)
# ════════════════════════════════════════════════════════════════════════════

class TransformerRolloutBuffer:
    """
    Rollout buffer for EdgeTransformerNet-based PPO.

    Each transition stores the FULL turn edge-feature matrix (E, 81)
    so the actor can be re-run with gradients during the PPO update.
    This is memory-heavier than storing a single 30-dim state vector,
    but it allows the transformer to re-score with the updated weights.

    Action semantics
    ----------------
    chosen_idx : index into the edge_list of the primary sampled edge.
    log_prob   : log π(chosen_idx | turn_feats) at collection time.
    """

    def __init__(self, gamma: float = GAMMA, lam: float = GAE_LAMBDA):
        self.gamma = gamma
        self.lam   = lam
        self.clear()

    def clear(self):
        self.turn_feats  = []   # (E_i, 81) numpy arrays — shared by actor & critic
        self.chosen_idxs = []   # int
        self.log_probs   = []   # float
        self.rewards     = []   # float
        self.dones       = []   # float  0/1
        self.values      = []   # float  V(s)

    def add(self, turn_feats: np.ndarray, chosen_idx: int,
            log_prob: float, reward: float, done: bool, value: float):
        self.turn_feats.append(turn_feats.astype(np.float32))
        self.chosen_idxs.append(int(chosen_idx))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def compute_gae(self, last_value: float = 0.0):
        advantages, returns = [], []
        gae = 0.0
        next_val = last_value
        for r, d, v in zip(reversed(self.rewards),
                           reversed(self.dones),
                           reversed(self.values)):
            delta = r + self.gamma * next_val * (1.0 - d) - v
            gae   = delta + self.gamma * self.lam * (1.0 - d) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_val = v
        return returns, advantages

    def __len__(self):
        return len(self.rewards)


# ════════════════════════════════════════════════════════════════════════════
# GAME COLLECTION
# ════════════════════════════════════════════════════════════════════════════

def _build_moves(obs: dict, actor: EdgeTransformerNet, device: str):
    """
    Score all candidate edges with the actor.  Return:
      turn_feats  : (E, 81) numpy
      chosen_idx  : int   (primary sampled edge index)
      log_prob    : float
      moves       : list of [src, angle, ships] for the game env
    Returns (None, None, None, []) if no candidates.
    """
    import math as _math
    from main import build_world

    candidates = build_all_candidate_edges(obs)
    if not candidates:
        return None, None, None, []

    edge_list   = [(src, tgt) for src, tgt, _ in candidates]
    feat_matrix = np.array([f for _, _, f in candidates], dtype=np.float32)

    X = torch.from_numpy(feat_matrix).unsqueeze(0).to(device)  # (1, E, 81)

    with torch.no_grad():
        actor.eval()
        al, fl = actor(X, pad_mask=None)
        logits   = al.squeeze(0).squeeze(-1)                    # (E,)
        act_prob = torch.sigmoid(logits).cpu().numpy()          # (E,)
        fraction = torch.sigmoid(fl.squeeze(0).squeeze(-1)).cpu().numpy()

        dist       = torch.distributions.Categorical(logits=logits)
        chosen_t   = dist.sample()
        chosen_idx = int(chosen_t.item())
        log_prob   = float(dist.log_prob(chosen_t).item())

    # Build actual moves via physics (plan_shot)
    try:
        world      = build_world(obs)
        planet_map = {p.id: p for p in world.planets}
        spent      = defaultdict(int)
        moves      = []
        committed  = 0

        # Sort by action prob so primary edge goes first
        order = sorted(range(len(edge_list)), key=lambda i: -act_prob[i])
        for i in order:
            if committed >= TOP_K_EDGES:
                break
            if act_prob[i] < ACTION_THRESHOLD and i != chosen_idx:
                continue   # always include primary edge even if below threshold
            src_id, tgt_id = edge_list[i]
            src = planet_map.get(src_id)
            tgt = planet_map.get(tgt_id)
            if src is None or tgt is None or src.owner != world.player:
                continue
            available = int(src.ships) - spent[src_id]
            if available <= 0 or fraction[i] < 0.08:
                continue
            send = max(1, int(available * min(0.90, float(fraction[i]))))
            shot = world.plan_shot(src_id, tgt_id, send)
            if shot is None:
                continue
            angle = shot[0]
            spent[src_id] += send
            moves.append([src_id, float(angle), send])
            committed += 1
    except Exception:
        moves = []

    return feat_matrix, chosen_idx, log_prob, moves


def play_game_collect(actor: EdgeTransformerNet,
                      critic: TransformerCritic,
                      agent_fns: list,
                      seed: int,
                      device: str):
    """
    Play one FFA game.  Player 0 uses the actor to pick and execute edges.
    Returns (TransformerRolloutBuffer, outcome_str).
    """
    from kaggle_environments import make

    n   = len(agent_fns)
    env = make("orbit_wars",
               configuration={"seed": seed, "randomSeed": seed},
               debug=False)
    env.reset(num_agents=n)

    buf        = TransformerRolloutBuffer(gamma=GAMMA, lam=GAE_LAMBDA)
    prev_stats = prev_sd = None
    prev_alive: set = set()

    states = env.step([[]] * n)
    step   = 0

    while True:
        st0    = states[0]
        status = str(_read(st0, "status", "ACTIVE"))
        if status != "ACTIVE":
            break

        obss = [_read(s, "observation") for s in states]

        # ── Player 0: actor decides moves ─────────────────────────────────
        p0_moves   = []
        store_this = False
        turn_feats = chosen_idx = log_prob_val = value_val = None
        shaped_r = 0.0

        if obss[0] is not None:
            rp = _read(obss[0], "planets", []) or []
            ci = set(_read(obss[0], "comet_planet_ids", []) or [])
            rf = _read(obss[0], "fleets", []) or []
            sd = snap(rp, ci)

            if sd:
                curr_stats = _game_stats(sd, 0)
                curr_alive = _alive_enemies(sd, rf, 0)
                shaped_r   = _shaped_reward(
                    curr_stats, prev_stats, prev_sd, sd,
                    prev_alive, curr_alive, player=0)
                prev_stats = curr_stats
                prev_sd    = sd
                prev_alive = curr_alive

                # Actor: score edges, sample primary, build moves
                turn_feats, chosen_idx, log_prob_val, p0_moves = \
                    _build_moves(obss[0], actor, device)

                # Critic: estimate V(s) from same edge features
                if turn_feats is not None:
                    with torch.no_grad():
                        critic.eval()
                        tf_t      = torch.from_numpy(turn_feats).unsqueeze(0).to(device)
                        value_val = float(critic(tf_t).item())
                    store_this = True
                else:
                    value_val = 0.0

        # ── Store transition (shaped reward from this step) ────────────────
        if store_this:
            buf.add(turn_feats, chosen_idx, log_prob_val,
                    shaped_r, False, value_val)

        # ── Execute all agents ─────────────────────────────────────────────
        actions = [p0_moves]
        for i in range(1, n):
            try:    a = agent_fns[i](obss[i])
            except: a = []
            actions.append(a or [])

        states = env.step(actions)
        step  += 1

    # ── Terminal win/loss override ─────────────────────────────────────────
    rewards = [float(_read(s, "reward", 0) or 0) for s in states]
    max_r   = max(rewards)
    winner  = -1 if max_r <= 0 else rewards.index(max_r)

    if buf.rewards:
        buf.rewards[-1] += 1.0 if winner == 0 else -0.5
        buf.dones[-1]   = True

    outcome = "WIN" if winner == 0 else ("DRAW" if winner < 0 else "LOSS")
    return buf, outcome


# ════════════════════════════════════════════════════════════════════════════
# PPO UPDATE
# ════════════════════════════════════════════════════════════════════════════

def ppo_update(actor: EdgeTransformerNet,
               critic: TransformerCritic,
               optimizer: optim.Optimizer,
               bufs: list,
               device: str,
               freeze_actor: bool = False):
    """
    PPO update with fixed entropy coefficient (ENT_COEF).

    freeze_actor=True: only the critic is updated (warm-up phase).
      The actor forward still runs (for log-probs), but its gradients
      are not applied.
    """
    if not bufs:
        return 0.0, 0.0

    # ── Collect all transitions from all games ────────────────────────────
    all_tf   = []
    all_ci   = []
    all_oldp = []
    all_ret  = []
    all_adv  = []

    for buf in bufs:
        if len(buf) < 2:
            continue
        rets, advs = buf.compute_gae(last_value=0.0)
        for tf, ci, op, ret, adv in zip(
                buf.turn_feats, buf.chosen_idxs, buf.log_probs,
                rets, advs):
            all_tf.append(tf)
            all_ci.append(ci)
            all_oldp.append(op)
            all_ret.append(ret)
            all_adv.append(adv)

    if not all_tf:
        return 0.0, 0.0

    N = len(all_tf)
    old_lp  = torch.FloatTensor(all_oldp).to(device)
    returns = torch.FloatTensor(all_ret).to(device)
    adv_all = torch.FloatTensor(all_adv).to(device)
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

    total_loss = total_ent = 0.0
    n_updates  = 0

    actor.train(); critic.train()

    for _ in range(PPO_EPOCHS):
        idxs = list(range(N))
        random.shuffle(idxs)

        for start in range(0, N, MINI_BATCH):
            batch = idxs[start:start + MINI_BATCH]

            # ── per-turn forward (variable E — cannot stack) ───────────────
            new_log_ps  = []
            entropies   = []
            values_list = []
            for b in batch:
                tf = torch.from_numpy(all_tf[b]).unsqueeze(0).to(device)   # (1, E, 81)
                al, _ = actor(tf, pad_mask=None)
                logits = al.squeeze(0).squeeze(-1)                          # (E,)
                dist   = torch.distributions.Categorical(logits=logits)
                ci     = torch.tensor(all_ci[b], device=device)
                new_log_ps.append(dist.log_prob(ci))
                entropies.append(dist.entropy())
                values_list.append(critic(tf).squeeze(0))                   # scalar

            new_lp  = torch.stack(new_log_ps)        # (batch,)
            entropy = torch.stack(entropies).mean()   # scalar
            values  = torch.stack(values_list)        # (batch,)

            # ── Loss ───────────────────────────────────────────────────────
            L_val = F.mse_loss(values, returns[batch])
            if freeze_actor:
                loss = VAL_COEF * L_val
            else:
                ratio  = torch.exp(new_lp - old_lp[batch])
                adv_b  = adv_all[batch]
                L_clip = torch.min(
                    ratio * adv_b,
                    torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b
                ).mean()
                loss = -L_clip + VAL_COEF * L_val - ENT_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            grad_params = (list(critic.parameters()) if freeze_actor
                           else list(actor.parameters()) + list(critic.parameters()))
            nn.utils.clip_grad_norm_(grad_params, 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_ent  += entropy.item()
            n_updates  += 1

    n = max(1, n_updates)
    return total_loss / n, total_ent / n


# ════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ════════════════════════════════════════════════════════════════════════════

def train(n_rounds:        int  = N_ROUNDS,
          games_per_round: int  = GAMES_PER_ROUND,
          actor_path:      str  = str(DEFAULT_ACTOR_PATH),
          device:          str  = "auto"):

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load actor ─────────────────────────────────────────────────────────
    if Path(actor_path).exists():
        actor = load_actor(actor_path, device=device)
        print(f"Actor loaded from  {actor_path}")
    else:
        print(f"WARNING: actor checkpoint not found at {actor_path}")
        print("  Run imitation training first:")
        print("  python imitation/model.py --dataset imitation/dataset.npz")
        actor = EdgeTransformerNet().to(device)
        print("  Using random-initialized actor.")

    # ── Load critic ────────────────────────────────────────────────────────
    critic = TransformerCritic().to(device)

    # ── Resume from transformer checkpoint if available ────────────────────
    best_wr        = 0.0
    start_rnd      = 1
    avg_loss       = 0.0
    avg_ent        = 0.0
    is_fresh_start = True   # flipped to False if a checkpoint loads successfully

    for resume_path in [SAVE_PATH, BEST_PATH]:
        if resume_path.exists():
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                actor.load_state_dict(ckpt["actor"])
                critic.load_state_dict(ckpt["critic"])
                best_wr        = ckpt.get("best_wr", 0.0)
                start_rnd      = ckpt.get("epoch", 1) + 1
                is_fresh_start = False
                print(f"Resumed from {resume_path.name}  "
                      f"(round {start_rnd-1}, best_wr={best_wr:.1%})")
                print(f"  Critic: loaded from checkpoint — warm-up skipped.")
                break
            except Exception as e:
                print(f"  Could not resume from {resume_path.name}: {e}")

    if is_fresh_start:
        print(f"  Critic: TransformerCritic — starts fresh.")
        print(f"  Actor will be frozen for first {CRITIC_WARMUP_ROUNDS} rounds (critic warm-up).")

    # ── Optimizers ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": actor.parameters(),  "lr": LR_ACTOR},
        {"params": critic.parameters(), "lr": LR_CRITIC},
    ], weight_decay=1e-4)

    # ── Load enemy agents ──────────────────────────────────────────────────
    _std_paths = glob.glob(str(ENEMY_DIR / "*" / "main.py"))
    _my_paths  = [p for p in glob.glob(str(ENEMY_DIR / "my" / "*.py"))
                  if not Path(p).name.startswith("__")]
    enemies = []
    for ef in sorted(_std_paths + _my_paths):
        p     = Path(ef)
        label = p.stem if p.parent.name == "my" else p.parent.name
        try:
            fn, _ = load_agent(ef)
            enemies.append((label, fn))
            print(f"  Enemy loaded: {label}")
        except Exception as e:
            print(f"  WARN {label}: {e}")

    if not enemies:
        print(f"No enemy agents found in {ENEMY_DIR}"); return None

    n_actor  = sum(p.numel() for p in actor.parameters()  if p.requires_grad)
    n_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"\nActor  params: {n_actor:,}")
    print(f"Critic params: {n_critic:,}")
    print(f"\nPPO (EdgeTransformer): {n_rounds} rounds x {games_per_round} games")
    print(f"  LR actor={LR_ACTOR}  LR critic={LR_CRITIC}")
    print(f"  ENT_COEF={ENT_COEF} (fixed)  CLIP_EPS={CLIP_EPS}")
    print(f"  Critic warm-up: first {CRITIC_WARMUP_ROUNDS} rounds actor is frozen")
    print("=" * 60)

    no_improve = 0

    for rnd in range(start_rnd, start_rnd + n_rounds):
        wins = losses = draws = 0
        all_bufs: list = []

        for g in range(games_per_round):
            # Alternate 1v1 / 4p FFA so RL fine-tunes both modes.
            # 1v1 games: 1 enemy  (is_1v1=1 features → network uses 1v1 weights)
            # 4p games : 3 enemies (is_4v4=1 features → network uses 4p weights)
            # Ratio: every 3rd game is 1v1, rest are 4p FFA.
            if g % 3 == 0:
                # 1v1 game
                filler = random.choice(enemies)
                fns    = [None, filler[1]]
                mode_tag = "1v1"
            else:
                # 4p FFA game
                fillers  = random.choices(enemies, k=min(3, len(enemies)))
                fns      = [None] + [fn for _, fn in fillers]
                mode_tag = "4p"

            seed = 42 + rnd * 1000 + g

            try:
                buf, outcome = play_game_collect(
                    actor, critic, fns, seed=seed, device=device)
                all_bufs.append(buf)
            except Exception as e:
                print(f"\n  game error: {e}")
                import traceback; traceback.print_exc()
                continue

            if   outcome == "WIN":  wins   += 1
            elif outcome == "LOSS": losses += 1
            else:                   draws  += 1

            sys.stdout.write(
                f"\r  Round {rnd}  game {g+1}/{games_per_round}"
                f" [{mode_tag}]  W:{wins} L:{losses} D:{draws}  "
            )
            sys.stdout.flush()

        # ── PPO update on all games from this round ────────────────────────
        # Warm-up only on a fresh start (no prior checkpoint).
        # When resuming, the critic is already calibrated — skip immediately.
        freeze_actor = is_fresh_start and ((rnd - 1) < CRITIC_WARMUP_ROUNDS)
        avg_loss, avg_ent = ppo_update(
            actor, critic, optimizer,
            all_bufs, device,
            freeze_actor=freeze_actor)
        for buf in all_bufs:
            buf.clear()

        wr = wins / max(1, games_per_round)
        tag = ""

        ckpt = {
            "epoch":   rnd,
            "best_wr": max(wr, best_wr),
            "loss":    avg_loss,
            "entropy": avg_ent,
            "actor":   actor.state_dict(),
            "critic":  critic.state_dict(),
        }
        torch.save(ckpt, SAVE_PATH)

        if wr > best_wr:
            best_wr    = wr
            no_improve = 0
            torch.save(ckpt, BEST_PATH)
            tag = f"  *** best ({wr:.1%}) -> transformer_best.pt"
        else:
            no_improve += 1
            tag = f"  (no improve, best={best_wr:.1%}, streak={no_improve})"

        warmup_str = "  [critic warmup — actor frozen]" if freeze_actor else ""
        print(f"\n  -> WR={wr:.1%}  loss={avg_loss:.4f}  "
              f"ent={avg_ent:.4f}{warmup_str}{tag}")

    print(f"\nDone. Best WR={best_wr:.1%}")
    return actor, critic


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="PPO fine-tuning of EdgeTransformerNet")
    p.add_argument("--rounds",   type=int,  default=N_ROUNDS)
    p.add_argument("--games",    type=int,  default=GAMES_PER_ROUND)
    p.add_argument("--actor",    type=str,  default=str(DEFAULT_ACTOR_PATH))
    p.add_argument("--device",   type=str,  default="auto")
    args = p.parse_args()

    train(
        n_rounds=args.rounds,
        games_per_round=args.games,
        actor_path=args.actor,
        device=args.device,
    )
