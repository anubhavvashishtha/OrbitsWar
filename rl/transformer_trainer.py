"""
rl/transformer_trainer.py
=========================
PPO fine-tuning for EdgeTransformerNet (actor) + TransformerCritic (critic).

Unlike self_play_trainer.py (which picks one of 6 abstract intents via a
small MLP), this trainer lets the EdgeTransformerNet directly score every
(src, tgt) edge using multi-Bernoulli PPO credit assignment.

Actor  : EdgeTransformerNet   (imitation/model.py)
         Input : (E, 96) edge features for the current turn
         Output: per-edge sigmoid scores → execute top-K above ACTION_THRESHOLD
         Pretrained from: imitation/checkpoints/best_model.pt

Critic : TransformerCritic     (lightweight transformer on edge features)
         Input : (E, 96) edge feature matrix — same as actor
         Output: scalar V(s)  (no Sigmoid — unbounded for proper GAE)

PPO credit assignment  (multi-Bernoulli)
----------------------------------------
  Every edge that fires above ACTION_THRESHOLD contributes equally to the
  turn log_prob: mean( log(sigmoid(logit_i)) ) over all executed edges.
  Edges that do not fire are dropped — no gradient from them.
  Training and submission execute the SAME set of edges — no mismatch.

Usage
-----
    python rl/transformer_trainer.py
    python rl/transformer_trainer.py --rounds 50 --games 8
    python rl/transformer_trainer.py --actor  imitation/checkpoints/best_model.pt
    python rl/transformer_trainer.py --critic rl/last.pt
"""

import sys, glob, os, random, argparse, math, json
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

# Cache build_world NOW before any enemy agents are loaded into sys.path.
# Enemy main.py files don't have build_world — lazy imports inside functions
# would find the wrong main.py after load_agent() adds enemy dirs to sys.path.
from main import build_world as _build_world

# ── imports from RL helpers ───────────────────────────────────────────────
from self_play_trainer import (
    load_agent, snap, _read,
    _event_reward, _game_stats, _alive_enemies,
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
HISTORY_PATH        = ROOT / "transformer_history.json"
CURVES_PATH         = ROOT / "transformer_curves.png"

# ── hyper-parameters ──────────────────────────────────────────────────────
N_ROUNDS        = 30      # ← set to 30 for test run, increase after checking results
GAMES_PER_ROUND = 32
LR_ACTOR        = 1e-5    # tighter than before — prevents policy collapse
LR_CRITIC       = 1e-4
CLIP_EPS        = 0.1     # tighter clipping — less aggressive policy change
VAL_COEF        = 0.5
VALUE_CLIP      = 0.1
PPO_EPOCHS      = 2       # fewer updates per batch — reduces overfitting to a round
MINI_BATCH      = 64
ACTION_THRESHOLD = 0.35
TOP_K_EDGES      = 5

ENT_COEF_START = 0.01
ENT_COEF_END   = 0.002
ENT_COEF_RAMP  = 50

# Set True to reset entropy ramp at session start (good for tier changes).
# Set False to continue entropy decay from where the last session left off
# (good for same-tier resumptions).
RESET_ENT_ON_RESUME = True

TARGET_KL       = 0.02    # stop early if policy diverges — prevents collapse
KL_STOP_MULT    = 1.5

TERMINAL_WIN      = 1.0
TERMINAL_LOSS     = -1.5
TERMINAL_DRAW     = -0.5

FORCE_1V1 = True

# ── Critic warm-up ─────────────────────────────────────────────────────────
CRITIC_WARMUP_ROUNDS = 20

# ── Curriculum ─────────────────────────────────────────────────────────────
# Manual progression — change START_TIER before each session:
#   0 → Tier 1 only        — Session 1
#   1 → Tier 1 + Tier 2    — Session 2
#   2 → All tiers          — Session 3+
#
# Separate tiers for 1v1 and 4v4 — agents perform very differently in each mode.
# Source: internal round-robin tournament (tournament_results.json).

CURRICULUM_TIERS_1V1 = [
    # Tier 1 — easy  (0–28% 1v1 WR)
    ['pascal-orbitwork-v14', 'tamrazov-starwars',
     'yuriygreben-architect', 'sigmaborov-starter', 'main_4'],
    # Tier 2 — medium  (42–57% 1v1 WR)
    ['main_5', 'main_2', 'main_3',
     'sigmaborov-reinforce', 'pilkwang-structured'],
    # Tier 3 — hard  (71–85% 1v1 WR)
    ['regressor', 'ykhnkf-distance-prioritized',
     'classifier_bot', 'main_1', 'main'],
]

CURRICULUM_TIERS_4V4 = [
    # Tier 1 — easy  (0 wins, worst avg rank in 4v4)
    ['main_5', 'pascal-orbitwork-v14', 'tamrazov-starwars',
     'sigmaborov-reinforce', 'main'],
    # Tier 2 — medium  (0–1 wins, mixed avg rank)
    ['yuriygreben-architect', 'sigmaborov-starter',
     'regressor', 'main_4', 'main_2'],
    # Tier 3 — hard  (1–3 wins, best avg rank)
    ['pilkwang-structured', 'classifier_bot',
     'main_3', 'ykhnkf-distance-prioritized', 'main_1'],
]

START_TIER = 1   # ← CHANGE THIS each session (0, 1, or 2)

# ── Self-play ───────────────────────────────────────────────────────────────
# Disabled by default — enable for local 24h runs after curriculum is complete.
# Every SELFPLAY_SNAPSHOT_EVERY rounds, a frozen copy of the current actor is
# added to the enemy pool.  The agent trains against its own improving past self
# → no fixed-pool ceiling, keeps learning indefinitely.
SELFPLAY_ENABLED        = False  # ← set True for local 24h runs
SELFPLAY_SNAPSHOT_EVERY = 15    # save a snapshot every N rounds
SELFPLAY_MAX_SNAPSHOTS  = 10    # keep only the N most recent in pool

# ── 4v4 placement rewards ───────────────────────────────────────────────────
# Hardcoded independently of 1v1 terminal values.
# 2nd place is treated as a partial win (decent in 4-player FFA).
# 3rd/4th scale the penalty by how badly the agent finished.
PLACEMENT_REWARD = {
    1:  1.0,   # 1st — same signal as a 1v1 win
    2:  0.3,   # 2nd — small positive, not punished
    3: -0.5,   # 3rd — moderate negative
    4: -1.0,   # 4th — same signal as a 1v1 loss
}


def _ent_coef(rnd: int, start_rnd: int) -> float:
    """Linear decay from ENT_COEF_START → ENT_COEF_END over ENT_COEF_RAMP rounds."""
    t = min(1.0, max(0.0, (rnd - start_rnd) / max(1, ENT_COEF_RAMP)))
    return ENT_COEF_START + (ENT_COEF_END - ENT_COEF_START) * t


def create_snapshot_agent(actor: EdgeTransformerNet, device: str):
    """
    Freeze a copy of the current actor and return it as an enemy agent function.
    The returned callable takes an obs dict and returns a list of moves —
    same interface as any enemy in play_game_collect.
    """
    import copy
    snapshot = EdgeTransformerNet().to(device)
    snapshot.load_state_dict(copy.deepcopy(actor.state_dict()))
    snapshot.eval()

    def agent_fn(obs):
        try:
            candidates = build_all_candidate_edges(obs)
            if not candidates:
                return []
            edge_list   = [(src, tgt) for src, tgt, _ in candidates]
            feat_matrix = np.array([f for _, _, f in candidates], dtype=np.float32)
            X = torch.from_numpy(feat_matrix).unsqueeze(0).to(device)
            with torch.no_grad():
                al, fl   = snapshot(X, pad_mask=None)
                act_prob = torch.sigmoid(al.squeeze(0).squeeze(-1)).cpu().numpy()
                fraction = torch.sigmoid(fl.squeeze(0).squeeze(-1)).cpu().numpy()
            world      = _build_world(obs)
            planet_map = {p.id: p for p in world.planets}
            spent: dict = {}
            moves = []
            committed = 0
            for i in sorted(range(len(edge_list)), key=lambda i: -act_prob[i]):
                if committed >= TOP_K_EDGES or act_prob[i] < ACTION_THRESHOLD:
                    break
                src_id, tgt_id = edge_list[i]
                src = planet_map.get(src_id)
                if src is None or src.owner != world.player:
                    continue
                available = int(src.ships) - spent.get(src_id, 0)
                if available <= 0 or fraction[i] < 0.08:
                    continue
                send = max(1, int(available * min(0.90, float(fraction[i]))))
                shot = world.plan_shot(src_id, tgt_id, send)
                if shot is None:
                    continue
                spent[src_id] = spent.get(src_id, 0) + send
                moves.append([src_id, float(shot[0]), send])
                committed += 1
            return moves
        except Exception:
            return []

    return agent_fn


def _save_curves(history: list, path: Path):
    """Save a 4-panel training curve PNG from the history list."""
    if len(history) < 2:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive — safe inside training loop
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rounds   = [h['round']        for h in history]
    wr       = [h['wr']           for h in history]
    loss     = [h['loss']         for h in history]
    ent      = [h['entropy']      for h in history]
    kl       = [h['kl']           for h in history]
    clip     = [h['clipfrac']     for h in history]
    ec       = [h['ent_coef']     for h in history]
    ev       = [h['events']       for h in history]
    mom      = [h['momentum']     for h in history]
    term     = [h['terminal']     for h in history]
    total    = [h['total_reward'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('PPO Transformer — Training Curves', fontsize=13)

    # ── Panel 1: Win rate ─────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(rounds, wr, 'b-', linewidth=1.5, label='win rate')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='50 % baseline')
    # shade critic warmup region
    warmup_end = history[0]['round'] + CRITIC_WARMUP_ROUNDS
    ax.axvspan(rounds[0], min(warmup_end, rounds[-1]),
               alpha=0.08, color='orange', label='critic warmup')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Round'); ax.set_ylabel('Win rate')
    ax.set_title('Win Rate')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2: PPO loss / entropy / KL ─────────────────────────────────
    ax  = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(rounds, loss, 'r-',  linewidth=1,   label='PPO loss')
    ax2.plot(rounds, ent, 'g--', linewidth=1,   label='entropy')
    ax2.plot(rounds, kl,  'm:',  linewidth=1,   label='KL')
    ax.set_xlabel('Round'); ax.set_ylabel('PPO loss', color='r')
    ax2.set_ylabel('Entropy / KL')
    ax.set_title('Loss / Entropy / KL')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 3: Reward decomposition ────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(rounds, ev,   'g-',  linewidth=1,   label='events')
    ax.plot(rounds, mom,  'b-',  linewidth=1,   label='momentum')
    ax.plot(rounds, term, 'r-',  linewidth=1,   label='terminal')
    ax.plot(rounds, total,'k-',  linewidth=2,   label='total')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Round'); ax.set_ylabel('Reward (summed over games)')
    ax.set_title('Reward Decomposition')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 4: ENT_COEF schedule + clip fraction ────────────────────────
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    ax.plot(rounds, ec,   color='purple',  linewidth=1.5, label='ent_coef')
    ax2.plot(rounds, clip, color='orange', linewidth=1,
             linestyle='--', label='clipfrac')
    ax.set_xlabel('Round'); ax.set_ylabel('ent_coef', color='purple')
    ax2.set_ylabel('clip fraction', color='orange')
    ax.set_title('Entropy Coef / Clip Fraction')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# CRITIC  — Transformer on the same (E, 96) edge features as the actor
# ════════════════════════════════════════════════════════════════════════════

class TransformerCritic(nn.Module):
    """
    Lightweight transformer critic that consumes the same (E, 96) edge
    feature matrix the actor already builds each turn.

    Architecture  (d_model=64, 2 encoder layers):
      4 component projectors  31/24/24/17 → 64 each
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
        self.src_proj    = nn.Sequential(nn.Linear(24, D), nn.LayerNorm(D), nn.GELU())
        self.tgt_proj    = nn.Sequential(nn.Linear(24, D), nn.LayerNorm(D), nn.GELU())
        self.edge_proj   = nn.Sequential(nn.Linear(17, D), nn.LayerNorm(D), nn.GELU())
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
        x        : (B, E, 96)  or (1, E, 96) for a single turn
        pad_mask : (B, E) bool — True = padding, excluded from mean-pool
        Returns  : (B,) scalar values
        """
        g = self.global_proj(x[..., :31])
        s = self.src_proj(x[..., 31:55])
        t = self.tgt_proj(x[..., 55:79])
        e = self.edge_proj(x[..., 79:96])
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

    Each transition stores the FULL turn edge-feature matrix (E, 96)
    so the actor can be re-run with gradients during the PPO update.
    This is memory-heavier than storing a single 30-dim state vector,
    but it allows the transformer to re-score with the updated weights.

    Action semantics
    ----------------
    executed_idxs : indices of edges that fired this turn (1–TOP_K_EDGES).
    log_prob      : mean log(sigmoid(logit_i)) over executed edges at collection time.
    """

    def __init__(self, gamma: float = GAMMA, lam: float = GAE_LAMBDA):
        self.gamma = gamma
        self.lam   = lam
        self.clear()

    def clear(self):
        self.turn_feats    = []   # (E_i, 96) numpy arrays — shared by actor & critic
        self.executed_idxs = []   # list[int] — indices of edges that fired this turn
        self.log_probs     = []   # float — mean logsigmoid over executed edges
        self.rewards       = []   # float
        self.dones         = []   # float  0/1
        self.values        = []   # float  V(s)
        self.total_shaped   = 0.0
        self.total_events   = 0.0   # planet captures / losses / eliminations
        self.total_momentum = 0.0   # continuous share-delta term
        self.terminal_bonus = 0.0

    def add(self, turn_feats: np.ndarray, executed_idxs: list,
            log_prob: float, reward: float, done: bool, value: float):
        self.turn_feats.append(turn_feats.astype(np.float32))
        self.executed_idxs.append(list(executed_idxs))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))
        self.total_shaped += float(reward)

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
      feat_matrix    : (E, 96) numpy
      executed_idxs  : list[int] — indices of edges that fired (1–TOP_K_EDGES)
      log_prob       : float — mean log(sigmoid(logit)) over executed edges
      moves          : list of [src, angle, ships]
    Returns (feat_matrix, [], None, []) if no edges fire above threshold.
    Returns (None, [], None, []) if no candidates at all.

    Multi-Bernoulli credit: every edge that fires contributes equally to the
    log_prob used for the PPO ratio.  Edges that do not fire are dropped —
    they produce no gradient signal.
    """
    candidates = build_all_candidate_edges(obs)
    if not candidates:
        return None, [], None, []

    edge_list   = [(src, tgt) for src, tgt, _ in candidates]
    feat_matrix = np.array([f for _, _, f in candidates], dtype=np.float32)

    X = torch.from_numpy(feat_matrix).unsqueeze(0).to(device)  # (1, E, 96)

    with torch.no_grad():
        actor.eval()
        al, fl   = actor(X, pad_mask=None)
        logits   = al.squeeze(0).squeeze(-1)                     # (E,)
        act_prob = torch.sigmoid(logits).cpu().numpy()           # (E,)
        fraction = torch.sigmoid(fl.squeeze(0).squeeze(-1)).cpu().numpy()

    # Execute top-K edges strictly above threshold — same as submission agent
    executed_indices = []
    try:
        world      = _build_world(obs)
        planet_map = {p.id: p for p in world.planets}
        spent      = defaultdict(int)
        moves      = []
        committed  = 0

        order = sorted(range(len(edge_list)), key=lambda i: -act_prob[i])
        for i in order:
            if committed >= TOP_K_EDGES:
                break
            if act_prob[i] < ACTION_THRESHOLD:
                break   # list is sorted descending — nothing below this is worth taking
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
            spent[src_id] += send
            moves.append([src_id, float(shot[0]), send])
            executed_indices.append(i)
            committed += 1
    except Exception:
        moves            = []
        executed_indices = []

    if not executed_indices:
        return feat_matrix, [], None, moves

    # Multi-Bernoulli log_prob: mean over executed edges so scale is
    # consistent whether 1 or 5 edges fire this turn.
    exec_t   = torch.tensor(executed_indices, dtype=torch.long)
    log_prob = float(F.logsigmoid(logits[exec_t]).mean().item())

    return feat_matrix, executed_indices, log_prob, moves


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
        turn_feats = exec_idxs = log_prob_val = value_val = None
        shaped_r = 0.0

        if obss[0] is not None:
            rp = _read(obss[0], "planets", []) or []
            ci = set(_read(obss[0], "comet_planet_ids", []) or [])
            rf = _read(obss[0], "fleets", []) or []
            sd = snap(rp, ci)

            if sd:
                curr_stats = _game_stats(sd, 0)
                curr_alive = _alive_enemies(sd, rf, 0)

                if prev_stats is not None and prev_sd is not None:
                    event_r    = _event_reward(prev_sd, sd, prev_alive, curr_alive, player=0)
                    my_s, my_pr, my_pl, tot_s, tot_pr, tot_pl = curr_stats
                    ps,   ppr,   ppl,   _,     _,      _ = prev_stats
                    momentum_r = MOMENTUM_SCALE * (
                        0.4 * (my_s  - ps)  / max(1.0, tot_s)
                        + 0.4 * (my_pr - ppr) / max(1.0, tot_pr)
                        + 0.2 * (my_pl - ppl) / max(1.0, tot_pl)
                    )
                else:
                    event_r = momentum_r = 0.0

                shaped_r = event_r + momentum_r
                buf.total_events   += event_r
                buf.total_momentum += momentum_r

                prev_stats = curr_stats
                prev_sd    = sd
                prev_alive = curr_alive

                # Actor: score edges, execute top-K above threshold, build moves
                turn_feats, exec_idxs, log_prob_val, p0_moves = \
                    _build_moves(obss[0], actor, device)

                # Only store turns where at least one edge actually fired
                if turn_feats is not None and exec_idxs and log_prob_val is not None:
                    with torch.no_grad():
                        critic.eval()
                        tf_t      = torch.from_numpy(turn_feats).unsqueeze(0).to(device)
                        value_val = float(critic(tf_t).item())
                    store_this = True
                else:
                    value_val = 0.0

        # ── Store transition (shaped reward from this step) ────────────────
        if store_this:
            buf.add(turn_feats, exec_idxs, log_prob_val,
                    shaped_r, False, value_val)

        # ── Execute all agents ─────────────────────────────────────────────
        actions = [p0_moves]
        for i in range(1, n):
            try:    a = agent_fns[i](obss[i])
            except: a = []
            actions.append(a or [])

        states = env.step(actions)
        step  += 1

    # ── Terminal reward override ───────────────────────────────────────────
    rewards = [float(_read(s, "reward", 0) or 0) for s in states]
    max_r   = max(rewards)
    winner  = -1 if max_r <= 0 else rewards.index(max_r)

    if buf.rewards:
        if n == 2:
            # 1v1: simple win/loss/draw
            if winner == 0:
                term = TERMINAL_WIN
            elif winner < 0:
                term = TERMINAL_DRAW
            else:
                term = TERMINAL_LOSS
            outcome = "WIN" if winner == 0 else ("DRAW" if winner < 0 else "LOSS")
        else:
            # 4v4: placement-based reward (2nd ≠ 4th)
            if max_r <= 0:
                term    = TERMINAL_DRAW
                outcome = "DRAW"
            else:
                sorted_players = sorted(range(len(rewards)), key=lambda i: -rewards[i])
                our_placement  = sorted_players.index(0) + 1   # 1-indexed rank
                term    = PLACEMENT_REWARD.get(our_placement, TERMINAL_LOSS)
                outcome = ("WIN" if our_placement == 1
                           else "DRAW" if our_placement == 2
                           else "LOSS")

        buf.terminal_bonus = term
        buf.rewards[-1]    = term   # override last step — game over
        buf.dones[-1]      = True

        # Wins keep events/momentum to reinforce which specific actions won.
        # Losses and draws are zeroed — only the terminal signal matters.
        # Draws are penalised (-0.5) to push agent beyond draw strategy toward wins.
        if outcome in ("LOSS", "DRAW"):
            for i in range(len(buf.rewards) - 1):
                buf.rewards[i] = 0.0
    else:
        outcome = "DRAW"

    return buf, outcome


# ════════════════════════════════════════════════════════════════════════════
# PPO UPDATE
# ════════════════════════════════════════════════════════════════════════════

def ppo_update(actor: EdgeTransformerNet,
               critic: TransformerCritic,
               optimizer: optim.Optimizer,
               bufs: list,
               device: str,
               freeze_actor: bool = False,
               ent_coef: float = ENT_COEF_END):
    """
    PPO update with scheduled entropy coefficient (ent_coef).

    freeze_actor=True: only the critic is updated (warm-up phase).
      The actor forward still runs (for log-probs), but its gradients
      are not applied.
    """
    _empty = {"loss": 0.0, "entropy": 0.0, "kl": 0.0, "clipfrac": 0.0, "vloss": 0.0}
    if not bufs:
        return _empty

    all_tf, all_exec, all_oldp, all_oldv = [], [], [], []
    all_ret, all_adv = [], []

    for buf in bufs:
        if len(buf) < 2:
            continue
        rets, advs = buf.compute_gae(last_value=0.0)
        for tf, ei, op, ov, ret, adv in zip(
                buf.turn_feats, buf.executed_idxs, buf.log_probs, buf.values,
                rets, advs):
            all_tf.append(tf)
            all_exec.append(ei)
            all_oldp.append(op)
            all_oldv.append(ov)
            all_ret.append(ret)
            all_adv.append(adv)

    if not all_tf:
        return _empty

    N = len(all_tf)
    old_lp = torch.FloatTensor(all_oldp).to(device)

    old_v = torch.FloatTensor(all_oldv).to(device)
    returns = torch.FloatTensor(all_ret).to(device)
    adv_all = torch.FloatTensor(all_adv).to(device)
    adv_all = (adv_all - adv_all.mean()) / (adv_all.std() + 1e-8)

    total_loss = total_ent = total_kl = total_clip = total_vloss = 0.0
    n_updates = 0

    actor.train()
    critic.train()

    for _ in range(PPO_EPOCHS):
        idxs = list(range(N))
        random.shuffle(idxs)
        stop_epoch = False

        for start in range(0, N, MINI_BATCH):
            batch = idxs[start:start + MINI_BATCH]

            new_log_ps, entropies, values_list = [], [], []
            for b in batch:
                tf = torch.from_numpy(all_tf[b]).unsqueeze(0).to(device)
                al, _ = actor(tf, pad_mask=None)
                logits = al.squeeze(0).squeeze(-1)   # (E,)

                # Multi-Bernoulli: mean log(sigmoid(logit)) over edges that fired
                exec_idxs = torch.tensor(all_exec[b], dtype=torch.long, device=device)
                new_log_ps.append(F.logsigmoid(logits[exec_idxs]).mean())

                # Bernoulli entropy over ALL candidate edges — reflects full policy uncertainty
                probs = torch.sigmoid(logits)
                ent   = -(probs * (probs + 1e-8).log()
                          + (1 - probs) * (1 - probs + 1e-8).log()).mean()
                entropies.append(ent)

                values_list.append(critic(tf).squeeze(0))

            new_lp = torch.stack(new_log_ps)
            entropy = torch.stack(entropies).mean()
            values = torch.stack(values_list)

            v_old = old_v[batch]
            v_clipped = v_old + torch.clamp(values - v_old, -VALUE_CLIP, VALUE_CLIP)
            L_val = torch.max(
                (values - returns[batch]).pow(2),
                (v_clipped - returns[batch]).pow(2),
            ).mean()

            if freeze_actor:
                loss = VAL_COEF * L_val
                ratio = torch.ones_like(new_lp)
            else:
                ratio = torch.exp(new_lp - old_lp[batch])
                adv_b = adv_all[batch]
                L_clip = torch.min(
                    ratio * adv_b,
                    torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_b,
                ).mean()
                loss = -L_clip + VAL_COEF * L_val - ent_coef * entropy

            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio.clamp(min=1e-8))).mean()
                clipfrac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean()

            optimizer.zero_grad()
            loss.backward()
            grad_params = (
                list(critic.parameters()) if freeze_actor
                else list(actor.parameters()) + list(critic.parameters())
            )
            nn.utils.clip_grad_norm_(grad_params, 0.5)
            optimizer.step()

            total_loss += loss.item()
            total_ent += entropy.item()
            total_kl += approx_kl.item()
            total_clip += clipfrac.item()
            total_vloss += L_val.item()
            n_updates += 1

            if not freeze_actor and approx_kl.item() > TARGET_KL * KL_STOP_MULT:
                stop_epoch = True
                break

        if stop_epoch:
            break

    n = max(1, n_updates)
    return {
        "loss": total_loss / n,
        "entropy": total_ent / n,
        "kl": total_kl / n,
        "clipfrac": total_clip / n,
        "vloss": total_vloss / n,
    }


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
    best_wr_ppo    = 0.0
    start_rnd      = 1
    ent_ramp_start = 1
    avg_loss       = 0.0
    avg_ent        = 0.0
    is_fresh_start = True   # flipped to False if a checkpoint loads successfully

    for resume_path in [SAVE_PATH, BEST_PATH]:
        if resume_path.exists():
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
                actor.load_state_dict(ckpt["actor"])
                critic.load_state_dict(ckpt["critic"])
                best_wr_ppo    = ckpt.get("best_wr_ppo", ckpt.get("best_wr", 0.0))
                if ckpt.get("epoch", 0) <= CRITIC_WARMUP_ROUNDS:
                    best_wr_ppo = 0.0
                start_rnd      = ckpt.get("epoch", 1) + 1
                if RESET_ENT_ON_RESUME:
                    ent_ramp_start = start_rnd   # fresh decay from this session
                else:
                    ent_ramp_start = ckpt.get("ent_ramp_start", start_rnd)
                # One-time manual override for old checkpoints missing ent_ramp_start.
                # Set _ENT_RAMP_START_OVERRIDE in the notebook before calling train().
                override = getattr(
                    __import__('rl.transformer_trainer', fromlist=['']),
                    '_ENT_RAMP_START_OVERRIDE', None)
                if override is not None:
                    ent_ramp_start = override
                is_fresh_start = False
                print(f"Resumed from {resume_path.name}  "
                      f"(round {start_rnd-1}, best_wr_ppo={best_wr_ppo:.1%})")
                print(f"  Critic: loaded from checkpoint — warm-up skipped.")
                break
            except Exception as e:
                print(f"  Could not resume from {resume_path.name}: {e}")

    if is_fresh_start:
        print(f"  Critic: TransformerCritic — starts fresh.")
        print(f"  Actor will be frozen for first {CRITIC_WARMUP_ROUNDS} rounds (critic warm-up).")

    # ── Load training history (keep rounds before start_rnd on resume) ─────
    history = []
    if HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text())
            history = [h for h in history if h.get('round', 0) < start_rnd]
        except Exception:
            history = []

    # ── Optimizers ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW([
        {"params": actor.parameters(),  "lr": LR_ACTOR},
        {"params": critic.parameters(), "lr": LR_CRITIC},
    ], weight_decay=1e-4)

    # ── Load enemy agents ──────────────────────────────────────────────────
    _std_paths = glob.glob(str(ENEMY_DIR / "*" / "main.py"))
    _my_paths  = [p for p in glob.glob(str(ENEMY_DIR / "my" / "*.py"))
                  if not Path(p).name.startswith("__")]
    all_enemies: dict = {}   # label → fn
    for ef in sorted(_std_paths + _my_paths):
        p     = Path(ef)
        label = p.stem if p.parent.name == "my" else p.parent.name
        try:
            fn, _ = load_agent(ef)
            all_enemies[label] = fn
            print(f"  Enemy loaded: {label}")
        except Exception as e:
            print(f"  WARN {label}: {e}")

    if not all_enemies:
        print(f"No enemy agents found in {ENEMY_DIR}"); return None

    # ── Curriculum pool setup ──────────────────────────────────────────────
    def _build_pools(tier_labels_list, label):
        """Build per-tier (label, fn) lists and active tier pools for one mode."""
        tier_agents = []
        assigned = set()
        for tier_labels in tier_labels_list:
            tier = [(lbl, all_enemies[lbl]) for lbl in tier_labels if lbl in all_enemies]
            tier_agents.append(tier)
            assigned.update(lbl for lbl in tier_labels if lbl in all_enemies)
        leftover = [(lbl, fn) for lbl, fn in all_enemies.items() if lbl not in assigned]
        if leftover:
            tier_agents[-1].extend(leftover)
            print(f"  [{label}] {len(leftover)} untiered enemies added to Tier {len(tier_labels_list)}")
        active = [tier_agents[t] for t in range(min(START_TIER + 1, len(tier_agents))) if tier_agents[t]]
        flat   = [e for pool in active for e in pool]
        return tier_agents, active, flat

    tier_agents_1v1, active_tier_pools_1v1, active_enemies_1v1 = _build_pools(CURRICULUM_TIERS_1V1, "1v1")
    tier_agents_4v4, active_tier_pools_4v4, active_enemies_4v4 = _build_pools(CURRICULUM_TIERS_4V4, "4v4")

    print("\n  1v1 curriculum:")
    for t, tier in enumerate(tier_agents_1v1):
        status = "ACTIVE" if t <= START_TIER else "locked"
        print(f"    Tier {t+1} [{status}]: {[lbl for lbl, _ in tier]}")
    print(f"    Active: {len(active_enemies_1v1)} enemies across {len(active_tier_pools_1v1)} tier(s)")

    print(f"  4v4 curriculum: {'DISABLED (FORCE_1V1=True)' if FORCE_1V1 else 'enabled'}")
    for t, tier in enumerate(tier_agents_4v4):
        status = ("ready" if t <= START_TIER else "locked") if not FORCE_1V1 else "idle"
        print(f"    Tier {t+1} [{status}]: {[lbl for lbl, _ in tier]}")
    print(f"    Pool built: {len(active_enemies_4v4)} enemies — not used while FORCE_1V1=True")

    n_actor  = sum(p.numel() for p in actor.parameters()  if p.requires_grad)
    n_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"\nActor  params: {n_actor:,}")
    print(f"Critic params: {n_critic:,}")
    print(f"\nPPO (EdgeTransformer): {n_rounds} rounds x {games_per_round} games")
    print(f"  LR actor={LR_ACTOR}  LR critic={LR_CRITIC}")
    print(f"  ENT_COEF {ENT_COEF_START:.3f}→{ENT_COEF_END:.3f} over {ENT_COEF_RAMP} rounds  CLIP_EPS={CLIP_EPS}")
    print(f"  Critic warm-up: first {CRITIC_WARMUP_ROUNDS} rounds actor is frozen")
    print(f"  FORCE_1V1={FORCE_1V1}  Terminal win/loss={TERMINAL_WIN}/{TERMINAL_LOSS}")
    print("=" * 60)

    no_improve = 0

    for rnd in range(start_rnd, start_rnd + n_rounds):
        wins = losses = draws = 0
        all_bufs: list = []

        g_1v1 = g_4v4 = 0   # separate counters for balanced tier interleaving per mode

        for g in range(games_per_round):
            if FORCE_1V1:
                mode_tag = "1v1"
                pool = active_tier_pools_1v1[g_1v1 % len(active_tier_pools_1v1)] if active_tier_pools_1v1 else active_enemies_1v1
                g_1v1 += 1
                filler = random.choice(pool)
                fns    = [None, filler[1]]
            elif g % 2 == 0:
                mode_tag = "1v1"
                pool = active_tier_pools_1v1[g_1v1 % len(active_tier_pools_1v1)] if active_tier_pools_1v1 else active_enemies_1v1
                g_1v1 += 1
                filler = random.choice(pool)
                fns    = [None, filler[1]]
            else:
                mode_tag = "4p"
                pool = active_tier_pools_4v4[g_4v4 % len(active_tier_pools_4v4)] if active_tier_pools_4v4 else active_enemies_4v4
                g_4v4 += 1
                fillers = random.choices(pool, k=min(3, len(pool)))
                fns     = [None] + [fn for _, fn in fillers]

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
        freeze_actor  = is_fresh_start and ((rnd - 1) < CRITIC_WARMUP_ROUNDS)
        ent_coef_val  = _ent_coef(rnd, ent_ramp_start)
        stats = ppo_update(
            actor, critic, optimizer,
            all_bufs, device,
            freeze_actor=freeze_actor,
            ent_coef=ent_coef_val)
        if isinstance(stats, dict):
            avg_loss = stats["loss"]
            avg_ent = stats["entropy"]
            avg_kl = stats["kl"]
        else:
            avg_loss, avg_ent = stats
            avg_kl = 0.0
            stats = {"clipfrac": 0.0}
        wr = wins / max(1, games_per_round)
        tag = ""
        event_sum  = sum(getattr(b, "total_events",   0.0) for b in all_bufs)
        mom_sum    = sum(getattr(b, "total_momentum", 0.0) for b in all_bufs)
        term_sum   = sum(getattr(b, "terminal_bonus", 0.0) for b in all_bufs)
        shaped_sum = event_sum + mom_sum
        clipfrac   = stats.get('clipfrac', 0.0)

        # ── Record history and save curves ────────────────────────────────
        history.append({
            'round':        rnd,
            'wr':           wr,
            'wins':         wins,
            'losses':       losses,
            'draws':        draws,
            'loss':         avg_loss,
            'entropy':      avg_ent,
            'kl':           avg_kl,
            'clipfrac':     clipfrac,
            'ent_coef':     ent_coef_val,
            'events':       event_sum,
            'momentum':     mom_sum,
            'terminal':     term_sum,
            'total_reward': shaped_sum + term_sum,
            'freeze_actor': freeze_actor,
        })
        try:
            HISTORY_PATH.write_text(json.dumps(history, indent=2))
            _save_curves(history, CURVES_PATH)
        except Exception as _e:
            print(f"  [warn] could not save history/curves: {_e}")

        for buf in all_bufs:
            buf.clear()

        ckpt = {
            "epoch": rnd,
            "ent_ramp_start": ent_ramp_start,
            "best_wr": best_wr_ppo,
            "best_wr_ppo": best_wr_ppo,
            "loss": avg_loss,
            "entropy": avg_ent,
            "kl": avg_kl,
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
        }
        torch.save(ckpt, SAVE_PATH)

        # ── Self-play snapshot ────────────────────────────────────────────────
        if SELFPLAY_ENABLED and rnd % SELFPLAY_SNAPSHOT_EVERY == 0:
            snap_fn    = create_snapshot_agent(actor, device)
            snap_label = f'self_r{rnd}'
            # Add snapshot to both mode pools (last active tier in each)
            active_tier_pools_1v1[-1].append((snap_label, snap_fn))
            active_tier_pools_4v4[-1].append((snap_label, snap_fn))
            active_enemies_1v1.append((snap_label, snap_fn))
            active_enemies_4v4.append((snap_label, snap_fn))
            # Drop oldest snapshot if over limit
            self_snaps_1v1 = [(l, f) for l, f in active_enemies_1v1 if l.startswith('self_')]
            if len(self_snaps_1v1) > SELFPLAY_MAX_SNAPSHOTS:
                oldest = self_snaps_1v1[0][0]
                active_tier_pools_1v1[-1] = [(l, f) for l, f in active_tier_pools_1v1[-1] if l != oldest]
                active_tier_pools_4v4[-1] = [(l, f) for l, f in active_tier_pools_4v4[-1] if l != oldest]
                active_enemies_1v1[:] = [(l, f) for l, f in active_enemies_1v1 if l != oldest]
                active_enemies_4v4[:] = [(l, f) for l, f in active_enemies_4v4 if l != oldest]
                print(f"\n  Self-play: dropped {oldest}, kept {SELFPLAY_MAX_SNAPSHOTS} snapshots")
            n_snaps = len([l for l, _ in active_enemies_1v1 if l.startswith('self_')])
            print(f"\n  Self-play snapshot saved: {snap_label}  "
                  f"({n_snaps} snapshots in pool)")

        if not freeze_actor and wr > best_wr_ppo:
            best_wr_ppo = wr
            no_improve = 0
            ckpt["best_wr"] = best_wr_ppo
            ckpt["best_wr_ppo"] = best_wr_ppo
            torch.save(ckpt, BEST_PATH)
            tag = f"  *** best PPO ({wr:.1%}) -> transformer_best.pt"
        else:
            no_improve += 1
            if freeze_actor:
                tag = f"  [warmup — best PPO stays {best_wr_ppo:.1%}]"
            else:
                tag = f"  (no improve, best PPO={best_wr_ppo:.1%}, streak={no_improve})"

        warmup_str = "  [critic warmup — actor frozen]" if freeze_actor else ""
        print(f"\n  -> WR={wr:.1%}  loss={avg_loss:.4f}  ent={avg_ent:.4f}  "
              f"kl={avg_kl:.4f}  clip={clipfrac:.3f}  ent_coef={ent_coef_val:.4f}"
              f"\n     reward:  events={event_sum:+.2f}  momentum={mom_sum:+.2f}  "
              f"terminal={term_sum:+.2f}  total={shaped_sum + term_sum:+.2f}"
              f"  tier={START_TIER+1}/{len(CURRICULUM_TIERS_1V1)}"
              f"{warmup_str}{tag}")

    print(f"\nDone. Best PPO WR={best_wr_ppo:.1%}")
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
