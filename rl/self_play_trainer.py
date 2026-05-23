"""
rl/self_play_trainer.py
=======================
Fine-tune OrbitsNet via self-play against enemy agents using TD learning.

Each round:
  1. Play GAMES_PER_ROUND FFA games (your agent vs enemy agents)
  2. Collect per-turn state snapshots and final outcomes
  3. Run TD(0) updates:  V(s_t) ← γ·V(s_{t+1})  for mid-game
                         V(s_T) ← won            for terminal
  4. Save checkpoint to rl/model.pt

Run AFTER replay_trainer.py for best results (uses that checkpoint).

Usage
-----
    python rl/self_play_trainer.py
    python rl/self_play_trainer.py --rounds 20 --games 12
"""
import os, sys, glob, random, importlib.util, argparse
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classifier"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH      = ROOT / "model.pt"
BEST_MODEL_PATH = ROOT / "model_best.pt"
ENEMY_DIR       = PROJECT_ROOT / "enemy agents"
MAIN_PY         = PROJECT_ROOT / "main.py"

GAMES_PER_ROUND = 8
N_ROUNDS        = 10
LR              = 1e-4
GAMMA           = 0.99
BATCH_SIZE      = 256
TD_EPOCHS       = 3
SNAPSHOT_EVERY  = 5


# ── Helpers shared with ffa_train.py ─────────────────────────────────────
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
    return [{'id':p[0],'owner':p[1],'ships':p[5],
             'production':p[6],'x':p[2],'y':p[3]}
            for p in raw_p if p[0] not in cids]


def _read(obs, key, default=None):
    if isinstance(obs, dict): return obs.get(key, default)
    return getattr(obs, key, default)


# ── Game runner ───────────────────────────────────────────────────────────
def play_game_collect(agent_fns, agent_names, seed=0):
    """
    Play one FFA game, collect per-turn state snapshots for each player.
    Returns: trajectories  (list of lists of dicts with 'feats', 'turn', 'won')
             rewards        list of final rewards
             outcome        'WIN' | 'LOSS' | 'DRAW'
    """
    from train_from_replays import extract_features
    from kaggle_environments import make

    n   = len(agent_fns)
    env = make("orbit_wars",
               configuration={"seed": seed, "randomSeed": seed},
               debug=False)
    env.reset(num_agents=n)
    trajectories = [[] for _ in range(n)]
    states = env.step([[]] * n)
    step   = 0

    while True:
        st0    = states[0]
        status = str(_read(st0, 'status', 'ACTIVE'))
        if status != 'ACTIVE':
            break

        obss = [_read(s, 'observation') for s in states]

        if step % SNAPSHOT_EVERY == 0 and obss[0] is not None:
            rp = _read(obss[0], 'planets', []) or []
            ci = set(_read(obss[0], 'comet_planet_ids', []) or [])
            rf = _read(obss[0], 'fleets', []) or []
            sd = snap(rp, ci)
            if sd:
                for i in range(n):
                    feats = extract_features(sd, i, step, n, fleets=rf)
                    trajectories[i].append({'feats': feats, 'turn': step})

        actions = []
        for i, obs in enumerate(obss):
            try:    a = agent_fns[i](obs)
            except: a = []
            actions.append(a or [])
        states = env.step(actions)
        step  += 1

    rewards = [float(_read(s, 'reward', 0) or 0) for s in states]
    max_r   = max(rewards)
    winner  = -1 if max_r <= 0 else rewards.index(max_r)

    for i, traj in enumerate(trajectories):
        won = 1.0 if i == winner else 0.0
        for snap_d in traj:
            snap_d['won'] = won

    outcome = "WIN" if winner == 0 else ("DRAW" if winner < 0 else "LOSS")
    return trajectories, rewards, outcome


# ── TD update ─────────────────────────────────────────────────────────────
def td_update(model, optimizer, trajectories):
    """
    Temporal-difference (TD-0) update over all players' trajectories.
      mid-game:  target = γ · V(s_{t+1})
      terminal:  target = won  (sparse outcome reward)
    """
    criterion = nn.MSELoss()
    model.train()

    all_states, all_targets = [], []
    for traj in trajectories:
        if not traj:
            continue
        won = traj[0].get('won', 0.0)
        n   = len(traj)
        for idx, s in enumerate(traj):
            if idx == n - 1:
                target = won
            else:
                nf = traj[idx + 1]['feats']
                with torch.no_grad():
                    next_v = float(model(torch.FloatTensor(nf).unsqueeze(0)))
                target = GAMMA * next_v + 0.05 * won  # small outcome anchor
            all_states.append(s['feats'])
            all_targets.append(target)

    if not all_states:
        return 0.0

    X = torch.FloatTensor(all_states)
    Y = torch.FloatTensor(all_targets)
    idx_list = list(range(len(X)))

    total_loss = 0.0
    for _ in range(TD_EPOCHS):
        random.shuffle(idx_list)
        for start in range(0, len(idx_list), BATCH_SIZE):
            batch = idx_list[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(X[batch]), Y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

    return total_loss / max(1, (len(idx_list) // BATCH_SIZE) * TD_EPOCHS)


# ── Main loop ─────────────────────────────────────────────────────────────
def train(n_rounds=N_ROUNDS, games_per_round=GAMES_PER_ROUND):
    if not TORCH_AVAILABLE:
        print("Install PyTorch:  pip install torch"); return

    from net import OrbitsNet

    model = OrbitsNet()
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("No model.pt found — starting from random weights.")
        print("Tip: run replay_trainer.py first for a better starting point.\n")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    my_fn, my_name = load_agent(str(MAIN_PY), force=True)

    enemies = []
    for ef in sorted(glob.glob(str(ENEMY_DIR / "*.py"))):
        try:
            fn, name = load_agent(ef)
            enemies.append((name, fn))
            print(f"  Loaded enemy: {name}")
        except Exception as e:
            print(f"  WARN {Path(ef).name}: {e}")

    if not enemies:
        print(f"No enemy agents found in {ENEMY_DIR}"); return

    print(f"\nSelf-play TD: {n_rounds} rounds × {games_per_round} games\n{'='*55}")

    best_wr     = 0.0
    no_improve  = 0

    # load existing best if present
    if BEST_MODEL_PATH.exists():
        best_wr = 0.0  # unknown — start tracking from here

    for rnd in range(1, n_rounds + 1):
        wins = losses = draws = 0
        round_loss = 0.0

        for g in range(games_per_round):
            fillers = random.choices(enemies, k=min(3, len(enemies)))
            fns     = [my_fn]   + [fn for _, fn in fillers]
            names   = [my_name] + [nm for nm, _  in fillers]
            seed    = 42 + rnd * 1000 + g

            try:
                trajs, _, outcome = play_game_collect(fns, names, seed=seed)
            except Exception as e:
                print(f"\n  game error: {e}"); continue

            loss = td_update(model, optimizer, trajs)
            round_loss += loss

            if   outcome == "WIN":  wins   += 1
            elif outcome == "LOSS": losses += 1
            else:                   draws  += 1

            sys.stdout.write(
                f"\r  Round {rnd}/{n_rounds}  game {g+1}/{games_per_round}"
                f"  W:{wins} L:{losses} D:{draws}"
                f"  td_loss={round_loss/(g+1):.4f}  ")
            sys.stdout.flush()

        wr = wins / games_per_round
        avg_loss = round_loss / games_per_round

        # always save latest
        torch.save(model.state_dict(), MODEL_PATH)

        if wr > best_wr:
            best_wr    = wr
            no_improve = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            tag = f"  *** best so far ({wr:.1%}) → model_best.pt"
        else:
            no_improve += 1
            tag = f"  (no improvement, best={best_wr:.1%}, streak={no_improve})"
            # restore best weights so next round starts from the best known state
            if BEST_MODEL_PATH.exists():
                model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))

        print(f"\n  -> WR={wr:.1%}  avg_td_loss={avg_loss:.4f}{tag}")

    print(f"\nSelf-play training complete.  Best WR={best_wr:.1%}")
    # return best model
    if BEST_MODEL_PATH.exists():
        model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  Final model.pt ← best checkpoint ({best_wr:.1%})")
    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=N_ROUNDS,        help="Training rounds")
    p.add_argument("--games",  type=int, default=GAMES_PER_ROUND, help="Games per round")
    args = p.parse_args()
    train(n_rounds=args.rounds, games_per_round=args.games)
