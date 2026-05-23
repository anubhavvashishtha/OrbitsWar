"""
rl/ppo_agent.py  —  Full RL training pipeline entry point.

Runs all three phases in order:
  Phase 1 — Pre-train on Kaggle replays          (replay_trainer.py)
  Phase 2 — Fine-tune via self-play vs enemies   (self_play_trainer.py)
  Phase 3 — Export weights as numpy Python file  (export_weights.py)

Usage
-----
    python rl/ppo_agent.py                       # full pipeline
    python rl/ppo_agent.py --skip-replay         # skip phase 1 (use existing model.pt)
    python rl/ppo_agent.py --rounds 20 --games 10
    python rl/ppo_agent.py --export-only         # just export existing model.pt

See HOW_TO_RUN.md for full instructions.
"""
import sys, argparse
from pathlib import Path

ROOT         = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classifier"))
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    p = argparse.ArgumentParser(description="Orbit Wars RL training pipeline")
    p.add_argument("--skip-replay",  action="store_true",
                   help="Skip Phase 1 (replay pretraining), use existing model.pt")
    p.add_argument("--resume",       action="store_true",
                   help="Resume replay pretraining from existing model.pt")
    p.add_argument("--rounds",       type=int, default=10,
                   help="Self-play training rounds (Phase 2)")
    p.add_argument("--games",        type=int, default=8,
                   help="Games per round (Phase 2)")
    p.add_argument("--export-only",  action="store_true",
                   help="Only export existing model.pt, skip training")
    args = p.parse_args()

    model_path = ROOT / "model.pt"

    # ── Phase 3 only ─────────────────────────────────────────────────
    if args.export_only:
        if not model_path.exists():
            print(f"No model.pt at {model_path}. Train first."); sys.exit(1)
        from export_weights import export
        export()
        return

    # ── Phase 1: Replay pretraining ───────────────────────────────────
    if not args.skip_replay:
        print("\n" + "=" * 55)
        print("  PHASE 1 — Pre-train on Kaggle replays")
        print("=" * 55)
        from replay_trainer import train as replay_train
        model = replay_train(resume=args.resume)
        if model is None:
            print("Phase 1 failed. Exiting."); sys.exit(1)
    else:
        print(f"\nPhase 1 skipped — using {model_path}")
        if not model_path.exists():
            print("  WARNING: model.pt not found. Self-play will start from random weights.")

    # ── Phase 2: Self-play fine-tuning ────────────────────────────────
    print("\n" + "=" * 55)
    print("  PHASE 2 — Self-play vs enemy agents (TD learning)")
    print("=" * 55)
    from self_play_trainer import train as self_play_train
    model = self_play_train(n_rounds=args.rounds, games_per_round=args.games)
    if model is None:
        print("Phase 2 failed. Exiting."); sys.exit(1)

    # ── Phase 3: Export ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PHASE 3 — Export weights to rl_value_fn.py")
    print("=" * 55)
    from export_weights import export
    export()

    print("\nAll phases complete.")
    print(f"Next: follow the integration steps printed above to use rl_value_fn.py in main.py")


if __name__ == "__main__":
    main()
