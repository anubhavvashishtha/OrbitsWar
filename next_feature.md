# Next Features / Ideas

## 1. Terminal Reward Tuning — DECIDED

**Problem:**
With TERMINAL_WIN=1 / TERMINAL_LOSS=-1, even a losing game gives net positive
reward because shaped rewards (events + momentum) average ~+2 per game.
Agent learns to capture planets without being penalized for losing.

**Decision: WIN=+2 / LOSS=-3 (asymmetric)**

Reasoning:
- Losing is more costly than winning is beneficial
- Break-even WR = 60%  (agent must win >60% to feel rewarded)
- At 50% WR: terminal = 8x2 + 8x(-3) = -8  (still punished at coin-flip)
- Losing game net: 2.0 (shaped) - 3 = -1.0  (clearly negative)
- Winning game net: 2.0 (shaped) + 2 = +4.0  (clearly positive)

**Status:** Updated in transformer_trainer.py — takes effect next session.
