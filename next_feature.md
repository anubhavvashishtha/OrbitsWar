# Next Features & Full Improvement Plan

---

## PHASE 1 — Curriculum PPO (Kaggle, in progress)

**Status: ACTIVE — Tier 1 DONE, moving to Tier 2**

Session-by-session plan using `START_TIER` in the notebook:

| Session | START_TIER | N_ROUNDS | Mode | Goal |
|---|---|---|---|---|
| S1 (done) | 0 | 30 | 1v1 | Tier 1 easy bots — reached 80% WR ✓ |
| S2 (next) | 1 | 30 | 1v1 | Check learning on Tier 1+2 medium bots |
| S3 | 2 | 150 | 1v1 | Full curriculum all 15 bots, stabilize WR |

Enemy tiers (from tournament results):
- Tier 1 easy:   tamrazov, pascal, main_5, yuriygreben, sigmaborov-reinforce
- Tier 2 medium: sigmaborov-starter, main_4, main_2, regressor, main_3
- Tier 3 hard:   pilkwang, ykhnkf, classifier_bot, main_1, main

Expected ELO after Phase 1: ~850-900

---

## PHASE 2 — Local 24h Run (RTX 4050, ~576 rounds)

**Status: READY — code implemented, waiting for Phase 1 completion**

Local setup confirmed: PyTorch 2.5.1+cu121, RTX 4050 6.4GB, Ryzen 7 8-core.
Estimated speed: ~2.5 min/round → 576 rounds per 24h.

Settings for local run:
    FORCE_1V1          = False         # 50/50 mix of 1v1 and 4v4
    SELFPLAY_ENABLED   = True          # enable after Phase 1 is done
    START_TIER         = 2             # all 15 curriculum enemies
    N_ROUNDS           = 9999          # run until manually stopped

### 2a. Self-Play Snapshot System (IMPLEMENTED)

Every 15 rounds, a frozen copy of the current actor is saved as an enemy.
Keeps max 10 snapshots, drops oldest. No ceiling on learning.

Constants: SELFPLAY_ENABLED, SELFPLAY_SNAPSHOT_EVERY=15, SELFPLAY_MAX_SNAPSHOTS=10

### 2b. 4v4 Placement-Based Terminal Reward (IMPLEMENTED)

1v1: +1.0 WIN / -1.0 LOSS (unchanged — was working fine)
4v4: placement-based (independent of 1v1 values):
    1st:  +1.0  (same as 1v1 win)
    2nd:  +0.3  (small positive, not punished)
    3rd:  -0.5  (moderate negative)
    4th:  -1.0  (same as 1v1 loss)

### 2c. Mixed 1v1 + 4v4 Training (IMPLEMENTED)

50/50 split (g % 2 == 0 → 1v1, else → 4v4).
4v4 opponents: 3 random picks from active_enemies (includes snapshots).

Expected ELO after Phase 2: ~950-1050

---

## Terminal Reward Decision

**1v1: +1.0 / -1.0 (original values — kept)**

Reason: +2/-3 asymmetric was considered but decided against.
The Tier 1 training reached 80% WR cleanly with +1/-1.
The collapse was caused by bad curriculum + high LR, not weak terminal signal.
Asymmetric values risk making the agent overly conservative (especially vs hard bots).
Revisit ONLY if agent plateaus with good event rewards but low WR.

---

## PHASE 3 — Architecture Upgrade (if needed)

Only if plateau persists after Phases 1-3:
Current: d_model=128, 3 layers, 651k params
Option:  d_model=256, 4 layers → ~2.6M params (4x bigger)

---

## TARGET

Medal threshold: 1050 ELO
Current:         ~794 ELO (after Phase 1 S1)
Gap:             +256 ELO

Realistic with Phases 1-2: 950-1050 ELO
