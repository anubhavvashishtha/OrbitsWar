# Strategy — TacticalQuadrantAgent (Hardened v2)

## Architecture Overview

The agent operates in five sequential phases every turn, combining **predictive tracking**, **distance-aware defense**, and **quadrant-based expansion**.

---

## Phase 0 — Predictive Fleet Tracking (Global, runs first)

Before any expansion logic:

1. **Incoming Fleet Detection**: Every enemy fleet in space has its angle compared against the direction to each friendly planet. If the angle difference is < 0.2 radians, the fleet is flagged as incoming and its ship count is added to `threatened_planets[planet_id]`.

2. **Vulnerable Enemy Detection**: An enemy planet is considered "vulnerable" if:
   - It is the **source** of an active enemy fleet in space (just spent ships), OR
   - Its current garrison is **below 20 ships**

3. **Cross-Quadrant Defense (Fix #6)**: For each threatened friendly planet, the agent sorts **all friendly planets** by distance (ignoring quadrant boundaries) and sends reinforcements from the closest available helpers — up to half of their garrison.

4. **Garrison-Drop Fallback**: The classic 1-turn-delayed detection (comparing last turn vs current ships) is kept as a secondary fallback reinforcement system.

---

## Phase 1 — Reactive Reinforcements (Per-Quadrant)

If a planet's garrison dropped unexpectedly (garrison-drop detection), same-quadrant static planets split the reinforcement deficit evenly and send help.

---

## Phase 1.5 — Punish Vulnerable Enemies

If an enemy planet in the current quadrant is flagged as vulnerable (low garrison or just launched), **tanks attack first** followed by **small peeps** — targeting the closest vulnerable enemy they can defeat.

The 95% "Snipe Bullet" (`is_snipe=True`) only fires against **enemy-owned** planets, never against neutral planets (Fix #4).

---

## Phase 2 — Quadrant Expansion

When enemies exist in the quadrant, every planet targets the **highest-production, lowest-population** enemy it can defeat. Priority order: `(-production, ships, distance)`.

No planet sends a second fleet to the same target in the same turn thanks to the `sent_targets` set (Fix #5).

---

## Phase 3 — Intercept Moving Planets

If the quadrant is fully owned and a moving planet is approaching from the previous quadrant, every static planet independently calculates its **own intercept angle** (Fix #7) and fires on the incoming target.

### Phase 3.5 — Next-Quadrant Sniping

If no incoming moving planets exist and the quadrant is fully owned, tanks and peeps fire at **vulnerable or high-production targets** in the next quadrant.

---

## Phase 4 — Peacetime Buff (Fixed #8)

When no enemies are present in the quadrant:
- If moving planets are nearby → buff them with ships (they'll carry into enemy territory)
- If no moving planets → tanks transfer ships to under-powered peeps proportionally:
  `send = 30% of the gap between tank ships and peep ships`
  No more hardcoded 30-ship cap.

---

## Moving Planets (Mid-Game) — Fixed #1 & #10

Moving planets are **always active** regardless of whether static planets exist:
1. **Priority 1**: Closest unoccupied or enemy planet in the next quadrant (high-prod first)
2. **Priority 2**: Closest unoccupied planet anywhere on the entire board

---

## End-Game Awareness (Fix #9)

After turn **950**, `aim_and_send` rejects any fleet that will not land before turn **1000**. This prevents bleeding ships into space in the final countdown.

---

## Key Design Decisions

| Feature | Detail |
|---------|--------|
| `is_snipe=True` | 95% fleet only fires on enemy-owned planets, not neutrals |
| `sent_targets` set | Prevents double-fleet on same target per turn |
| Predictive defense | Angle-based tracking of enemy fleets in flight |
| Cross-quadrant reinforcement | Ignores quadrant lines entirely for defense |
| Per-planet Phase 3 intercept | Each ship computes its own angle independently |
| Proportional peacetime buff | Based on ship gap ratio, not hardcoded threshold |
| End-game cutoff at turn 950 | Prevents ships from being wasted off the board |
