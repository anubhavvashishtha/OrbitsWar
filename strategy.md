# TacticalQuadrantAgent — Complete Strategy Breakdown

> **Agent name**: `TacticalQuadrantAgent`
> **File**: `main.py` (435 lines)
> **Architecture**: Stateful class instantiated once, called every turn via `agent(obs, config)`

---

## Table of Contents

1. [Core Philosophy](#1-core-philosophy)
2. [Constants & Tuning Knobs](#2-constants--tuning-knobs)
3. [Planet Classification System](#3-planet-classification-system)
4. [Navigation & Aiming Engine](#4-navigation--aiming-engine)
5. [State Management (Turn-over-Turn Memory)](#5-state-management-turn-over-turn-memory)
6. [The Quadrant System](#6-the-quadrant-system)
7. [Fleet Sizing Logic (`aim_and_send`)](#7-fleet-sizing-logic-aim_and_send)
8. [Execution Phases (Decision Order)](#8-execution-phases-decision-order)
   - [Phase 0: Opening — Moving Planet Bootstrapping](#phase-0-opening--moving-planet-bootstrapping)
   - [Phase 1: Reinforcements](#phase-1-reinforcements)
   - [Phase 1.5: Punish Vulnerable Enemies](#phase-15-punish-vulnerable-enemies)
   - [Phase 2: Defense / Expansion in Quadrant](#phase-2-defense--expansion-in-quadrant)
   - [Phase 3: Previous Quadrant — Shared Fleet Intercept](#phase-3-previous-quadrant--shared-fleet-intercept)
   - [Phase 3.5: Next Quadrant Sniping](#phase-35-next-quadrant-sniping)
   - [Phase 4: Peacetime Buffing](#phase-4-peacetime-buffing)
   - [Phase 5: Moving Planet Normal Logic](#phase-5-moving-planet-normal-logic)
9. [Target Locking System (`_targeted`)](#9-target-locking-system-_targeted)
10. [Vulnerability Detection](#10-vulnerability-detection)
11. [Full Decision Flowchart](#11-full-decision-flowchart)
12. [Summary of Strategic Identity](#12-summary-of-strategic-identity)

---

## 1. Core Philosophy

The agent divides the 100×100 board into **4 quadrants** centered on the sun and fights a **sector-by-sector territorial war**. The fundamental strategy is:

1. **Secure your quadrant** — own every static planet in your sector first
2. **Expand outward** — once your quadrant is clean, push into adjacent quadrants
3. **React to attacks** — detect ship losses retroactively and send reinforcements
4. **Exploit weakness** — opportunistically hit enemy planets that just sent fleets away
5. **Buffer in peacetime** — when there's nothing to fight, stockpile ships on moving planets or weak allies

The agent treats static planets (outer ring, beyond rotation radius) as the **backbone of territory** and moving planets (inner ring, orbiting the sun) as **secondary assets** — mobile strike platforms or ship sinks.

---

## 2. Constants & Tuning Knobs

| Constant | Value | Purpose |
|---|---|---|
| `CENTER_X`, `CENTER_Y` | `CENTER` (from game) | Sun position (center of board) |
| `SUN_R` | `10.0` | Sun's collision radius |
| `MAX_SPEED` | `6.0` | Maximum fleet speed (for large fleets) |
| `SUN_SAFETY` | `1.5` | Extra clearance buffer added to sun radius when checking fleet paths |
| `ROTATION_LIMIT` | `50.0` | Distance threshold: planets inside this orbit; planets outside are static |
| `LAUNCH_CLEARANCE` | `0.1` | Ships launch from `planet.radius + 0.1` to avoid self-collision |
| `INTERCEPT_TOLERANCE` | `1` | Allowed turn mismatch when intercepting moving planets (±1 turn) |
| `HORIZON` | `110` | Maximum future turns considered for any calculation |
| `ROUTE_SEARCH_HORIZON` | `60` | Maximum turns searched in brute-force intercept finding |

### Speed Formula

```
fleet_speed(ships):
    if ships <= 1: return 1.0
    ratio = clamp(log(ships) / log(1000), 0, 1)
    return 1.0 + 5.0 * ratio^1.5
```

- **1 ship** → speed 1.0/turn
- **~30 ships** → speed ~2.5/turn
- **~100 ships** → speed ~3.8/turn
- **1000 ships** → speed 6.0/turn (max)

Bigger fleets move faster. This means large all-in attacks arrive sooner than trickle reinforcements.

---

## 3. Planet Classification System

Every planet is classified along two axes:

### Static vs Moving

```python
is_static_planet(planet, initial_by_id):
    # Uses INITIAL position (turn 0), not current position
    distance_from_center + planet.radius >= ROTATION_LIMIT (50.0)
```

- **Static planets**: outer ring, don't orbit, permanent positions
- **Moving planets**: inner ring, orbit the sun with `angular_velocity`

### Production Tier

- **Tanks**: `production >= 3` — high-value backbone planets
- **Small peeps**: `production < 3` — low-value supporting planets

### Ownership Categories

- **My planets** (`owner == player`)
- **Unoccupied** (`owner != player` AND not in `_targeted` dict) — includes both neutral and enemy
- **Comets** — identified via `comet_planet_ids`, **completely excluded** from all logic

---

## 4. Navigation & Aiming Engine

The agent has a sophisticated aiming system with multiple fallback layers to handle both static and orbiting targets.

### Layer 1: Basic Geometry (`actual_path_geometry`)

Given source planet `(sx, sy, sr)` and target point `(tx, ty, tr)`:
1. Compute direct angle: `atan2(ty - sy, tx - sx)`
2. Compute launch point: `sr + 0.1` units from source center along that angle
3. Compute travel distance: total center-to-center distance minus both radii minus launch clearance
4. Return: angle, launch point, endpoint, travel distance

### Layer 2: Sun Safety Check (`safe_angle_and_distance`)

Wraps Layer 1 and checks whether the fleet path segment comes within `SUN_R + SUN_SAFETY` (11.5 units) of the board center. Returns `None` if the path clips the sun.

### Layer 3: Arrival Estimation (`estimate_arrival`)

Wraps Layer 2. Computes travel time in turns:
```
turns = ceil(distance / fleet_speed(ships))
```
Returns `(angle, turns)` or `None` if path hits sun.

### Layer 4: Iterative Prediction (`aim_with_prediction`)

For **moving targets** that orbit the sun, the agent uses iterative convergence:

1. Estimate arrival time to target's **current** position
2. Predict where the target will be in that many turns
3. Re-estimate arrival to the **predicted** position
4. Repeat up to **5 iterations** until position delta < 0.3 and turn delta ≤ 1

This handles the "lead your shot" problem — the target moves while your fleet is in transit.

**Fallback**: If the iterative method fails (e.g., sun blockage at every predicted position), it calls `search_safe_intercept`.

### Layer 5: Brute-Force Intercept Search (`search_safe_intercept`)

Last resort for moving targets. Brute-forces all future turns from 1 to `ROUTE_SEARCH_HORIZON` (60):

1. For each candidate turn `t`: predict target position at turn `t`
2. Estimate fleet arrival time to that position
3. Accept if `|arrival_turns - t| <= INTERCEPT_TOLERANCE` (within 1 turn)
4. Among valid intercepts, pick the one with smallest `(delta, arrival_turns, candidate_turns)` — prioritizes accuracy, then speed

**For static targets**: If sun blocks the direct path → `None` (no attack possible). No fallback.

---

## 5. State Management (Turn-over-Turn Memory)

The agent maintains persistent state across turns:

| State | Type | Purpose |
|---|---|---|
| `_initial_by_id` | `{id: Planet}` | Snapshot of all planets from turn 1 — used to determine static vs moving |
| `_targeted` | `{planet_id: expire_turn}` | Planets currently being attacked — prevents duplicate targeting |
| `current_turn` | `int` | Turn counter (starts at 0, incremented each call) |
| `last_turn_ships` | `{id: ships}` | Ship counts from previous turn — used for attack detection |
| `last_turn_owner` | `{id: owner}` | Ownership from previous turn — used for attack detection |
| `last_turn_ejected` | `{id: ships_sent}` | Ships the agent sent FROM each planet last turn — subtracted from expected count |

### Attack Detection Logic

For each planet the agent owned **both last turn and this turn**:
```
expected_ships = last_turn_ships + production - ships_we_sent
actual_ships   = current ships
if actual < expected:
    lost = expected - actual   # enemy fleet landed!
    mark as under_attack
    reinforcement_needs[planet] = lost + 1
```

This is **retroactive detection only** — the agent discovers attacks 1 turn after they land. It has no forward-looking threat assessment.

---

## 6. The Quadrant System

The board is divided into 4 quadrants by the sun's position:

```
         y >= CENTER
    Q1 (x<0,y≥0)  |  Q0 (x≥0,y≥0)
   ────────────────┼────────────────
    Q2 (x<0,y<0)  |  Q3 (x≥0,y<0)
         y < CENTER
```

### Directional Rotation

The "previous" and "next" quadrant depend on `angular_velocity`:

- If `ang_vel >= 0` (counter-clockwise rotation):
  - **Previous quadrant**: `(q - 1) % 4` — planets rotate *from* here into `q`
  - **Next quadrant**: `(q + 1) % 4` — planets in `q` rotate *towards* here
- If `ang_vel < 0` (clockwise): reversed

This is used to predict which moving planets will enter your quadrant soon (previous) and where your moving planets are heading (next).

### Quadrant Ownership

```python
quadrant_static_owned[q] = True if ALL static planets in q belong to you
                           (also True if there are no static planets in q)
```

This determines whether a quadrant is "secured" — triggering expansion into adjacent quadrants.

---

## 7. Fleet Sizing Logic (`aim_and_send`)

The central dispatch function. Given a source planet, target planet, and minimum ship requirement:

### For Attacks (target not owned by player):

1. **Reserve**: `max(1, 5% of available ships)` — kept on the source planet
2. **Send amount**: `max(min_needed, available - reserve)` — sends nearly everything
3. **Iterative refinement** (up to 4 rounds):
   - Aim at target with current send amount
   - If enemy-owned: recalculate needed as `target_ships + production × travel_turns + 1`
   - If neutral: needed = `target_ships + 1`
   - If recalculated need ≤ send amount → commit
   - If recalculated need > available → abort
   - Otherwise: increase send amount and retry

### For Reinforcements (target owned by player):

1. **No reserve** — send exactly the requested amount
2. **No production adjustment** — send `min_needed_ships` as-is

### On Success:

- Append move `[source_id, angle, ships]`
- Deduct from `available_ships[source_id]`
- If attack: add target to `_targeted` with expiry `current_turn + travel_turns + 2`

### Key Implication:

For attacks, the agent almost always sends **95% of its ships**. The 5% reserve is a minimal buffer. This is an aggressive all-in strategy that leaves source planets nearly defenseless after attacking.

---

## 8. Execution Phases (Decision Order)

Phases execute in strict order. Each phase deducts from `available_ships`, so earlier phases have priority over later ones.

---

### Phase 0: Opening — Moving Planet Bootstrapping
**Lines 264–270** | **Trigger**: Agent has moving planets but NO static planets

This only fires at the very start of the game (or after losing all static territory).

**Logic**:
- For each owned moving planet:
  - If there are unoccupied static planets: attack the one with `(fewest ships, shortest distance)`
  - Otherwise: do nothing (`pass`)

**Purpose**: Establish a static foothold. Moving planets are unreliable as a sole base — get onto the outer ring ASAP.

---

### Phase 1: Reinforcements
**Lines 296–303** | **Trigger**: Planets in this quadrant detected as `under_attack`

**Logic**:
- For each attacked planet in quadrant `q`:
  - Find all other owned static planets in `q` (helpers)
  - Split reinforcement need equally: `ceil(needed / num_helpers)`
  - Each helper sends that amount to the attacked planet

**Details**:
- Uses `aim_and_send` with the attacked planet as target (which is owned → no reserve deduction)
- Reinforcements come from static planets only — moving planets don't participate
- Equal splitting regardless of distance or available ships

---

### Phase 1.5: Punish Vulnerable Enemies
**Lines 305–316** | **Trigger**: Enemies in this quadrant who recently launched fleets (`vulnerable_planet_ids`)

**Logic**:
- `vulnerable_planet_ids` = set of `from_planet_id` from all enemy fleets currently on the board
- For each tank, then each small peep:
  - Filter vulnerable enemies to those with fewer ships than available
  - Attack the nearest one

**Purpose**: Opportunistic counter-attack. If an enemy just sent ships away, their source planet is weak — hit it now.

---

### Phase 2: Defense / Expansion in Quadrant
**Lines 318–336** | **Trigger**: There are enemies (non-player planets) in this quadrant

**Logic** (for both tanks and small peeps, tanks first):
1. First try to attack **small enemies** (`production < 3`), nearest first
   - Skip planets already in `vulnerable_planet_ids` (handled in Phase 1.5)
   - Must have fewer ships than attacker's available
2. If no valid small target: try **heavy enemies** (`production >= 3`), nearest first
3. Attack via `aim_and_send`

**Targeting priority**: Small/weak targets first (easy captures), then heavy targets. Always nearest-first within tier.

---

### Phase 3: Previous Quadrant — Shared Fleet Intercept
**Lines 338–359** | **Trigger**: Quadrant `q` is fully owned AND there are no enemies in `q`

Looks at the **previous quadrant** (where moving planets rotate FROM toward this quadrant).

**Logic**:
- Find unoccupied moving planets in the previous quadrant
- Pick the one closest to any of our static planets in `q`
- Calculate needed ships (iterative refinement, up to 4 rounds):
  - Neutral: `ships + 1`
  - Enemy: `ships + production × turns + 1`
- **Split the needed ships across ALL static planets in the quadrant**
- Each planet sends `ceil(needed / count)` ships to the same target

**Purpose**: Coordinated multi-planet attack on moving planets approaching your sector. This is the agent's ONLY multi-planet coordination mechanic.

**Important details**:
- The aim is calculated using only `my_q_static[0]` (first planet in list) — all other planets aim independently via `aim_and_send`
- Ships are split evenly regardless of distance
- Fleets will NOT arrive at the same time since planets are at different distances

---

### Phase 3.5: Next Quadrant Sniping
**Lines 361–388** | **Trigger**: Phase 3 fired but found no moving planet targets in previous quadrant

Looks at the **next quadrant** (where planets rotate TOWARD from this quadrant).

**Logic** (for tanks, then small peeps):
1. Find unoccupied static planets in the next quadrant
2. Priority order:
   - **Vulnerable** targets first (just sent fleets)
   - **Small** targets second (`production < 3`)
   - **Heavy** targets last (`production >= 3`)
3. Within each tier: nearest first, must have fewer ships than available
4. Attack via `aim_and_send`

**Purpose**: Cross-quadrant expansion when your home quadrant is secure and the previous quadrant is clear.

---

### Phase 4: Peacetime Buffing
**Lines 390–407** | **Trigger**: No enemies exist in this quadrant

**Logic**:
1. **If owned moving planets exist in this quadrant**:
   - Each tank sends `production × 2` ships to the nearest owned moving planet
   - Purpose: Stockpile ships on mobile platforms

2. **Else if no enemies in the previous quadrant either**:
   - Each tank sends `production × 1` ships to the weakest small peep (must have < 30 ships)
   - Purpose: Shore up weak planets when there's truly nothing else to do

**Purpose**: Productive use of idle ships during peace. Tanks drip-feed ships to moving planets (for future offensive) or to weak neighbors (for defense).

---

### Phase 5: Moving Planet Normal Logic
**Lines 409–421** | **Trigger**: Agent has static planets AND owned moving planets (runs after all quadrant phases)

**Logic for each owned moving planet**:
- Get the moving planet's current quadrant `q`
- **If quadrant `q` is fully owned (all static planets mine)**:
  - Attack the nearest unoccupied moving planet (any quadrant)
- **Else (quadrant not secured)**:
  - Attack the nearest unoccupied static planet IN the same quadrant

**Purpose**: Moving planets either help secure their local quadrant (attack static enemies) or, if the quadrant is safe, hunt other moving planets globally.

---

## 9. Target Locking System (`_targeted`)

When a planet is attacked, its ID is added to `_targeted` with an expiry time of `current_turn + travel_turns + 2`.

**Effects**:
- Locked planets are **excluded from `unoccupied`** list — no other planet will independently attack them
- Prevents waste from multiple planets attacking the same target

**Cleanup**: Each turn, expired entries are removed:
```python
for tid, expire_turn in list(self._targeted.items()):
    if self.current_turn >= expire_turn:
        del self._targeted[tid]
```

**Limitations**:
- Lock persists even if the fleet is destroyed en route (sun, combat)
- Lock persists even if target changes owner (captured by someone else)
- Prevents intentional multi-planet coordinated attacks on the same target (except Phase 3's explicit split)
- The `+2` buffer is arbitrary — may be too short or too long depending on circumstances

---

## 10. Vulnerability Detection

The agent identifies "vulnerable" enemy planets — those that just launched fleets and are likely weakened.

```python
enemy_fleets = [f for f in raw_fleets if f[1] != player and f[1] != -1]
vulnerable_planet_ids = {f[5] for f in enemy_fleets if len(f) > 5 and f[5] is not None}
```

- `f[5]` is `from_planet_id` — the source planet the fleet launched from
- If an enemy fleet exists on the board, its source planet is considered "vulnerable"
- **No estimation** of how many ships actually left — could be 1 ship or 1000

Vulnerable planets get **priority targeting** in Phase 1.5 and Phase 3.5 (attacked before regular enemies).

---

## 11. Full Decision Flowchart

```
Turn starts
│
├─ Expire stale _targeted entries
├─ Parse observation (planets, fleets, comets)
├─ Filter out comets
├─ Detect attacks via ship count delta
├─ Classify planets (static/moving, tanks/peeps, by quadrant)
│
├─ PHASE 0: Moving planets with no static base?
│   └─ Attack nearest low-ship static planet
│
├─ FOR EACH QUADRANT q (0→3):
│   │
│   ├─ PHASE 1: Any attacked planets in q?
│   │   └─ Distribute reinforcements from all helpers in q
│   │
│   ├─ PHASE 1.5: Any vulnerable enemies in q?
│   │   └─ Tanks then peeps attack nearest vulnerable enemy
│   │
│   ├─ PHASE 2: Any enemies in q?
│   │   ├─ YES → Tanks then peeps attack (small first, then heavy, nearest)
│   │   └─ NO, but q is fully owned →
│   │       ├─ PHASE 3: Moving planets in previous quadrant?
│   │       │   └─ YES → Shared fleet attack (split ships across q's static planets)
│   │       │   └─ NO →
│   │       │       └─ PHASE 3.5: Static enemies in next quadrant?
│   │       │           └─ Attack (vulnerable → small → heavy, nearest)
│   │
│   └─ PHASE 4: No enemies in q?
│       ├─ Owned moving planets in q? → Tanks buff them (production×2)
│       └─ No moving planets, prev q also clear? → Tanks buff weak peeps
│
├─ PHASE 5: Moving planet normal logic
│   ├─ Quadrant secured? → Attack nearest unoccupied moving planet (global)
│   └─ Quadrant NOT secured? → Attack nearest unoccupied static planet (local)
│
└─ Record state (ships, owners, ejected) for next turn's attack detection
```

---

## 12. Summary of Strategic Identity

| Aspect | Description |
|---|---|
| **Macro strategy** | Sector-by-sector territorial expansion from starting quadrant outward |
| **Micro strategy** | Nearest-first targeting, all-in fleet sizing with 5% reserve |
| **Defense** | Retroactive only — detects attacks after damage, then splits reinforcements |
| **Offense** | Greedy local attacks; opportunistic strikes on weakened enemies |
| **Coordination** | Minimal — only Phase 3 splits a fleet across multiple planets (unsynchronized) |
| **Economy** | No explicit economy management; peacetime buffing is the only ship redistribution |
| **Comets** | Completely ignored |
| **Endgame** | No awareness of timer or score differential |
| **Multi-player** | All opponents treated identically — no threat assessment or alliance logic |
| **Planet valuation** | Binary (tank vs peep at production 3 threshold); production magnitude not weighted |
| **Sun routing** | Binary — direct path or give up; no relay/waypoint routing |

**In one sentence**: A reactive, quadrant-locked, nearest-first greedy expander that fights with overwhelming local force but lacks coordination, foresight, and strategic depth.
