# Tactical Quadrant Defensive Agent: Comprehensive Strategy

This document outlines the complete, step-by-step strategy executed by your `TacticalQuadrantAgent` in the game of Orbit Wars.

## 1. Core Concepts & Definitions
- **Quadrants:** The game board is divided into 4 quadrants relative to the center (0, 1, 2, 3). The agent organizes all of its operations quadrant-by-quadrant.
- **Previous Quadrant:** Calculated based on the rotational direction (`angular_velocity`). This is the quadrant from which moving planets will inevitably enter our "home" quadrant.
- **Tanks:** Friendly stationary planets that are heavy producers (`production >= 3`).
- **Small Peeps:** Friendly stationary planets that are low producers (`production < 3`).
- **Target Lock:** To avoid wasting ships, the agent remembers which planets it recently sent a fleet to. A target is locked and ignored for `ETA + 2 turns` to give the fleet time to arrive and conquer.

---

## 2. Turn Execution Flow
Every turn, the agent processes information and executes actions in the following strict priority order:

### Phase 0: Attack Detection (Sudden Drop Logic)
Before making any moves, the agent audits the garrison (ship count) of every planet we own.
- It compares the current ships against what was expected (last turn's ships + production - ships we sent away).
- If a planet has strictly *fewer* ships than expected, it means an enemy fleet struck it.
- The planet is instantly flagged as **Under Attack** and the exact number of lost ships is recorded.

### Phase 1: Initial Expansion (The Genesis Protocol)
If the game starts and we **do not own any stationary planets** (our main planet is moving):
- Our moving planet scans the entire board globally.
- It targets the unowned stationary planet that minimizes a combination of **lowest population** and **closest distance**.
- Once that planet is captured, its quadrant officially becomes our first "Home Quadrant," and the rest of the strategy kicks in.

### Phase 2: Emergency Reinforcements (Highest Priority)
For every quadrant we have a presence in:
- The agent checks if any friendly planets in this quadrant were flagged as **Under Attack** in Phase 0.
- If so, all other friendly stationary planets in that same quadrant pause their offensive plans.
- They calculate the exact number of ships the attacked planet lost (plus 1) and divide that burden evenly among themselves.
- They immediately send these reinforcements to rebuild the attacked planet's garrison.

### Phase 3: Invader Defense (Heavy vs Heavy)
If enemy planets (either moving or static) are detected inside our quadrant:
- **Tanks** seek out and attack **Heavy Enemy Producers** (`production >= 3`) in the quadrant. If none exist, they help clean up small enemies.
- **Small Peeps** seek out and attack **Small Enemy Producers** (`production < 3`) in the quadrant. If none exist, they help swarm the heavy enemies.

### Phase 4: Shared Fleet Anticipation
If our quadrant is completely secure (we own all stationary planets in it):
- The agent looks at the **Previous Quadrant** to find unowned or enemy moving planets that are rotating towards us.
- It calculates the ships needed to capture the most threatening incoming planet.
- It divides this cost evenly among all our stationary planets in the home quadrant.
- A massive, perfectly synchronized **Shared Fleet** is launched to intercept the incoming planet before it becomes a problem.

### Phase 5: Peacetime Buffing (Support Network)
If our quadrant is completely secure, AND there are no enemies approaching from the previous quadrant:
- It is officially peacetime.
- **Tanks** look for friendly **Small Peeps** in their quadrant that have a dangerously low population (less than 30 ships).
- Tanks will send exactly 1 turn's worth of their production as a care package to buff the Small Peeps' defenses against future attacks.

### Phase 6: Exploration (Moving Planets)
Moving planets that wander outside of our fully secured quadrants act as explorers:
- They actively hunt for the closest unowned stationary planets in whatever quadrant they are currently flying through.
- If their current quadrant is already fully owned by us, they aggressively target the closest unowned moving planets instead.

---

## 3. Launch Conditions
No fleet is ever launched recklessly. For a planet to execute *any* of the attacks or reinforcements listed above, it must verify:
1. It has a sun-safe flight path.
2. It currently has enough available ships to successfully fulfill its share of the attack (meaning it won't send piecemeal fleets that just bounce off the enemy's defenses).
3. The target isn't already "Target Locked" by a previous launch.
