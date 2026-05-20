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

### Phase 1.5: Bait and Punish (Vulnerable Target Sniping)
Before attacking normal targets, the agent scans the entire system for enemy fleets currently in transit.
- It identifies the exact enemy planet that launched those fleets (`from_planet_id`).
- These planets are flagged as **Vulnerable** because their garrisons are depleted.
- Both **Tanks** and **Small Peeps** will immediately drop other offensive priorities to snipe these vulnerable planets in their quadrant or the next quadrant.

### Phase 2: Invader Defense (Baiting)
If enemy planets (either moving or static) are detected inside our quadrant:
- **Tanks** seek out and attack **Small Enemy Producers** (`production < 3`) first. This acts as bait—forcing the enemy to send support fleets (which we punish in Phase 1.5). If no small targets exist, they attack heavy producers.
- **Small Peeps** also prioritize **Small Enemy Producers**, swarming them, or helping with heavy enemies if needed.

### Phase 3: Shared Fleet Anticipation
If our quadrant is completely secure (we own all stationary planets in it):
- The agent looks at the **Previous Quadrant** to find unowned or enemy moving planets that are rotating towards us.
- A massive, perfectly synchronized **Shared Fleet** is launched from all our stationary planets in the home quadrant to intercept the incoming planet before it becomes a problem.

### Phase 3.5: Next Quadrant Expansion
If our quadrant is secure AND there are no incoming threats from the previous quadrant:
- The agent looks forward into the **Next Quadrant**.
- It identifies any stationary planets we don't own (neutral or enemy-occupied).
- Our stationary planets launch coordinated strikes to snipe these targets and begin aggressive expansion, prioritizing vulnerable or small targets first.

### Phase 4: Peacetime Buffing (Support Network)
If there are no immediate threats in the quadrant:
- **Tanks** first look for **Friendly Moving Planets (Comets)** currently traveling through their quadrant. They will send a large buff (2 turns of production) to reinforce the comet before it ventures into enemy territory.
- If no comets are around and no enemies are in the previous quadrant, Tanks look for friendly **Small Peeps** with dangerously low populations (less than 30 ships) and send 1 turn's worth of production as a care package.

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
