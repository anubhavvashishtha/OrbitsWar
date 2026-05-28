# Next Feature: Fleet-State Features

## Why

The current edge feature vector (81-dim) is a **static snapshot** — it describes
ships on planets and production rates, but knows nothing about in-flight fleets.
This means the model cannot reason about:

- A source planet that is about to be attacked (should not send ships away)
- A target planet that already has a friendly fleet arriving (double-send waste)
- An enemy planet getting reinforced right now (bad time to attack)
- A neutral planet being raced to by an enemy fleet (urgency to expand)

All this information exists in `fleets_raw` in the raw observation. It is just
never passed to the network.

---

## What to add

### Step 1 — Build a per-planet arrival table (one pass over fleets_raw)

Before scoring any edge, call `fleet_target_planet()` (already in `main.py`) on
every in-flight fleet to build:

```python
arrivals[planet_id] = {
    "friendly_ships": int,   # total friendly ships arriving (all ETAs)
    "enemy_ships":    int,   # total enemy ships arriving (all ETAs)
    "first_threat":   int,   # turns until first enemy fleet lands (999 if none)
    "first_friendly": int,   # turns until first friendly fleet lands (999 if none)
}
```

This table is computed once per turn and passed into `planet_features()` and
`edge_features()`.

---

### Step 2 — New per-planet features (add to `planet_features()`)

Current: 18 features. New: 18 + 6 = **24 features**.

| # | Feature | Formula | What it tells the model |
|---|---------|---------|--------------------------|
| 19 | `friendly_ships_arriving` | `arrivals[pid].friendly_ships / total_ships` (capped 1.0) | Friendly reinforcement already in flight |
| 20 | `enemy_ships_arriving` | `arrivals[pid].enemy_ships / total_ships` (capped 1.0) | Threat or enemy self-reinforcement |
| 21 | `net_arrival_balance` | `(friendly - enemy) / total_ships` (clamp -1,1) | Signed: positive = we are winning the fleet race here |
| 22 | `turns_to_first_threat` | `min(arrivals[pid].first_threat, 100) / 100.0` | Urgency: 0 = threat imminent, 1 = no threat soon |
| 23 | `already_covered` | `1.0 if arrivals[pid].friendly_ships > 0 else 0.0` | Binary: friendly fleet already heading here |
| 24 | `contested` | `1.0 if friendly > 0 and enemy > 0 else 0.0` | Binary: both sides racing to this planet |

Applied to **both source and target planet slots** in every edge.

---

### Step 3 — New per-edge features (add to `edge_features()`)

Current: 14 features. New: 14 + 3 = **17 features**.

| # | Feature | Formula | What it tells the model |
|---|---------|---------|--------------------------|
| 15 | `src_can_spare` | `(src_ships - enemy_arriving_at_src) / total_ships` (clamp -1,1) | Negative = source is under threat, cannot afford to send |
| 16 | `tgt_net_balance` | same as net_arrival_balance for tgt (repeated at edge level for attention) | Redundant but helps gradient flow in early layers |
| 17 | `src_threat_urgent` | `1.0 if first_threat_at_src <= 5 else 0.0` | Hard flag: enemy lands at source within 5 turns |

---

## New total feature dimensions

| Slice | Old | New |
|-------|-----|-----|
| global (31-d) | 31 | 31 (unchanged) |
| src planet | 18 | 24 |
| tgt planet | 18 | 24 |
| edge | 14 | 17 |
| **Total** | **81** | **96** |

---

## Implementation checklist

- [ ] `features.py`: add `fleet_arrivals_table(fleets_raw, planets_raw)` helper
- [ ] `features.py`: extend `planet_features()` signature to accept `arrivals` dict, add 6 features
- [ ] `features.py`: extend `edge_features()` signature to accept `arrivals` dict, add 3 features
- [ ] `features.py`: update `N_PLANET_FEATS = 24`, `N_EDGE_FEATS = 17`, `N_TOTAL_FEATS = 96`
- [ ] `features.py`: update `build_edge_sample()` to call arrival table and pass it through
- [ ] `features.py`: update `build_all_candidate_edges()` same
- [ ] `extract_dataset.py`: re-run to regenerate `dataset.npz` (81-dim → 96-dim)
- [ ] `imitation/model.py`: `train()` — no change needed, reads `N_TOTAL_FEATS` from features
- [ ] `transformer_trainer.py`: no change needed, feature builder is called inline
- [ ] Re-run full pipeline: imitation training → PPO fine-tuning

---

## What decisions this unlocks

**Without fleet features (current):**
```
Planet X: 5 ships → score high → send fleet
Planet X: 5 ships → score high → send ANOTHER fleet  ← double-waste
Planet A: 20 ships → enemy fleet with 25 ships arrives turn 3, model blind → sends ships away anyway → loses A
```

**With fleet features:**
```
Planet X: already_covered=1, friendly_arriving=15 → low score → skip
Planet A: src_threat_urgent=1, src_can_spare=-0.3 → don't send from here, must defend
Planet B enemy: enemy_ships_arriving=0.4 (reinforcement coming) → bad attack window → skip
Planet C enemy: net_arrival_balance=+0.2 (our fleet is winning the race) → commit more
```

---

## Notes

- Fleet target is computed via `fleet_target_planet()` in `main.py` — same logic already
  used by `build_arrival_ledger()`. Reuse it, don't duplicate.
- Use `fleets_raw` format: `[id, owner, x, y, angle, from_planet_id, ships]`
- `arrivals` table is cheap to build: O(F) where F = number of in-flight fleets (~10-30 per turn)
- This is a **full pipeline re-run** — imitation model input dimension changes,
  so `best_model.pt` will be invalid and must be retrained from scratch.
