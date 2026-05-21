"""
rl/features.py
==============
Converts raw Orbit Wars observations into fixed-size numpy tensors
that can be fed into a neural network.

All encoding is pure numpy — no torch dependency — so this file
can be imported both during training and inside the Kaggle submission.
"""

import math
import numpy as np

# ── Fixed tensor dimensions ────────────────────────────────────────────────
MAX_PLANETS    = 48   # 40 permanent + 8 buffer for comet planets
MAX_FLEETS     = 100  # generous upper bound
PLANET_FEATURES = 13  # per-planet feature count  (see encode_obs docstring)
FLEET_FEATURES  = 12  # per-fleet feature count
GLOBAL_FEATURES = 10  # global scalar features

# ── Action space ───────────────────────────────────────────────────────────
ANGLE_BINS   = 32
ANGLE_VALUES = np.array(
    [-math.pi + (i + 0.5) * (2 * math.pi / ANGLE_BINS) for i in range(ANGLE_BINS)],
    dtype=np.float32,
)

SHIP_FRACTIONS = np.array([0.25, 0.33, 0.50, 0.67, 0.75, 0.90], dtype=np.float32)
N_FRACTIONS    = len(SHIP_FRACTIONS)


# ──────────────────────────────────────────────────────────────────────────
def encode_obs(obs, player=None, num_players=None):
    """
    Convert a raw kaggle Orbit Wars observation into fixed-size numpy arrays.

    Planet tensor features (13 per planet, padded to MAX_PLANETS):
        [owner_neutral, owner_p0, owner_p1, owner_p2, owner_p3,  # one-hot (5)
         is_mine,                                                  # (1)
         x_norm, y_norm,                                          # (2)
         radius_norm, ships_norm, production_norm,                # (3)
         dist_to_center_norm,                                     # (1)
         is_valid]                                                 # (1) → 0 for padding

    Fleet tensor features (12 per fleet, padded to MAX_FLEETS):
        [owner_neutral, owner_p0, owner_p1, owner_p2, owner_p3,  # one-hot (5)
         is_mine,                                                  # (1)
         x_norm, y_norm,                                          # (2)
         sin_angle, cos_angle,                                    # (2)
         ships_norm,                                              # (1)
         is_valid]                                                 # (1)

    Global features (10):
        [player_p0, player_p1, player_p2, player_p3,             # one-hot (4)
         turn_norm, angular_velocity_norm,                        # (2)
         my_ships_ratio, my_prod_ratio, my_planet_ratio,          # (3)
         num_players_norm]                                        # (1) (num_players-2)/2: 0.0=1v1, 1.0=4P

    Returns
    -------
    planet_tensor   : float32 [MAX_PLANETS, PLANET_FEATURES]
    fleet_tensor    : float32 [MAX_FLEETS,  FLEET_FEATURES]
    global_tensor   : float32 [GLOBAL_FEATURES]
    my_planet_mask  : bool    [MAX_PLANETS]   True for planets I own
    valid_planet_mask: bool   [MAX_PLANETS]   True for real (non-padding) planets
    planet_ids      : list[int]               raw planet IDs at each slot
    planet_ships    : float32 [MAX_PLANETS]   raw ship counts (for action conversion)
    """
    # ── Unpack observation ─────────────────────────────────────────────────
    if isinstance(obs, dict):
        me               = player if player is not None else obs.get("player", 0)
        raw_planets      = obs.get("planets", [])
        raw_fleets       = obs.get("fleets", [])
        angular_velocity = obs.get("angular_velocity", 0.035)
        step             = obs.get("step", 0)
    else:
        me               = player if player is not None else getattr(obs, "player", 0)
        raw_planets      = obs.planets
        raw_fleets       = obs.fleets
        angular_velocity = obs.angular_velocity
        step             = getattr(obs, "step", 0)

    if num_players is None:
        num_players = 2
        for p in raw_planets:
            if p[1] >= 2:  # owner ID >= 2 means 4-player game
                num_players = 4
                break

    # ── Planet tensor ──────────────────────────────────────────────────────
    planet_tensor     = np.zeros((MAX_PLANETS, PLANET_FEATURES), dtype=np.float32)
    my_planet_mask    = np.zeros(MAX_PLANETS, dtype=bool)
    valid_planet_mask = np.zeros(MAX_PLANETS, dtype=bool)
    planet_ids        = [-1] * MAX_PLANETS
    planet_ships      = np.zeros(MAX_PLANETS, dtype=np.float32)

    my_ships = my_prod = my_planet_count = 0.0
    all_ships = all_prod = total_planet_count = 0.0

    for i, p in enumerate(raw_planets[:MAX_PLANETS]):
        pid, owner, x, y, radius, ships, production = p

        owner_oh = [0.0] * 5
        if owner == -1:
            owner_oh[0] = 1.0
        elif 0 <= owner <= 3:
            owner_oh[owner + 1] = 1.0

        is_mine = 1.0 if owner == me else 0.0
        dist    = math.hypot(x - 50.0, y - 50.0)

        planet_tensor[i] = [
            *owner_oh,               # 5
            is_mine,                 # 1
            x / 100.0,               # 1
            y / 100.0,               # 1
            radius / 3.0,            # 1
            min(ships / 500.0, 1.0), # 1
            production / 5.0,        # 1
            dist / 70.0,             # 1
            1.0,                     # is_valid = 1
        ]

        valid_planet_mask[i] = True
        planet_ids[i]        = int(pid)
        planet_ships[i]      = float(ships)
        total_planet_count  += 1

        if owner == me:
            my_planet_mask[i]  = True
            my_ships          += ships
            my_prod           += production
            my_planet_count   += 1

        if owner != -1:
            all_ships += ships
            all_prod  += production

    # ── Fleet tensor ───────────────────────────────────────────────────────
    fleet_tensor = np.zeros((MAX_FLEETS, FLEET_FEATURES), dtype=np.float32)

    for i, f in enumerate(raw_fleets[:MAX_FLEETS]):
        fid, owner, x, y, angle, from_pid, ships = f

        owner_oh = [0.0] * 5
        if 0 <= owner <= 3:
            owner_oh[owner + 1] = 1.0

        is_mine = 1.0 if owner == me else 0.0

        fleet_tensor[i] = [
            *owner_oh,               # 5
            is_mine,                 # 1
            x / 100.0,               # 1
            y / 100.0,               # 1
            math.sin(angle),         # 1
            math.cos(angle),         # 1
            min(ships / 500.0, 1.0), # 1
            1.0,                     # is_valid
        ]

        if owner == me:
            my_ships  += ships
        all_ships += ships

    # ── Global tensor ──────────────────────────────────────────────────────
    player_oh = [0.0, 0.0, 0.0, 0.0]
    if 0 <= me <= 3:
        player_oh[me] = 1.0

    global_tensor = np.array([
        *player_oh,                                          # 4
        step / 500.0,                                        # 1
        (angular_velocity - 0.025) / 0.025,                 # 1
        my_ships / (all_ships + 1e-6),                       # 1
        my_prod  / (all_prod  + 1e-6),                       # 1
        my_planet_count / (total_planet_count + 1e-6),       # 1
        (num_players - 2) / 2.0,                             # 1  0.0=1v1  1.0=4P
    ], dtype=np.float32)

    return (planet_tensor, fleet_tensor, global_tensor,
            my_planet_mask, valid_planet_mask,
            planet_ids, planet_ships)


def actions_to_moves(send_mask, angle_bins, fraction_bins, planet_ids, planet_ships):
    """
    Convert per-planet action arrays back into a list of Orbit Wars moves.

    Parameters
    ----------
    send_mask     : bool  array [MAX_PLANETS]  — True = send a fleet
    angle_bins    : int   array [MAX_PLANETS]  — ANGLE_BINS index
    fraction_bins : int   array [MAX_PLANETS]  — SHIP_FRACTIONS index
    planet_ids    : list  [MAX_PLANETS]        — real planet IDs
    planet_ships  : float array [MAX_PLANETS]  — current garrison sizes

    Returns
    -------
    moves : list of [planet_id, angle_radians, num_ships]
    """
    moves            = []
    remaining_ships  = planet_ships.copy()

    for i in range(MAX_PLANETS):
        if not send_mask[i] or planet_ids[i] == -1:
            continue

        ships_avail = int(remaining_ships[i])
        if ships_avail <= 0:
            continue

        fraction  = float(SHIP_FRACTIONS[fraction_bins[i]])
        num_ships = max(1, int(ships_avail * fraction))
        angle     = float(ANGLE_VALUES[angle_bins[i]])

        moves.append([planet_ids[i], angle, num_ships])
        remaining_ships[i] -= num_ships   # prevent over-sending in one turn

    return moves
