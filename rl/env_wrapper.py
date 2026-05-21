"""
rl/env_wrapper.py
=================
Wraps the kaggle orbit_wars environment into a clean RL training interface
with dense reward shaping on top of the sparse ±1 terminal reward.

Reward shaping
--------------
Raw env reward is 0 every step and ±1 at the final step only.
We add a shaped reward at every step:

    shaped_r = SHAPE_SCALE * (board_value(new) - board_value(old))

board_value(obs) ≈ P(I'm winning right now), in [0, 1].

At the terminal step we also add TERMINAL_SCALE * (±1) so the agent
still learns the win/loss signal clearly.
"""

from kaggle_environments import make
import numpy as np

SHAPE_SCALE    = 0.5   # weight for per-step shaping
TERMINAL_SCALE = 2.0   # weight for final ±1 reward


class OrbitWarsEnv:
    """
    A lightweight wrapper that exposes reset() / step() for RL training.

    Parameters
    ----------
    opponent    : "random", a callable agent function, or a file-path string
    num_players : 2 or 4
    reward_shaping : bool (disable for evaluation)
    """

    def __init__(self, opponents, reward_shaping=True):
        self.opponents      = opponents
        self.num_players    = len(opponents) + 1
        self.reward_shaping = reward_shaping

        self._env          = None
        self._trainer      = None
        self._player       = 0
        self._last_bv      = 0.5
        self.step_count    = 0

    # ── Board value ─────────────────────────────────────────────────────────
    def _board_value(self, obs):
        """
        Estimate 'how well I'm doing right now'.
        Returns a float in [0, 1]; 0.25 = even in a 4-player game.
        """
        if isinstance(obs, dict):
            me          = self._player
            raw_planets = obs.get("planets", [])
            raw_fleets  = obs.get("fleets",  [])
        else:
            me          = self._player
            raw_planets = obs.planets
            raw_fleets  = obs.fleets

        my_ships = my_prod = my_planets = 0.0
        all_ships = all_prod = total_planets = 0.0

        for p in raw_planets:
            _, owner, _, _, _, ships, production = p
            all_ships    += max(ships, 0)
            total_planets += 1
            if owner == me:
                my_ships   += ships
                my_prod    += production
                my_planets += 1
            if owner != -1:
                all_prod += production

        for f in raw_fleets:
            _, owner, _, _, _, _, ships = f
            all_ships += max(ships, 0)
            if owner == me:
                my_ships += ships

        ship_ratio   = my_ships   / (all_ships   + 1e-6)
        prod_ratio   = my_prod    / (all_prod    + 1e-6)
        planet_ratio = my_planets / (total_planets + 1e-6)

        return 0.5 * ship_ratio + 0.3 * prod_ratio + 0.2 * planet_ratio

    # ── Reset ────────────────────────────────────────────────────────────────
    def reset(self):
        """Start a fresh episode. Returns the initial observation."""
        agents = [None] + self.opponents

        self._env     = make("orbit_wars", debug=False)
        self._trainer = self._env.train(agents)

        obs = self._trainer.reset()

        if isinstance(obs, dict):
            self._player = obs.get("player", 0)
        else:
            self._player = getattr(obs, "player", 0)

        self._last_bv  = self._board_value(obs)
        self.step_count = 0
        return obs

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, moves):
        """
        Apply a list of moves to the environment.

        Parameters
        ----------
        moves : list of [planet_id, angle, num_ships]

        Returns
        -------
        obs, shaped_reward, done, info
        """
        obs, reward, done, info = self._trainer.step(moves)
        self.step_count += 1

        if done:
            terminal_r   = float(reward) if reward is not None else -1.0
            shaped       = TERMINAL_SCALE * terminal_r
            self._last_bv = 0.5
        else:
            new_bv  = self._board_value(obs)
            shaped  = SHAPE_SCALE * (new_bv - self._last_bv) if self.reward_shaping else 0.0
            self._last_bv = new_bv

        return obs, shaped, bool(done), info
