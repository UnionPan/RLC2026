"""Double-Oracle Fictitious Play (two-player, zero-sum) scaffold."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


class DoubleOracleFictitiousPlay:
    """Double-Oracle fictitious play for two-player zero-sum games.

    You provide:
      - payoff_fn(policy_a, policy_b) -> float (payoff for player A)
      - best_response_fn(player, opponent_mixed) -> policy
    """

    def __init__(
        self,
        payoff_fn: Callable[[object, object], float],
        best_response_fn: Callable[[str, Dict[object, float]], object],
        init_policy_a,
        init_policy_b,
        max_iters: int = 50,
    ):
        self.payoff_fn = payoff_fn
        self.best_response_fn = best_response_fn
        self.max_iters = max_iters

        self.policies_a: List[object] = [init_policy_a]
        self.policies_b: List[object] = [init_policy_b]
        self.meta_game = np.zeros((1, 1), dtype=np.float32)
        self.meta_game[0, 0] = self.payoff_fn(init_policy_a, init_policy_b)

    def solve(self) -> Tuple[Dict[object, float], Dict[object, float], np.ndarray]:
        for _ in range(self.max_iters):
            mix_a, mix_b = self._solve_meta_game()
            br_a = self.best_response_fn("A", mix_b)
            br_b = self.best_response_fn("B", mix_a)

            added = False
            if br_a not in self.policies_a:
                self._add_policy_a(br_a)
                added = True
            if br_b not in self.policies_b:
                self._add_policy_b(br_b)
                added = True

            if not added:
                break

        mix_a, mix_b = self._solve_meta_game()
        return mix_a, mix_b, self.meta_game.copy()

    def _add_policy_a(self, policy):
        self.policies_a.append(policy)
        new_row = np.array([self.payoff_fn(policy, b) for b in self.policies_b], dtype=np.float32)
        self.meta_game = np.vstack([self.meta_game, new_row[None, :]])

    def _add_policy_b(self, policy):
        self.policies_b.append(policy)
        new_col = np.array([self.payoff_fn(a, policy) for a in self.policies_a], dtype=np.float32)
        self.meta_game = np.hstack([self.meta_game, new_col[:, None]])

    def _solve_meta_game(self) -> Tuple[Dict[object, float], Dict[object, float]]:
        # Simple fictitious play: uniform over current policy set.
        mix_a = {p: 1.0 / len(self.policies_a) for p in self.policies_a}
        mix_b = {p: 1.0 / len(self.policies_b) for p in self.policies_b}
        return mix_a, mix_b
