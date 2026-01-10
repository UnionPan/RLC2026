import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.grid.pursuit_evasion import raw_env


def test_observation_shape_and_visibility():
    env = raw_env(width=5, height=5, n_pursuers=1, view_radius=1, wall_density=0.0)
    env.reset(seed=0)

    obs = env.observe(env.agent_selection)
    assert obs.shape == (3, 3, 4)
    assert obs.dtype == np.float32
    assert obs[1, 1, 3] == 1.0

    env.close()


def test_capture_terminates_and_rewards():
    env = raw_env(width=5, height=5, n_pursuers=1, view_radius=1, wall_density=0.0)
    env.reset(seed=1)

    env._positions["pursuer_0"] = (2, 2)
    env._positions["evader_0"] = (2, 3)
    env.agent_selection = "pursuer_0"

    env.step(4)

    assert all(env.terminations.values())
    assert env.rewards["pursuer_0"] == env.config.capture_reward
    assert env.rewards["evader_0"] == -env.config.capture_reward

    env.close()


def test_truncation_after_max_steps():
    env = raw_env(width=5, height=5, n_pursuers=1, view_radius=1, wall_density=0.0, max_steps=1)
    env.reset(seed=2)

    for _ in range(len(env.agents)):
        env.step(0)

    assert all(env.truncations.values())
    assert not any(env.terminations.values())

    env.close()
