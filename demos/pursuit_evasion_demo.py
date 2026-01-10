import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.grid.pursuit_evasion import env


def run_demo(steps: int = 200, render_mode: str = "human") -> None:
    pe = env(
        width=11,
        height=11,
        n_pursuers=1,
        view_radius=2,
        wall_density=0.15,
        max_steps=steps,
        render_mode=render_mode,
    )
    pe.reset(seed=0)

    for _ in range(steps):
        agent = pe.agent_selection
        action = pe.action_space(agent).sample()
        pe.step(action)

        if render_mode in {"human", "rgb_array"}:
            pe.render()
        else:
            frame = pe.render()
            if frame is not None:
                print(frame)
                time.sleep(0.05)

        if any(pe.terminations.values()) or any(pe.truncations.values()):
            break

    pe.close()


if __name__ == "__main__":
    mode = "human"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    run_demo(render_mode=mode)
