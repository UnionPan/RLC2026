import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.grid import competitive_fourrooms_env


def run_demo(steps: int = 200, render_mode: str = "human") -> None:
    env = competitive_fourrooms_env(
        width=13,
        height=13,
        view_radius=3,
        max_steps=steps,
        render_mode=render_mode,
    )
    env.reset(seed=0)

    for _ in range(steps):
        agent = env.agent_selection
        action = env.action_space(agent).sample()
        env.step(action)

        if render_mode in {"human", "rgb_array"}:
            env.render()
        else:
            frame = env.render()
            if frame is not None:
                print(frame)
                time.sleep(0.05)

        if any(env.terminations.values()) or any(env.truncations.values()):
            break

    env.close()


if __name__ == "__main__":
    mode = "human"
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    run_demo(render_mode=mode)
