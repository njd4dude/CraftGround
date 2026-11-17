"""Sanity check for chop/envs/natural.py: reset a few times and save what the agent sees.

    run.bat check_natural.py

Each reset should put the agent on the ground in a forest, at a new spot.
Frames go to check_frames/natural_epN_*.png. Prints where each episode started
and how many logs a few seconds of random play chopped.
"""

import argparse
import time
from pathlib import Path

from PIL import Image

from chop.envs import natural as natural_env
from chop.envs.natural import NaturalChopTreesEnv

natural_env.DEBUG = True

FRAMES = Path("check_frames")
FRAMES.mkdir(exist_ok=True)
EPISODES = 6
RANDOM_STEPS = 200


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8023,
                        help="use a port no other script is on; two scripts on one port reset each other's game")
    env = NaturalChopTreesEnv(port=parser.parse_args().port)
    try:
        for ep in range(EPISODES):
            t0 = time.monotonic()
            image, _ = env.reset()
            reset_s = time.monotonic() - t0
            x, y, z = env.spot
            Image.fromarray(image).save(FRAMES / f"natural_ep{ep}_start.png")
            logs = 0
            for _ in range(RANDOM_STEPS):
                image, _, terminated, truncated, info = env.step(env.action_space.sample())
                logs = info["logs_chopped"]
                if terminated or truncated:
                    break
            Image.fromarray(image).save(FRAMES / f"natural_ep{ep}_end.png")
            print(f"episode {ep}: {'forest' if env.found else 'NO FOREST'} at ({x:.0f}, {y:.0f}, {z:.0f}), "
                  f"{logs} logs in {RANDOM_STEPS} random steps, reset took {reset_s:.1f}s")
    finally:
        env.close()


if __name__ == "__main__":
    main()
