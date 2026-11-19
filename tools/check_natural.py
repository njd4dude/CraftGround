"""Sanity check for chop/envs/natural.py: reset a few times and save what the agent sees.

    run.bat tools/check_natural.py
    run.bat tools/check_natural.py --spawn wooded --episodes 10   # any biome with trees

Each reset should put the agent on the ground in a forest (or, with --spawn wooded, any
biome with trees: the biome each spawn passed is printed), at a new spot.
Frames go to check_frames/natural_epN_*.png. Prints where each episode started
and how many logs a few seconds of random play chopped.
"""

import argparse
import time
from pathlib import Path

from PIL import Image

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs import natural as natural_env
from chop.envs.natural import NaturalChopTreesEnv

natural_env.DEBUG = True

FRAMES = Path("check_frames")
FRAMES.mkdir(exist_ok=True)
RANDOM_STEPS = 200
# Biomes to name in the printout (a passing spawn is in one of them; tags name the group).
NAMED_BIOMES = ["snowy_taiga", "grove", "taiga", "old_growth_pine_taiga", "old_growth_spruce_taiga",
                "forest", "flower_forest", "birch_forest", "old_growth_birch_forest", "dark_forest",
                *[b.removeprefix("minecraft:") for b in natural_env.WOODED_BIOMES if not b.startswith("#")],
                "jungle", "sparse_jungle", "bamboo_jungle", "savanna", "savanna_plateau", "windswept_savanna"]


def biome_here(env: NaturalChopTreesEnv) -> str:
    """Which NAMED_BIOMES the player stands in (one command each; only a pass replies)."""
    for b in NAMED_BIOMES:
        _, reply = env._command(f"execute as @p at @s if biome ~ ~ ~ minecraft:{b}", ("Test passed",),
                                timeout=natural_env.CHECK_TIMEOUT_S, quiet=True)
        if reply:
            return b
    return "?"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8023,
                        help="use a port no other script is on; two scripts on one port reset each other's game")
    parser.add_argument("--spawn", choices=natural_env.SPAWNS, default="forest")
    parser.add_argument("--episodes", type=int, default=6)
    args = parser.parse_args()
    env = NaturalChopTreesEnv(port=args.port, spawn=args.spawn)
    try:
        for ep in range(args.episodes):
            t0 = time.monotonic()
            image, reset_info = env.reset()
            reset_s = time.monotonic() - t0
            biome = biome_here(env)
            x, y, z = env.spot
            Image.fromarray(image).save(FRAMES / f"natural_ep{ep}_start.png")
            logs = 0
            for _ in range(RANDOM_STEPS):
                image, _, terminated, truncated, info = env.step(env.action_space.sample())
                logs = info["logs_chopped"]
                if terminated or truncated:
                    break
            Image.fromarray(image).save(FRAMES / f"natural_ep{ep}_end.png")
            print(f"episode {ep}: {'found' if env.found else 'NOT FOUND'} {biome} at ({x:.0f}, {y:.0f}, {z:.0f}), "
                  f"{logs} logs in {RANDOM_STEPS} random steps, reset took {reset_s:.1f}s "
                  f"({reset_info['near_tries']} near + {reset_info['far_tries']} far tries)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
