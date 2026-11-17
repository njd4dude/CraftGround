"""Scripted sanity check for the tree-chopping environment.

Walks straight at the tree in front of spawn and chops its trunk, then resets
and does it again to confirm the arena rebuilds. Saves what the agent sees to
check_frames/ so you can eyeball the arena.

Run this before training: if it doesn't report logs chopped, training can't work.
"""

from pathlib import Path

from PIL import Image

from chop.envs.common import ACTION_NAMES
from chop.envs.arena import ChopTreesEnv

A = {name: i for i, name in enumerate(ACTION_NAMES)}
FRAMES = Path("check_frames")
FRAMES.mkdir(exist_ok=True)

# Spawn faces yaw 0 (+Z), straight at the tree at (0, 5). Walk up, chop the
# log at eye level, then look down/up to reach the rest of the trunk.
SCRIPT = (
    [A["forward"]] * 12
    + [A["attack"]] * 3
    + [A["look_down"]] * 3 + [A["attack"]] * 3
    + [A["look_up"]] * 5 + [A["attack"]] * 3
    + [A["look_up"]] * 1 + [A["attack"]] * 3
    + [A["look_up"]] * 1 + [A["attack"]] * 3
)


def run_episode(env: ChopTreesEnv, episode: int, face_tree: bool) -> int:
    if face_tree:
        # Bypass the random spawn direction so the script knows where the tree is.
        obs, _ = env.env.reset(options={"extra_commands": ["tp @p 0.5 -60 0.5 0 0"]})
        env._steps = 0
        env._logs_at_start = env._logs_mined_total(obs)
        env._logs_chopped = 0
        image = env._image(obs)
    else:
        image, _ = env.reset()
    Image.fromarray(image).save(FRAMES / f"ep{episode}_start.png")

    total = 0.0
    for t, action in enumerate(SCRIPT):
        image, reward, terminated, truncated, info = env.step(action)
        total += reward
        if reward:
            print(f"  step {t:3d} {ACTION_NAMES[action]:<10} reward +{reward:.0f}")
        if terminated or truncated:
            break
    Image.fromarray(image).save(FRAMES / f"ep{episode}_end.png")
    print(f"episode {episode}: logs chopped = {info['logs_chopped']}, total reward = {total:.0f}")
    return info["logs_chopped"]


def main():
    env = ChopTreesEnv(port=8023)
    try:
        results = [run_episode(env, ep, face_tree=True) for ep in range(2)]
    finally:
        env.close()

    if all(r > 0 for r in results):
        print("\nCHECK PASSED: logs give reward and trees regrow after reset.")
    else:
        print(f"\nCHECK FAILED: logs chopped per episode = {results}. See check_frames/.")


if __name__ == "__main__":
    main()
