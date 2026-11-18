"""Check that an arena reset repairs dug-up ground.

Episode 1 digs a 3-deep pit and puts the player over it: they should fall in (proves the pit is
real). Episode 2 is a normal reset with the player put on the same spot: the ground should be
back, so they stand at ground level. Before the fix, holes stayed for the rest of a run.

    run.bat tools/check_ground.py
"""

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.arena import GROUND_Y, ChopTreesEnv
from chop.envs.common import ACTION_NAMES

SPOT = (8, 8)  # inside the walls, away from the trees
PIT = f"fill {SPOT[0]} {GROUND_Y - 3} {SPOT[1]} {SPOT[0]} {GROUND_Y - 1} {SPOT[1]} air"
ON_SPOT = f"tp @p {SPOT[0] + 0.5} {GROUND_Y} {SPOT[1] + 0.5}"


def player_y_after_reset(env: ChopTreesEnv, commands: list[str]) -> float:
    env.reset(options={"extra_commands": commands})
    obs = None
    step = env.env.step

    def watched(action):
        nonlocal obs
        out = step(action)
        obs = out[0]
        return out
    env.env.step = watched
    try:
        for _ in range(20):  # time to fall
            env.step(ACTION_NAMES.index("noop"))
    finally:
        env.env.step = step
    return obs["full"].y


def main():
    env = ChopTreesEnv(port=8023)
    try:
        env.reset()  # the first reset after launch ignores teleports
        dug = player_y_after_reset(env, [PIT, ON_SPOT])
        after = player_y_after_reset(env, [ON_SPOT])
    finally:
        env.close()
    print(f"episode 1, pit dug under the player: y = {dug:.2f} (ground level is {GROUND_Y})")
    print(f"episode 2, same spot after a reset:  y = {after:.2f}")
    fell = dug < GROUND_Y - 1
    repaired = abs(after - GROUND_Y) < 0.1
    if not fell:
        print("CHECK INVALID: the player didn't fall into the pit, so the test didn't dig one")
    print("\nCHECK PASSED: the reset repaired the ground" if fell and repaired
          else "\nCHECK FAILED" if fell else "")


if __name__ == "__main__":
    main()
