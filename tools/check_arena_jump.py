"""Scripted check for the arena's jump option (train.py --jump).

For multi actions and for single actions, checks the action space matches the natural world's,
that a jump lifts the player off the ground, and that jumping at the wall can't get it out.

    run.bat tools/check_arena_jump.py
"""

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.arena import ARENA_RADIUS, GROUND_Y, ChopTreesEnv
from chop.envs.common import (ACTION_NAMES_JUMP, LOOK_DEGREES, MOVES, MULTI_NVEC_JUMP, TURN_DEGREES)

ok = True


def check(cond: bool, what: str):
    global ok
    ok &= cond
    print(("  ok   " if cond else "  FAIL ") + what)


def run(actions: str, jump_action, forward_jump_action, noop_action):
    print(f"{actions} actions, jump on")
    env = ChopTreesEnv(port=8023, actions=actions, jump=True)
    try:
        space = (list(env.action_space.nvec) if actions == "multi" else env.action_space.n)
        want = MULTI_NVEC_JUMP if actions == "multi" else len(ACTION_NAMES_JUMP)
        check(space == want, f"action space {space} (natural world: {want})")
        env.reset(options={"extra_commands": [f"tp @p 0.5 {GROUND_Y} 0.5 0 0"]})
        for _ in range(3):  # settle, as tools/check_natural_multi.py does
            obs, *_ = env.env.step(env._game_action(noop_action, camera=True))
        heights = []
        for a in [jump_action] + [noop_action] * 6:
            obs, *_ = env.env.step(env._game_action(a, camera=True))
            heights.append(obs["full"].y)
        check(max(heights) > GROUND_Y + 0.5, f"jump lifts the player: highest y {max(heights):.2f}")
        # Face the north wall (yaw 180 = -z) from just inside it and jump into it for a while.
        env.reset(options={"extra_commands": [f"tp @p 0.5 {GROUND_Y} {-(ARENA_RADIUS - 2) + 0.5} 180 0"]})
        for _ in range(60):
            obs, *_ = env.env.step(env._game_action(forward_jump_action, camera=False))
        z = obs["full"].z
        check(z > -ARENA_RADIUS, f"still inside after 60 forward-jumps into the wall: z {z:.2f}")
    finally:
        env.close()


def main():
    multi = lambda move, jump: [MOVES.index(move), TURN_DEGREES.index(0), LOOK_DEGREES.index(0), jump]  # noqa: E731
    run("multi", multi("none", 1), multi("forward", 1), multi("none", 0))
    run("single", ACTION_NAMES_JUMP.index("jump"), ACTION_NAMES_JUMP.index("forward_jump"),
        ACTION_NAMES_JUMP.index("noop"))
    print("ALL OK" if ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
