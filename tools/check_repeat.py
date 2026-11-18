"""Check --action-repeat: each agent step's own action should show up in that step's observation.

With 1 tick per step CraftGround shows actions 1-3 steps late (tools/check_timing.py). With 4
ticks per step this checks that a turn, a walk and a chop each show up on the step they're sent,
that a turn still turns by the amount chosen (not 4x), and that rewards add up over the ticks.

    run.bat tools/check_repeat.py
"""

import numpy as np

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.arena import GROUND_Y, ChopTreesEnv
from chop.envs.common import ACTION_NAMES, LOOK_DEGREES, MOVES, TURN_DEGREES

REPEAT = 4
OPEN = f"tp @p 0.5 {GROUND_Y} 0.5 0 0"
AT_TREE = f"tp @p 0.5 {GROUND_Y} 3.5 0 0"   # crosshair on the trunk at (0, 5), within reach
FAR = f"tp @p 0.5 {GROUND_Y} -8.5 180 0"      # facing the wall, nothing to chop in reach
AT_TREE_LOOKING_AWAY = f"tp @p 0.5 {GROUND_Y} 3.5 -45 0"  # at the trunk, turned 45 degrees off it


def multi(move: str = "none", turn: float = 0, look: float = 0) -> list[int]:
    return [MOVES.index(move), TURN_DEGREES.index(turn), LOOK_DEGREES.index(look)]


class Watched:
    """Remembers the last raw observation the wrapped env got, to read yaw / position."""

    def __init__(self, env: ChopTreesEnv):
        self.env, self.last = env, None
        step = env.env.step

        def watched_step(action):
            out = step(action)
            self.last = out[0]
            return out
        env.env.step = watched_step

    def start(self, tp: str, settle: int = 2):
        image, _ = self.env.reset(options={"extra_commands": [tp]})  # the real reset, as in training
        self.settle_reward = 0.0
        for _ in range(settle):  # settle after the teleport
            image, reward, *_ = self.env.step(self.idle)
            self.settle_reward += reward
        return image


def run(w: Watched, name: str, tp: str, action, check, settle: int = 2) -> bool:
    image = w.start(tp, settle)
    f0 = w.last["full"]
    new_image, reward, *_ = w.env.step(action)
    f = w.last["full"]
    img = np.abs(new_image.astype(np.int16) - image.astype(np.int16)).mean()
    measured = {"yaw": f.yaw - f0.yaw, "pitch": f.pitch - f0.pitch,
                "moved": float(np.hypot(f.x - f0.x, f.z - f0.z)), "reward": reward, "image": float(img),
                "settle_reward": w.settle_reward}
    ok, why = check(measured)
    print(f"  {'ok  ' if ok else 'FAIL'} {name:<34} yaw {measured['yaw']:+6.1f}  pitch {measured['pitch']:+6.1f}  "
          f"moved {measured['moved']:.2f}  reward {reward:.0f} (settling {w.settle_reward:.0f})  "
          f"image change {img:5.2f}   {why}")
    return ok


def turn_onto_trunk(w: Watched) -> bool:
    """Multi holds attack all the time, so what causes a chop is aiming at a log. Stand at the trunk
    turned away (attack swinging at nothing), turn onto it, and see which step the log lands in.
    Run in more than one episode: attack held across a reset used to stop working (arena.reset)."""
    w.start(AT_TREE_LOOKING_AWAY)
    rewards = [w.env.step(multi(turn=45))[1]] + [w.env.step(multi())[1] for _ in range(2)]
    landed = next((i for i, r in enumerate(rewards) if r), None)
    ok = w.settle_reward == 0 and landed is not None
    where = "never" if landed is None else ("the same step" if landed == 0 else f"{landed} step(s) later")
    print(f"  {'ok  ' if ok else 'FAIL'} {'turn onto the trunk, attack held':<34} "
          f"rewards per step {[int(r) for r in rewards]} (settling {w.settle_reward:.0f}): log lands {where}")
    return ok


def main():
    results = []

    env = ChopTreesEnv(port=8023, actions="multi", action_repeat=REPEAT)
    w = Watched(env)
    w.idle = multi()
    env.reset()  # the very first reset after launch ignores teleports, so get it out of the way
    print(f"multi actions, {REPEAT} ticks per step")
    try:
        results.append(run(w, "turn right 10 (same step, 10 not 40)", OPEN, multi(turn=10),
                           lambda m: (8 <= m["yaw"] <= 11 and m["image"] > 1, "expect yaw about +10 and a new image")))
        results.append(run(w, "look down 10", OPEN, multi(look=10),
                           lambda m: (8 <= m["pitch"] <= 11 and m["image"] > 1, "expect pitch about +10")))
        results.append(run(w, "walk forward", OPEN, multi("forward"),
                           lambda m: (m["moved"] > 0.3 and m["image"] > 1, "expect walking speed and a new image")))
        results.append(turn_onto_trunk(w))
        results.append(turn_onto_trunk(w))  # next episode: attack must still work after a reset
    finally:
        env.close()

    env = ChopTreesEnv(port=8023, actions="single", action_repeat=REPEAT)
    w = Watched(env)
    w.idle = ACTION_NAMES.index("noop")
    env.reset()
    print(f"\nsingle actions, {REPEAT} ticks per step")
    try:
        results.append(run(w, "turn_right (15, not 60)", OPEN, ACTION_NAMES.index("turn_right"),
                           lambda m: (13 <= m["yaw"] <= 16 and m["image"] > 1, "expect yaw about +15")))
        results.append(run(w, "attack at the trunk", AT_TREE, ACTION_NAMES.index("attack"),
                           lambda m: (m["reward"] >= 1, "expect the log in this same step")))
        results.append(run(w, "attack with nothing in reach", FAR, ACTION_NAMES.index("attack"),
                           lambda m: (m["reward"] == 0, "expect no reward")))
    finally:
        env.close()

    print("\nCHECK PASSED" if all(results) else "\nCHECK FAILED")


if __name__ == "__main__":
    main()
