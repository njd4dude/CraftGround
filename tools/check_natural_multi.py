"""Scripted check for the natural world's multi actions (train.py --env natural --actions multi).

Plays two episodes at real forest spots and checks that:
  1. turns and looks move the camera by the degrees asked for,
  2. jump lifts the player off the ground,
  3. forward walks,
  4. attack is held in a second episode too (attack only presses on off -> on, so a reset could
     leave it dead): looking straight down with attack held digs out the block underfoot, so the
     player drops.

    run.bat tools/check_natural_multi.py
"""

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.common import LOOK_DEGREES, MOVES, MULTI_NVEC_JUMP, TURN_DEGREES
from chop.envs.natural import NaturalChopTreesEnv


def act(move: str = "none", turn: float = 0, look: float = 0, jump: bool = False) -> list[int]:
    return [MOVES.index(move), TURN_DEGREES.index(turn), LOOK_DEGREES.index(look), int(jump)]


class Watched:
    """Runs env.step and keeps the last raw CraftGround observation (position, camera)."""

    def __init__(self, env: NaturalChopTreesEnv):
        self.env, self.full = env, None
        inner = env.env.step

        def step(action):
            out = inner(action)
            self.full = out[0]["full"]
            return out
        env.env.step = step

    def step(self, a, n: int = 1):
        for _ in range(n):
            self.env.step(a)
        return self.full


def main():
    env = NaturalChopTreesEnv(port=8023, actions="multi")
    w = Watched(env)
    ok = True

    def check(passed: bool, what: str):
        nonlocal ok
        print(("  ok    " if passed else "  FAIL  ") + what)
        ok = ok and passed

    try:
        check(list(env.action_space.nvec) == MULTI_NVEC_JUMP, f"action space {list(env.action_space.nvec)}")
        env.reset()
        start = w.step(act(), 3)  # settle

        f = w.step(act(turn=20))
        f = w.step(act())          # camera moves show one step late
        d_yaw = (f.yaw - start.yaw + 180) % 360 - 180
        check(abs(d_yaw - 20) < 1, f"turn +20: yaw moved {d_yaw:+.1f}")
        p0 = f.pitch
        f = w.step(act(look=10))
        f = w.step(act())
        check(abs((f.pitch - p0) - 10) < 1, f"look +10: pitch moved {f.pitch - p0:+.1f}")
        w.step(act(look=-10), 1)
        w.step(act(), 2)

        y0 = w.full.y
        peak = y0
        w.step(act(jump=True))
        for _ in range(6):
            peak = max(peak, w.step(act()).y)
        check(peak - y0 > 0.5, f"jump: rose {peak - y0:.2f} blocks")

        # Forest spots often have a trunk right in front, so try four directions.
        moves = []
        for _ in range(4):
            x0, z0 = w.full.x, w.full.z
            f = w.step(act("forward"), 10)
            moves.append(((f.x - x0) ** 2 + (f.z - z0) ** 2) ** 0.5)
            w.step(act(turn=45), 2)
            w.step(act(), 2)
        check(max(moves) > 1, "forward 10 ticks, 4 directions: moved "
              + ", ".join(f"{m:.2f}" for m in moves) + " blocks")

        # Episode 2: attack must still work after a reset.
        env.reset()
        w.step(act(), 3)
        for _ in range(5):              # look straight down (pitch +90; it clamps at 90)
            w.step(act(look=20))
        y_before = w.step(act()).y
        y_after = w.step(act(), 60).y   # attack is held the whole time
        check(y_after < y_before - 0.5,
              f"episode 2, attack held looking down for 60 ticks: y {y_before:.2f} -> {y_after:.2f}")
    finally:
        env.close()
    print("\nCHECK PASSED" if ok else "\nCHECK FAILED")


if __name__ == "__main__":
    main()
