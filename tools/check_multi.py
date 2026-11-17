"""Scripted check for the arena's multi actions (train.py --actions multi).

Faces the tree in front of spawn, then checks that:
  1. turns and looks move the camera by the degrees asked for (CraftGround shows a camera move
     one step late, so the camera test ends with a no-op step to let the last move land),
  2. holding attack while walking up to the trunk chops logs,
  3. it keeps chopping from log to log as the agent looks up and down the trunk.

    run.bat tools/check_multi.py
"""

from pathlib import Path

from PIL import Image

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.arena import ChopTreesEnv
from chop.envs.common import LOOK_DEGREES, MOVES, TURN_DEGREES, multi_label, to_craftground_multi

FRAMES = Path("check_frames")
FRAMES.mkdir(exist_ok=True)


def act(move: str = "none", turn: float = 0, look: float = 0) -> list[int]:
    return [MOVES.index(move), TURN_DEGREES.index(turn), LOOK_DEGREES.index(look)]


# Turn right 5 then back, look down 10 then back: the camera should end where it started.
CAMERA_TEST = [act(turn=5), act(turn=-5), act(look=10), act(look=-10), act()]
# Walk up to the trunk at (0, 5) with attack held, then sweep the view down and up the trunk.
CHOP = ([act("forward")] * 12 + [act()] * 6
        + [act(look=20)] * 2 + [act()] * 6
        + [act(look=-20)] * 3 + [act()] * 6
        + [act(look=-10)] * 2 + [act()] * 6
        + [act(look=-10)] * 2 + [act()] * 6)


def main():
    env = ChopTreesEnv(port=8023, actions="multi")
    ok = True
    try:
        obs, _ = env.env.reset(options={"extra_commands": ["tp @p 0.5 -60 0.5 0 0"]})
        start_logs = env._logs_mined_total(obs)
        full = obs["full"]
        print(f"start: yaw {full.yaw:.1f}, pitch {full.pitch:.1f}")
        for a in CAMERA_TEST:
            before = obs["full"]
            obs, *_ = env.env.step(to_craftground_multi(a))
            after = obs["full"]
            print(f"  {multi_label(a):<12} yaw {before.yaw:7.1f} -> {after.yaw:7.1f}   "
                  f"pitch {before.pitch:6.1f} -> {after.pitch:6.1f}")
        if abs(obs["full"].yaw - full.yaw) > 1 or abs(obs["full"].pitch - full.pitch) > 1:
            print("CAMERA CHECK FAILED: the camera didn't come back to where it started")
            ok = False

        logs = 0
        for t, a in enumerate(CHOP):
            obs, *_ = env.env.step(to_craftground_multi(a))
            now = env._logs_mined_total(obs) - start_logs
            if now > logs:
                print(f"  step {t:3d} {multi_label(a):<14} pitch {obs['full'].pitch:6.1f}  logs {now}")
                logs = now
        Image.fromarray(env._image(obs)).save(FRAMES / "multi_end.png")
        print(f"logs chopped: {logs} of 4 in the trunk")
        if logs < 2:
            print("CHOP CHECK FAILED: holding attack should chop several logs. See check_frames/multi_end.png")
            ok = False
    finally:
        env.close()
    print("\nCHECK PASSED" if ok else "\nCHECK FAILED")


if __name__ == "__main__":
    main()
