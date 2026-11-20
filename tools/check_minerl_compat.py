"""Sanity checks for the arena's MineRL mode (chop/envs/minerl.py, make_env(actions="minerl")).

    run.bat tools/check_minerl_compat.py

Saves the full 320x180 frame and the 64x64 network input to check_frames/, checks that
each camera action turns the view by the right amount and direction, and that walking
into a tree with attack held chops it with the iron axe.
"""

from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.arena import GROUND_Y, make_env
from chop.envs.minerl import PITCH_DEGREES, YAW_DEGREES

OUT = Path("check_frames")
STAND, NO_YAW, NO_PITCH = 0, YAW_DEGREES.index(0), PITCH_DEGREES.index(0)


def main():
    OUT.mkdir(exist_ok=True)
    env = make_env(port=8123, actions="minerl")
    # Keep the raw observation message around so we can read yaw/pitch.
    inner_step, inner_reset = env.env.step, env.env.reset

    tick_heights: list[float] = []  # the player's y after every game tick (for the jump check)

    def step(a):
        out = inner_step(a)
        env._last_full = out[0]["full"]
        tick_heights.append(env._last_full.y)
        return out

    def reset(**kw):
        out = inner_reset(**kw)
        env._last_full = out[0]["full"]
        return out

    env.env.step, env.env.reset = step, reset

    obs, _ = env.reset()
    for _ in range(10):  # let the arena finish building
        obs, *_ = env.step([STAND, NO_YAW, NO_PITCH])
    print("obs", obs.shape, obs.dtype, "full frame", env.last_pov.shape, env.last_pov.dtype)
    Image.fromarray(env.last_pov).save(OUT / "minerl_full_320x180.png")
    Image.fromarray(obs).save(OUT / "minerl_obs_64x64.png")
    Image.fromarray(obs).resize((256, 256), Image.NEAREST).save(OUT / "minerl_obs_64x64_x4.png")

    # The reported yaw/pitch lag one tick behind, so each turn is measured across the
    # action plus one no-op decision. Minecraft turns in whole mouse steps of 0.15
    # degrees per tick, so a 4-tick decision can fall short by up to 4 * 0.15 = 0.6.
    ok = True
    noop = [STAND, NO_YAW, NO_PITCH]
    for i, deg in enumerate(YAW_DEGREES):
        y0 = env._last_full.yaw
        env.step([STAND, i, NO_PITCH])
        env.step(noop)
        d = ((env._last_full.yaw - y0 + 180) % 360) - 180
        good = abs(d - deg) <= 0.65
        ok &= good
        print(f"yaw   {deg:+4d} -> turned {d:+7.2f}  {'ok' if good else 'MISMATCH'}")
    for i, deg in enumerate(PITCH_DEGREES):
        p0 = env._last_full.pitch
        env.step([STAND, NO_YAW, i])
        env.step(noop)
        d = env._last_full.pitch - p0
        good = abs(d - deg) <= 0.65
        ok &= good
        print(f"pitch {deg:+4d} -> moved  {d:+7.2f}  {'ok' if good else 'MISMATCH'} (from {p0:+.1f})")
        env.step([STAND, NO_YAW, PITCH_DEGREES.index(-deg)])  # back to roughly level
        env.step(noop)

    # Face the tree at (5, 0) (yaw -90 = east) and walk into it with attack held.
    # A few idle decisions first so reset()'s own random-facing tp has run.
    env.reset()
    for _ in range(3):
        env.step(noop)
    env.unwrapped.add_commands(["tp @p 0.5 -60 0.5 -90 0"])
    total = 0.0
    for t in range(10):
        _, r, term, trunc, info = env.step([1, NO_YAW, NO_PITCH])
        total += r
    good = total >= 1
    ok &= good
    print(f"walk into tree: {info['logs_chopped']} logs  {'ok' if good else 'NO LOG CHOPPED'}")
    Image.fromarray(env.last_pov).save(OUT / "minerl_full_after_walk.png")

    # Move choices 1 (forward) and 2 (forward+jump) in open ground: x=-9.5 facing south (yaw 0 = +z)
    # has no tree ahead for ~12 blocks. Forward should walk on the ground; forward+jump should also
    # leave it (a jump is ~1.25 blocks high). Height is recorded every tick, not just per decision.
    for move, name, should_jump in [(1, "forward", False), (2, "forward+jump", True)]:
        env.unwrapped.add_commands(["tp @p -9.5 -60 0.5 0 0"])
        for _ in range(3):
            env.step(noop)
        z0 = env._last_full.z
        tick_heights.clear()
        for _ in range(4):  # 4 decisions = 16 ticks, about one jump
            env.step([move, NO_YAW, NO_PITCH])
        rise = max(tick_heights) - GROUND_Y
        walked = env._last_full.z - z0
        good = walked > 0.5 and (rise > 0.5 if should_jump else rise < 0.1)
        ok &= good
        print(f"{name:13s}: walked {walked:+.2f} blocks, highest {rise:.2f} above ground  "
              f"{'ok' if good else 'MISMATCH'}")
    env.close()
    print("ALL CHECKS OK" if ok else "CHECKS FAILED")


if __name__ == "__main__":
    main()
