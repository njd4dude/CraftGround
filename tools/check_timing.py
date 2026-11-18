"""When does an action show up in what step() returns?

For each kind of action (camera turn, camera look, walk forward, attack) this sends the action on
ONE step, surrounded by no-ops, and prints every step's yaw / pitch / position / logs mined and how
much the image changed from the previous step. If the change lands on the step the action was
sent, the observation is up to date; if it lands on the step after, it lags one step.

Runs both action formats: the single actions (CraftGround V1, 15-degree camera steps) and the
multi actions (V2, exact degrees).

    run.bat tools/check_timing.py
"""

import numpy as np
from craftground.environment.action_space import no_op, no_op_v2

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs.arena import GROUND_Y, ChopTreesEnv

BEFORE, AFTER = 4, 5  # no-op steps before and after the action under test

# Where each test starts. Trunk at block (0, 5) spans z 5..6; standing at z 3.5 facing +Z (yaw 0)
# the crosshair is on its second log, within reach.
OPEN = f"tp @p 0.5 {GROUND_Y} 0.5 0 0"
AT_TREE = f"tp @p 0.5 {GROUND_Y} 3.5 0 0"


def v1(**kw) -> list[int]:
    act = no_op()
    if "yaw" in kw:
        act[4] = 12 + kw["yaw"]      # +1 = 15 degrees right
    if "pitch" in kw:
        act[3] = 12 + kw["pitch"]    # +1 = 15 degrees down
    if kw.get("forward"):
        act[0] = 1
    if kw.get("attack"):
        act[5] = 3
    return act


def v2(**kw) -> dict:
    act = no_op_v2()
    act["camera_yaw"] = float(kw.get("yaw", 0))
    act["camera_pitch"] = float(kw.get("pitch", 0))
    act["forward"] = bool(kw.get("forward"))
    act["attack"] = bool(kw.get("attack"))
    return act


def run_test(env: ChopTreesEnv, name: str, tp: str, idle, action) -> dict:
    obs, _ = env.env.reset(options={"extra_commands": [tp]})
    for _ in range(3):  # let the teleport and the first frames settle
        obs, *_ = env.env.step(idle)
    logs0 = env._logs_mined_total(obs)
    prev = obs
    print(f"\n{name}")
    print("  step  sent      yaw    pitch       x       z  logs  image change")
    first = {}
    for t in range(BEFORE + 1 + AFTER):
        sent = action if t == BEFORE else idle
        obs, *_ = env.env.step(sent)
        f, p = obs["full"], prev["full"]
        img = np.abs(obs["pov"].astype(np.int16) - prev["pov"].astype(np.int16)).mean()
        logs = env._logs_mined_total(obs) - logs0
        changes = {
            "camera": abs(f.yaw - p.yaw) > 0.1 or abs(f.pitch - p.pitch) > 0.1,
            "position": abs(f.x - p.x) > 1e-3 or abs(f.z - p.z) > 1e-3,
            "logs": logs > env._logs_mined_total(prev) - logs0,
            "image": img > 1.0,
        }
        for k, v in changes.items():
            if v and k not in first:
                first[k] = t - BEFORE  # 0 = same step as the action, 1 = one step later
        tag = "ACTION" if t == BEFORE else ""
        print(f"  {t - BEFORE:+4d}  {tag:<6} {f.yaw:7.1f} {f.pitch:7.1f} {f.x:7.2f} {f.z:7.2f} {logs:5d} {img:12.2f}")
        prev = obs
    print("  first change (steps after the action): "
          + ", ".join(f"{k} {'+' if v >= 0 else ''}{v}" for k, v in first.items()) if first else "  no change")
    return first


def main():
    results = {}
    for fmt, make_act, actions in (
        ("single (V1)", v1, "single"),
        ("multi (V2)", v2, "multi"),
    ):
        env = ChopTreesEnv(port=8023, actions=actions)
        try:
            idle = make_act()
            turn = make_act(yaw=1) if fmt.startswith("single") else make_act(yaw=15)
            look = make_act(pitch=1) if fmt.startswith("single") else make_act(pitch=15)
            results[fmt] = {
                "turn": run_test(env, f"[{fmt}] turn right 15 degrees", OPEN, idle, turn),
                "look": run_test(env, f"[{fmt}] look down 15 degrees", OPEN, idle, look),
                "walk": run_test(env, f"[{fmt}] walk forward one step", OPEN, idle, make_act(forward=True)),
                "attack": run_test(env, f"[{fmt}] attack the log in front", AT_TREE, idle, make_act(attack=True)),
            }
        finally:
            env.close()

    print("\nSUMMARY: steps after the action until it shows up (0 = same step, 1 = one step late)")
    for fmt, tests in results.items():
        for test, first in tests.items():
            print(f"  {fmt:<12} {test:<7} " + ", ".join(f"{k} {v:+d}" for k, v in first.items()))


if __name__ == "__main__":
    main()
