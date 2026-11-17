"""Find the fixed forest spots evaluate.py uses in the natural world.

    python make_eval_spots.py                  # 100 spots -> results/eval_spots_natural.json
    python make_eval_spots.py --count 200 --n-envs 2
    python make_eval_spots.py --held-out       # spots in the held-out world (natural_env.HELD_OUT_SEED)
                                               # -> results/eval_spots_natural_seed<seed>.json

Why: a random-spawn evaluation tests every model on 100 different forests, and some
forests are much better than others, so two 100-episode means differ by +/-3-4 logs
from spawn luck alone. With one saved set of spots every model plays the same forests.

How: each instance drops the player at random points anywhere in the world and keeps
the landings that pass the same forest test training resets use (natural_env.FOREST_CHECK,
on dirt/grass). A kept spot must be MIN_SPACING blocks from every other one: an episode
lasts 50 s, so the agent walks at most ~215 blocks and never reaches another spot's trees
(each Minecraft instance also has its own copy of the world). Each spot gets a fixed
facing, drawn from a seeded random generator. The spots themselves can't be reproduced
(Minecraft picks spreadplayers' landings), which is why they're saved.

Writing the file again replaces the evaluation set, and scores on different sets
aren't comparable, so the script refuses to overwrite it unless given --force.
"""

import argparse
import json
import math
import random
import time
from datetime import datetime

from chop.envs import natural as natural_env
from chop.infra.vec import DEFAULT_N_ENVS, close_vec_env, game_folder, instance_ports, launch_vec_env

MIN_SPACING = 300  # blocks between any two spots
YAW_SEED = 20261116


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS["natural"])
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--force", action="store_true", help="replace an existing spot file")
    parser.add_argument("--seed", default=natural_env.SEED,
                        help=f"world seed (default: the training world, {natural_env.SEED})")
    parser.add_argument("--held-out", action="store_true",
                        help=f"use the held-out world, seed {natural_env.HELD_OUT_SEED} (same as --seed)")
    args = parser.parse_args()
    if args.held_out:
        args.seed = natural_env.HELD_OUT_SEED

    out = natural_env.eval_spots_path(args.seed)
    if out.exists() and not args.force:
        parser.error(f"{out} exists; scores on a new set aren't comparable with the old one (use --force)")

    ports = instance_ports(args.port, args.n_envs)
    env = launch_vec_env([lambda p=p: natural_env.make_env(port=p, env_path=game_folder(p), seed=args.seed)
                          for p in ports], ports)
    rng = random.Random(YAW_SEED)
    spots: list[dict] = []
    tries = rejected_close = 0
    t0 = time.perf_counter()
    try:
        env.reset()
        while len(spots) < args.count:
            for found in env.env_method("find_far_spot"):
                tries += 1
                if found is None or len(spots) >= args.count:
                    continue
                if any(math.dist((found["x"], found["z"]), (s["x"], s["z"])) < MIN_SPACING for s in spots):
                    rejected_close += 1
                    continue
                spots.append({**found, "yaw": rng.choice(range(-180, 180, 15))})
                print(f"spot {len(spots):3d}/{args.count}: {found}  ({tries} tries, "
                      f"{time.perf_counter() - t0:.0f} s)", flush=True)
    finally:
        close_vec_env(env, ports)

    out.write_text(json.dumps({
        "created": datetime.now().isoformat(timespec="seconds"),
        "world_seed": args.seed,
        "forest_check": natural_env.FOREST_CHECK,
        "min_spacing": MIN_SPACING,
        "yaw_seed": YAW_SEED,
        "tries": tries,
        "spots": spots,
    }, indent=1), encoding="utf-8")
    print(f"{len(spots)} spots from {tries} tries ({rejected_close} too close to another) -> {out}")


if __name__ == "__main__":
    main()
