"""How much memory natural-world instances use over time, at a given render distance.

    python tools/memtest.py --render-distance 4 --minutes 20      # 4 instances, random actions
    python tools/memtest.py --render-distance 4 --jvm-max-memory 2G

Plays random actions (episodes and resets as in training, no model) and every minute
records each instance's memory (all of its processes: cmd, Gradle, Minecraft) and the
system's free RAM, to results/memtest.csv. Instances grow as episodes hop to new
forests (new terrain), so compare runs of the same length.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs import natural as natural_env
from chop.infra.instances import instance_processes
from chop.infra.vec import close_vec_env, game_folder, instance_ports, launch_vec_env

CSV = Path("results/memtest.csv")
FIELDS = ["started", "render_distance", "jvm_max_memory", "n_envs", "minute", "steps", "free_gb", "instances_gb", "per_instance_gb"]


def instance_gb(port: int) -> float:
    total = 0
    for p in instance_processes(port):
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-distance", type=int, default=natural_env.RENDER_DISTANCE)
    parser.add_argument("--jvm-max-memory", default=natural_env.JVM_MAX_MEMORY, help="Minecraft's heap limit, e.g. 2G")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--port", type=int, default=8023)
    args = parser.parse_args()

    ports = instance_ports(args.port, args.n_envs)
    rd, heap = args.render_distance, args.jvm_max_memory
    env = launch_vec_env([lambda p=p: natural_env.make_env(port=p, env_path=game_folder(p),
                                                           render_distance=rd, jvm_max_memory=heap)
                          for p in ports], ports)
    started = datetime.now().isoformat(timespec="seconds")
    new_file = not CSV.exists()
    rng = np.random.default_rng()
    try:
        env.reset()
        t0 = time.monotonic()
        next_log, steps = 0.0, 0
        with CSV.open("a", newline="", encoding="utf-8") as f:
            out = csv.DictWriter(f, FIELDS)
            if new_file:
                out.writeheader()
            while True:
                minute = (time.monotonic() - t0) / 60
                if minute >= next_log:
                    per = [instance_gb(p) for p in ports]
                    row = {"started": started, "render_distance": rd, "jvm_max_memory": heap, "n_envs": args.n_envs,
                           "minute": round(minute, 1), "steps": steps,
                           "free_gb": round(psutil.virtual_memory().available / 2**30, 2),
                           "instances_gb": round(sum(per), 2),
                           "per_instance_gb": " ".join(f"{g:.2f}" for g in per)}
                    out.writerow(row)
                    f.flush()
                    print(row, flush=True)
                    next_log += 1
                    if minute >= args.minutes:
                        break
                env.step(rng.integers(0, env.action_space.n, size=args.n_envs))
                steps += args.n_envs
    finally:
        close_vec_env(env, ports)


if __name__ == "__main__":
    main()
