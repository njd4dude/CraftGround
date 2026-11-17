"""Measure how fast the environments run, and record it in BENCHMARK.md.

    run.bat benchmark.py                                  # arena + natural, 1 2 4 6 instances
    run.bat benchmark.py --envs natural --instances 4     # just one setup
    run.bat benchmark.py --instances 1 2 4 6 8            # include 8 (needs ~18 GB free RAM)

For each world and instance count it starts that many Minecraft instances side
by side (like train.py / evaluate.py do), plays WARMUP steps per instance
untimed, then times STEPS steps per instance with random actions and no model.
So the numbers are the environment's own speed: what training or evaluation can
reach at most. A model adds its own time on top (choosing actions, PPO updates).

Episode resets are included in the timing, the same way they happen during
training (every 1000 steps, plus earlier in the arena when all logs are chopped),
because in the natural world they are a real part of the cost. The share of time
spent resetting is reported separately.

RAM per instance = (used system RAM while running - used RAM before launching) / instances.
A setup is skipped if free RAM looks too small for it.

Results go to results/benchmark.csv (one row per setup and run) and BENCHMARK.md
(tables rebuilt from the CSV; notes you write below the tables are kept).
"""

import argparse
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from chop.envs import natural as natural_env
from chop.envs import arena as arena_env
from chop.infra.instances import clear_stale_loom_locks, env_path_for
from chop.infra.launch_lock import SerialLaunch
from chop.reports.csvio import append_row, read_rows
from chop.reports.markdown import END, START, table, write_report

ENVS = {"arena": arena_env, "natural": natural_env}
BENCH_CSV = Path("results") / "benchmark.csv"
BENCH_REPORT = Path("BENCHMARK.md")
RESETS_CSV = Path("results") / "benchmark_resets.csv"  # one row per timed reset (where its time went)
RESET_FIELDS = ["run", "env", "n_envs", "instance", "step", "reset_seconds", "base_reset_s", "search_s",
                "setup_s", "render_s", "near_tries", "far_tries", "found"]
FIELDS = ["timestamp", "env", "n_envs", "steps_per_instance", "seconds", "steps_per_s", "per_instance",
          "resets", "avg_reset_s", "min_reset_s", "median_reset_s", "max_reset_s", "reset_share", "ram_gb_per_instance", "launch_s", "machine", "status"]

STEPS = 3000              # timed steps per instance (~3 episodes, so resets are included)
WARMUP = 200              # untimed steps per instance first
RAM_PER_INSTANCE_GB = 2.5  # assumed when deciding whether a setup fits in free RAM
RAM_SPARE_GB = 3.0

# What the train.py docstring claimed for the arena (raw env steps/s), to compare against.
CLAIMED_ARENA = {1: 451, 2: 714, 4: 858, 8: 1076}

TEMPLATE = f"""# CraftGround Benchmark: Environment Speed

How many env steps per second the tree-chopping environments run at, measured by
`benchmark.py`. Rebuild these tables by running it again; each run adds rows.

{START}
{END}

## Notes

Write your observations here. This section is never overwritten.
"""


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        import subprocess
        name = subprocess.run(["powershell", "-NoProfile", "-c", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        cpu = name or cpu
    except Exception:
        pass
    ram = psutil.virtual_memory().total / 2**30
    return f"{cpu}, {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads, {ram:.0f} GB RAM"


def own_descendants() -> list[psutil.Process]:
    try:
        return psutil.Process(os.getpid()).children(recursive=True)
    except psutil.Error:
        return []


def kill_leftovers(procs: list[psutil.Process]) -> int:
    """Kill processes this run started that are still alive after env.close() (CraftGround
    sometimes leaves Minecraft instances running)."""
    alive = [p for p in procs if p.is_running()]
    for p in alive:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=10)
    return len(alive)


def run_setup(env_name: str, n: int, port: int, steps: int, warmup: int, machine_name: str,
              serial_launch: bool = False, shared_folder: bool = False) -> dict:
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "env": env_name, "n_envs": n,
           "steps_per_instance": steps, "machine": machine_name}
    free_gb = psutil.virtual_memory().available / 2**30
    need_gb = n * RAM_PER_INSTANCE_GB + RAM_SPARE_GB
    if free_gb < need_gb:
        print(f"[{env_name} x{n}] skipped: {free_gb:.1f} GB free RAM, needs ~{need_gb:.0f} GB")
        return {**row, "status": f"skipped (only {free_gb:.1f} GB free)"}

    make_env = ENVS[env_name].make_env
    used_before = psutil.virtual_memory().used
    print(f"[{env_name} x{n}] launching {n} Minecraft instance(s)...")
    t_launch = time.perf_counter()
    clear_stale_loom_locks()  # see chop/infra/instances.py

    def build(p: int):
        env = make_env(port=p, env_path=None if shared_folder else env_path_for(p, verbose=True))
        return SerialLaunch(env, name=str(p)) if serial_launch else env

    factories = [(lambda p=port + 10 * i: build(p)) for i in range(n)]
    env = SubprocVecEnv(factories) if n > 1 else DummyVecEnv(factories)
    procs: list[psutil.Process] = []
    try:
        env.reset()
        launch_s = time.perf_counter() - t_launch
        procs = own_descendants()  # remember them now, so leftovers can be cleaned up after close

        def random_actions():
            return np.array([env.action_space.sample() for _ in range(n)])

        for _ in range(warmup):
            env.step(random_actions())

        resets, reset_s, peak_used = 0, 0.0, psutil.virtual_memory().used
        reset_times: list[float] = []
        print(f"[{env_name} x{n}] timing {steps} steps per instance...")
        t0 = time.perf_counter()
        for k in range(steps):
            _, _, dones, _ = env.step(random_actions())
            for i in np.flatnonzero(dones):
                info = env.reset_infos[i]
                resets += 1
                reset_s += info.get("reset_seconds", 0.0)
                reset_times.append(info.get("reset_seconds", 0.0))
                append_row(RESETS_CSV, RESET_FIELDS, {
                    "run": row["timestamp"], "env": env_name, "n_envs": n, "instance": int(i), "step": k,
                    **{f: (f"{info[f]:.2f}" if isinstance(info.get(f), float) else info.get(f, ""))
                       for f in RESET_FIELDS[5:]}})
            if k % 250 == 0:
                peak_used = max(peak_used, psutil.virtual_memory().used)
        seconds = time.perf_counter() - t0
    except Exception as e:
        print(f"[{env_name} x{n}] failed: {e!r}")
        return {**row, "status": f"failed: {type(e).__name__}"}
    finally:
        env.close()
        time.sleep(3)
        left = kill_leftovers(procs)
        if left:
            print(f"[{env_name} x{n}] killed {left} leftover process(es)")

    total = n * steps / seconds
    result = {
        **row,
        "seconds": f"{seconds:.1f}",
        "steps_per_s": f"{total:.0f}",
        "per_instance": f"{total / n:.0f}",
        "resets": resets,
        "avg_reset_s": f"{reset_s / resets:.2f}" if resets else "",
        "min_reset_s": f"{min(reset_times):.2f}" if resets else "",
        "median_reset_s": f"{float(np.median(reset_times)):.2f}" if resets else "",
        "max_reset_s": f"{max(reset_times):.2f}" if resets else "",
        "reset_share": f"{reset_s / (n * seconds):.2f}",
        "ram_gb_per_instance": f"{(peak_used - used_before) / 2**30 / n:.2f}",
        "launch_s": f"{launch_s:.0f}",
        "status": "ok",
    }
    print(f"[{env_name} x{n}] {total:.0f} steps/s total, {total / n:.0f} per instance, "
          f"{resets} resets ({result['avg_reset_s'] or '-'} s avg, {result['min_reset_s'] or '-'}/"
          f"{result['median_reset_s'] or '-'}/{result['max_reset_s'] or '-'} s min/median/max, "
          f"{float(result['reset_share']):.0%} of time), "
          f"~{result['ram_gb_per_instance']} GB RAM per instance")
    return result


def spread(r: dict) -> str:
    vals = [r.get(k) for k in ("min_reset_s", "median_reset_s", "max_reset_s")]
    return " / ".join(vals) if all(vals) else "–"


def build_report() -> None:
    rows = read_rows(BENCH_CSV)
    ok = [r for r in rows if r.get("status") == "ok"]
    parts = [f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"]

    parts.append(
        "## How it was measured\n\n"
        "`benchmark.py` starts N Minecraft instances side by side, plays "
        f"{WARMUP} untimed steps per instance, then times {STEPS:,} steps per instance with **random "
        "actions and no model**, so these are the environments' own speeds: the most that training or "
        "evaluation can reach. Episode resets happen during the timed steps just as in training and are "
        "included. **Reset share** is the part of instance time spent inside reset() (spawn search, "
        "terrain loading); since instances step in lockstep, a slow reset also stalls the others, so its "
        "real cost is higher. **RAM/instance** is the rise in used system RAM divided by N.\n\n"
        f"Machine: {ok[-1]['machine'] if ok else machine()}"
    )

    for env_name in ENVS:
        latest: dict[int, dict] = {}
        for r in ok:
            if r["env"] == env_name:
                latest[int(r["n_envs"])] = r  # newest measurement per instance count
        if not latest:
            parts.append(f"## {env_name.capitalize()}\n\n_Not measured yet._")
            continue
        base = float(latest[min(latest)]["steps_per_s"]) if 1 in latest else None
        out = []
        for n in sorted(latest):
            r = latest[n]
            speedup = f"{float(r['steps_per_s']) / base:.2f}×" if base else "–"
            claim = f"{CLAIMED_ARENA[n]}" if env_name == "arena" and n in CLAIMED_ARENA else "–"
            cells = [n, f"**{r['steps_per_s']}**", r["per_instance"], speedup, r["resets"],
                     r["avg_reset_s"] or "–", spread(r), f"{float(r['reset_share']):.0%}", r["ram_gb_per_instance"],
                     r["launch_s"], r["timestamp"][:16].replace("T", " ")]
            if env_name == "arena":
                cells.insert(2, claim)
            out.append(cells)
        headers = ["Instances", "Steps/s total", "Per instance", "vs 1 instance", "Resets", "Avg reset (s)", "Min / median / max (s)",
                   "Reset share", "RAM/instance (GB)", "Launch (s)", "Measured"]
        if env_name == "arena":
            headers.insert(2, "Claimed in train.py")
        world = ("walled superflat arena, `chop/envs/arena.py`" if env_name == "arena"
                 else "forest in a normal world, `chop/envs/natural.py`")
        parts.append(f"## {env_name.capitalize()} ({world})\n\nLatest measurement per instance count.\n\n"
                     + table(headers, out))

    if rows:
        hist = [[r["timestamp"][:16].replace("T", " "), r["env"], r["n_envs"], r.get("steps_per_s") or "–",
                 r.get("per_instance") or "–", r.get("status")] for r in reversed(rows)]
        parts.append("## All runs\n\nEvery measurement, newest first.\n\n"
                     + table(["When", "World", "Instances", "Steps/s total", "Per instance", "Status"], hist))
    write_report(BENCH_REPORT, TEMPLATE, "\n\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", choices=sorted(ENVS), default=["arena", "natural"])
    parser.add_argument("--instances", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--steps", type=int, default=STEPS, help=f"timed steps per instance (default {STEPS})")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--port", type=int, default=8523, help="first instance's port (others go up by 10)")
    parser.add_argument("--serial-launch", action="store_true",
                        help="start instances one at a time (another way to avoid startup collisions, chop/infra/launch_lock.py)")
    parser.add_argument("--shared-folder", action="store_true",
                        help="run all instances from CraftGround's one shared game folder (old behaviour)")
    args = parser.parse_args()

    machine_name = machine()
    print(f"Machine: {machine_name}")
    t_start = time.perf_counter()
    for env_name in args.envs:
        for n in sorted(args.instances):
            row = run_setup(env_name, n, args.port, args.steps, args.warmup, machine_name, args.serial_launch,
                            args.shared_folder)
            append_row(BENCH_CSV, FIELDS, row)
            build_report()
    print(f"\nDone in {(time.perf_counter() - t_start) / 60:.1f} min. Results in {BENCH_REPORT} and {BENCH_CSV}")


if __name__ == "__main__":
    main()
