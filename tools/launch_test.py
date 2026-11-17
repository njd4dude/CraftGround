"""Does launching many Minecraft instances at once crash, and why? Arena vs natural.

    run.bat tools/launch_test.py --envs arena natural --instances 6 --trials 4
    run.bat tools/launch_test.py --envs arena --instances 6 --trials 4 --serial    # with launch_lock.SerialLaunch

Each trial starts N instances of one world at the same moment (like train.py,
evaluate.py and benchmark.py do), waits for each one's first reset, then closes
everything. Nothing is played. Gradle's own output for every instance is saved
to launch_logs/<env>_t<trial>_<port>.log; CraftGround normally discards it, but
it holds the Loom cache-lock messages that explain a crash:

  "Found existing cache lock file (...), rebuilding loom cache"   a build decided
      another build's lock was left over from a crash and rebuilt the shared cache
  "Locking process does not exist" / "disowned"                    why it decided that
  "Failed to remap minecraft"                                       the crash it causes

Each instance runs in its own process with a time limit, so a failed instance is
reported instead of hanging the test. Results are appended to
results/launch_test.csv.
"""

import argparse
import multiprocessing as mp
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.reports.csvio import append_row

LOG_DIR = Path("launch_logs")
CSV = Path("results") / "launch_test.csv"
FIELDS = ["timestamp", "env", "n_envs", "trial", "serial", "started", "failed", "timed_out",
          "rebuilt_cache", "lock_messages", "max_launch_s", "error"]
TIMEOUT_S = 420
LOOM_PATTERNS = {
    "rebuild": "rebuilding loom cache",
    "missing": "Locking process does not exist",
    "disowned": "disowned the lock",
    "waiting": "Waiting for lock to be released",
    "held": "is currently held by pid",
    "remap_fail": "Failed to remap minecraft",
}


def _worker(env_name: str, port: int, log_path: str, serial: bool, out: mp.Queue, env_path=None):
    import craftground.environment.environment as cge

    real_popen = subprocess.Popen

    def logging_popen(cmd, **kw):  # send this instance's Gradle output to its own file
        if "gradlew" in str(cmd):
            # CraftGround runs Gradle with -w (warnings only), which hides Loom's lock messages.
            cmd = str(cmd).replace(" -w ", " ")
            log = open(log_path, "w", encoding="utf-8", errors="replace")
            kw["stdout"], kw["stderr"] = log, subprocess.STDOUT
        return real_popen(cmd, **kw)

    cge.subprocess.Popen = logging_popen
    from chop.envs import arena as arena_env, natural as natural_env

    env = {"arena": arena_env, "natural": natural_env}[env_name].make_env(port=port, env_path=env_path)
    if serial:
        from chop.infra.launch_lock import SerialLaunch
        env = SerialLaunch(env, name=str(port))
    t = time.perf_counter()
    try:
        env.reset()
        out.put((port, "ok", time.perf_counter() - t, ""))
    except Exception as e:
        out.put((port, "failed", time.perf_counter() - t, f"{type(e).__name__}: {e}"))
    finally:
        try:
            env.close()
        except Exception:
            pass


def kill_tree(proc: psutil.Process):
    procs = [proc] + proc.children(recursive=True) if proc.is_running() else []
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=10)


def plant_disowned_lock():
    """Write the lock file Loom leaves behind after a crashed build ("disowned"), so the
    next launch finds it. Name = '.' + last 16 hex chars of sha1(projectDir + '::') + '.lock'."""
    import hashlib

    import craftground_runtime_mc121
    project_dir = str(Path(craftground_runtime_mc121.__file__).parent)
    h = hashlib.sha1(f"{project_dir}::".encode()).hexdigest()[-16:]
    (Path.home() / ".gradle/caches/fabric-loom" / f".{h}.lock").write_text("disowned")


def run_trial(env_name: str, n: int, trial: int, port: int, serial: bool, plant: bool = False,
              separate: bool = False) -> dict:
    LOG_DIR.mkdir(exist_ok=True)
    if plant:
        plant_disowned_lock()
    ports = [port + 10 * i for i in range(n)]
    paths = [None] * n
    if separate:
        from chop.infra.instances import env_path_for
        paths = [env_path_for(p) for p in ports]
    stamp = datetime.now()
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    folder = "sep" if separate else "shared"
    logs = {p: LOG_DIR / f"{env_name}_{folder}_t{trial}_{p}.log" for p in ports}
    procs = {p: ctx.Process(target=_worker, args=(env_name, p, str(logs[p]), serial, out, paths[i]))
             for i, p in enumerate(ports)}
    mode = " serial" if serial else ""
    mode += " separate-folders" if separate else ""
    print(f"[{env_name} x{n} trial {trial}{mode}] launching...", flush=True)
    for pr in procs.values():
        pr.start()

    results, deadline = {}, time.time() + TIMEOUT_S * (n if serial else 1)
    while len(results) < n and time.time() < deadline:
        try:
            p, status, secs, err = out.get(timeout=5)
            results[p] = (status, secs, err)
            print(f"  port {p}: {status} after {secs:.0f}s {err[:100]}", flush=True)
        except Exception:
            if all(not pr.is_alive() for pr in procs.values()):
                break
    for pr in procs.values():
        pr.join(timeout=30)
        if pr.is_alive():
            kill_tree(psutil.Process(pr.pid))
    time.sleep(5)

    found = {k: 0 for k in LOOM_PATTERNS}
    for p, log in logs.items():
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        for k, pat in LOOM_PATTERNS.items():
            if re.search(re.escape(pat), text):
                found[k] += 1
    ok = [r for r in results.values() if r[0] == "ok"]
    row = {
        "timestamp": stamp.isoformat(timespec="seconds"), "env": env_name, "n_envs": n, "trial": trial,
        "serial": serial, "started": len(ok), "failed": sum(r[0] == "failed" for r in results.values()),
        "timed_out": n - len(results), "rebuilt_cache": found["rebuild"],
        "lock_messages": " ".join(f"{k}={v}" for k, v in found.items() if v),
        "max_launch_s": f"{max((r[1] for r in ok), default=0):.0f}",
        "error": next((r[2] for r in results.values() if r[2]), "")[:200],
    }
    print(f"  => {row['started']}/{n} started, {row['failed']} failed, {row['timed_out']} timed out; "
          f"Loom: {row['lock_messages'] or 'no lock messages'}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", nargs="+", choices=["arena", "natural"], default=["arena", "natural"])
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--port", type=int, default=8523)
    parser.add_argument("--serial", action="store_true", help="launch one at a time (launch_lock.SerialLaunch)")
    parser.add_argument("--plant-disowned", action="store_true",
                        help="before each trial, leave the lock file a crashed Loom build leaves behind")
    parser.add_argument("--separate-folders", action="store_true",
                        help="each instance runs from its own copy of the game folder (chop/infra/instances.py)")
    parser.add_argument("--alternate", action="store_true",
                        help="each trial runs one shared-folder and one separate-folder launch, "
                             "swapping which goes first every trial")
    args = parser.parse_args()
    rows = []
    for trial in range(1, args.trials + 1):
        if args.alternate:
            modes = [False, True] if trial % 2 else [True, False]
        else:
            modes = [args.separate_folders]
        for env_name in args.envs:  # alternate worlds so time of day / machine state affects both equally
            for separate in modes:
                row = run_trial(env_name, args.instances, trial, args.port, args.serial, args.plant_disowned,
                                separate)
                tags = ("[planted disowned lock] " if args.plant_disowned else "") + \
                       ("[separate folders] " if separate else "") + \
                       ("[alternating] " if args.alternate else "")
                row["error"] = (tags + row["error"])[:200]
                row["_separate"] = separate
                append_row(CSV, FIELDS, {k: v for k, v in row.items() if k in FIELDS})
                rows.append(row)
    print("\nSummary:")
    for env_name in args.envs:
        for separate in sorted({r["_separate"] for r in rows}):
            rs = [r for r in rows if r["env"] == env_name and r["_separate"] == separate]
            bad = sum(1 for r in rs if r["started"] < args.instances)
            print(f"  {env_name} {'separate folders' if separate else 'shared folder'}: "
                  f"{bad}/{len(rs)} trials had an instance fail; "
                  f"cache rebuilt in {sum(1 for r in rs if r['rebuilt_cache'])}/{len(rs)}")


if __name__ == "__main__":
    main()
