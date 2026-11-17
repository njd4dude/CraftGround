"""Where Minecraft instances' memory goes, over time: run it next to training or memtest.py.

    python tools/memprobe.py --minutes 50                  # instances on ports 8023 8033 8043 8053
    python tools/memprobe.py --ports 8023 8033 --every 60

Every --every seconds, for each port's Minecraft process it records, to results/memprobe.csv:
  process      rss (what Windows counts as in RAM) and private (memory only it owns)
  java_total   everything the JVM itself knows about (Native Memory Tracking, committed)
  nmt:<area>   that total split up: Java Heap, Class (loaded code), Thread, Code (JIT),
               GC, Compiler, Internal, Other (direct buffers), Symbol, ...
  heap_used    live + not-yet-collected objects inside the heap (GC.heap_info)
  outside_java private - java_total: memory the JVM doesn't track (graphics driver,
               OpenGL/LWJGL native allocations, CraftGround's native library)
plus the total of each port's Gradle launcher processes and of all python.exe processes.

Needs the instances to run with -XX:NativeMemoryTracking (CraftGround turns it on).
"""

import argparse
import csv
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.infra.instances import instance_processes

CSV = Path("results/memprobe.csv")
JCMD = Path(psutil.Process().exe()).parent / "Library" / "lib" / "jvm" / "bin" / "jcmd.exe"
NMT_LINE = re.compile(r"^-\s+(.+?) \(reserved=(\d+)KB, committed=(\d+)KB\)", re.M)
NMT_TOTAL = re.compile(r"Total: reserved=(\d+)KB, committed=(\d+)KB")
HEAP_USED = re.compile(r"total (\d+)K, used (\d+)K")


def jcmd(pid: int, command: str) -> str:
    try:
        return subprocess.run([str(JCMD), str(pid), command], capture_output=True, text=True, timeout=30).stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def minecraft_and_launchers(port: int):
    mc, launchers = None, []
    for p in instance_processes(port):
        try:
            if "fabric.dli" in " ".join(p.cmdline()):
                mc = p
            else:
                launchers.append(p)
        except psutil.Error:
            pass
    return mc, launchers


def sample(port: int) -> dict[str, float]:
    """Megabytes per metric for one instance (empty if it isn't running)."""
    mc, launchers = minecraft_and_launchers(port)
    if mc is None:
        return {}
    out: dict[str, float] = {}
    try:
        mem = mc.memory_info()
        out["process:rss"] = mem.rss / 2**20
        out["process:private"] = getattr(mem, "private", mem.rss) / 2**20
    except psutil.Error:
        return {}
    nmt = jcmd(mc.pid, "VM.native_memory summary")
    if total := NMT_TOTAL.search(nmt):
        out["java_total"] = int(total.group(2)) / 1024
        for name, _reserved, committed in NMT_LINE.findall(nmt):
            out[f"nmt:{name.strip()}"] = int(committed) / 1024
        out["outside_java"] = out["process:private"] - out["java_total"]
    if heap := HEAP_USED.search(jcmd(mc.pid, "GC.heap_info")):
        out["heap_committed"] = int(heap.group(1)) / 1024
        out["heap_used"] = int(heap.group(2)) / 1024
    launcher_rss = 0
    for p in launchers:
        try:
            launcher_rss += p.memory_info().rss
        except psutil.Error:
            pass
    out["gradle_launchers:rss"] = launcher_rss / 2**20
    return out


def python_rss_mb() -> float:
    total = 0
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info["name"] or "").lower() == "python.exe":
                total += p.memory_info().rss
        except psutil.Error:
            pass
    return total / 2**20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ports", type=int, nargs="+", default=[8023, 8033, 8043, 8053])
    parser.add_argument("--every", type=float, default=120, help="seconds between samples")
    parser.add_argument("--minutes", type=float, default=60, help="stop after this long")
    args = parser.parse_args()

    started = datetime.now().isoformat(timespec="seconds")
    new_file = not CSV.exists()
    t0 = time.monotonic()
    with CSV.open("a", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        if new_file:
            out.writerow(["started", "minute", "port", "metric", "mb"])
        while (minute := (time.monotonic() - t0) / 60) <= args.minutes:
            rows = [(port, k, v) for port in args.ports for k, v in sample(port).items()]
            rows.append(("all", "python:rss", python_rss_mb()))
            rows.append(("all", "system:available", psutil.virtual_memory().available / 2**20))
            out.writerows([started, round(minute, 1), port, k, round(v, 1)] for port, k, v in rows)
            f.flush()
            mc = [v for _, k, v in rows if k == "process:private"]
            print(f"[{minute:5.1f} min] {len(mc)} instance(s), private MB: {' '.join(f'{v:.0f}' for v in mc)}",
                  flush=True)
            time.sleep(args.every)


if __name__ == "__main__":
    main()
