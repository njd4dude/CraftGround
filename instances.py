"""Give each Minecraft instance its own copy of CraftGround's game folder.

Why: by default every CraftGround instance runs from the one folder installed with
the package (site-packages/craftground_runtime_mc121). Instances started at the
same moment then collide there: two pick the same "New World (N)" save name, or
two Gradle builds write the same logging config, and one instance fails to start
(about 1 launch in 7 with 6 instances; see results/launch_test.csv). With a folder
each, 25 of 25 test rounds started cleanly, just as fast.

    from instances import env_path_for
    env = tree_env.make_env(port=port, env_path=env_path_for(port))

env_path_for(port):
  * returns C:/Users/<you>/craftground_instances/mc121_port<port>, copying it from
    the installed package the first time (about 50 MB) and building it (mod code and
    native library, ~1-2 min, once; new copies build one at a time). Folders go by
    port because every running instance needs its own port anyway, so two scripts
    running at once (say training and an evaluation) never share a folder.
  * deletes that copy's old world saves (CraftGround never removes them).
  * deletes stale Loom lock files (clear_stale_loom_locks).

Stale Loom locks: Gradle's Fabric Loom plugin keeps one cache of converted
Minecraft files for all folders (~/.gradle/caches/fabric-loom), guarded by a lock
file per folder. A run that is killed (Ctrl+C, out of memory, a crash) can leave
its lock behind; the next launch then rebuilds the shared cache while other
instances are reading it, and fails with "Failed to remap minecraft" / "file is
being used by another process". Deleting locks whose owner is gone prevents the
rebuild (with no lock, Loom just uses the cache).

After upgrading CraftGround, call refresh_instances() (or delete the folder) so
the copies are made again from the new version.
"""

import shutil
import time
from pathlib import Path

import craftground_runtime_mc121
import psutil

SOURCE = Path(craftground_runtime_mc121.__file__).parent
ROOT = Path.home() / "craftground_instances"
LOOM_CACHE = Path.home() / ".gradle" / "caches" / "fabric-loom"

# Left out of copies: world saves and logs, and CMake build files that hold the original
# folder's absolute path (CraftGround's own list in runtime_packages.py plus Visual Studio outputs).
SKIP_DIRS = {"saves", "logs", "CMakeFiles", "_deps", "x64", "Debug", "Release", "native-lib.dir", "__pycache__"}
SKIP_FILES = {"CMakeCache.txt", "cmake_install.cmake", "compile_commands.json", "Makefile"}
SKIP_SUFFIXES = {".vcxproj", ".filters", ".sln"}


def _ignore(directory: str, names: list[str]) -> set[str]:
    skipped = set()
    for name in names:
        path = Path(directory) / name
        if path.is_dir() and name in SKIP_DIRS:
            skipped.add(name)
        elif path.is_file() and (name in SKIP_FILES or path.suffix in SKIP_SUFFIXES):
            skipped.add(name)
    return skipped


BUILT_MARKER = ".craftground_built"
BUILD_TIMEOUT_S = 900


def _build(folder: Path) -> None:
    """Compile a new copy's mod (Kotlin/Java) and native library once, so launches skip it.

    Builds go one at a time (a lock shared by all processes): Gradle sends Kotlin
    compiles from every build on the machine to one shared compiler process, and six
    at once ran it out of memory ("GC overhead limit exceeded"), failing launches."""
    import subprocess

    from filelock import FileLock

    with FileLock(str(ROOT / ".build.lock")):
        if (folder / BUILT_MARKER).exists():
            return  # another process built it while we waited
        print(f"[instances] building {folder.name} (first use, ~1-2 min)...", flush=True)
        log = folder / "build_first_use.log"
        with open(log, "w", encoding="utf-8", errors="replace") as out:
            result = subprocess.run(".\\gradlew.bat classes compileCpp --no-daemon", cwd=folder, shell=True,
                                    stdout=out, stderr=subprocess.STDOUT, timeout=BUILD_TIMEOUT_S)
        if result.returncode != 0:
            raise RuntimeError(f"Building {folder} failed (exit {result.returncode}); see {log}")
        (folder / BUILT_MARKER).write_text("ok")


def instance_folder(port: int) -> Path:
    """The copy used by the instance on this port, created and built from the installed package if missing."""
    target = ROOT / f"mc121_port{port}"
    if not (target / "gradlew.bat").exists():
        ROOT.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(SOURCE, tmp, ignore=_ignore)
        (tmp / "run" / "saves").mkdir(parents=True, exist_ok=True)
        tmp.rename(target)  # only complete copies get the real name
    if not (target / BUILT_MARKER).exists():
        _build(target)  # built at its final path: Gradle treats a moved folder as a new project
    return target


def clear_old_saves(folder: Path) -> int:
    """Delete the worlds left in a copy's saves folder. Only call it right before this
    copy's instance starts: nothing else uses the copy, so nothing is open in there."""
    saves = folder / "run" / "saves"
    removed = 0
    for world in saves.glob("*") if saves.exists() else []:
        shutil.rmtree(world, ignore_errors=True)
        removed += not world.exists()
    return removed


def clear_stale_loom_locks(min_age_s: float = 60.0) -> list[str]:
    """Delete Loom lock files whose owner is gone: ones that say "disowned" (left by a
    crashed build) or name a process that no longer exists. A lock of a running build
    is kept, so this is safe while other runs are going. An empty or unreadable lock is
    only removed once it is min_age_s old (a build may be writing it right now)."""
    removed = []
    for lock in LOOM_CACHE.glob(".*.lock"):
        try:
            text = lock.read_text().strip()
            age = time.time() - lock.stat().st_mtime
        except OSError:
            continue
        if text == "disowned":
            stale = True
        elif text.isdigit():
            stale = not psutil.pid_exists(int(text))
        else:
            stale = age > min_age_s
        if stale:
            try:
                lock.unlink()
                removed.append(f"{lock.name} ({text or 'empty'})")
            except OSError:
                pass  # another instance removed it first, or it's in use
    return removed


def env_path_for(port: int, verbose: bool = False) -> str:
    """Folder to pass as env_path for the instance on this port, ready to launch."""
    folder = instance_folder(port)
    saves = clear_old_saves(folder)
    locks = clear_stale_loom_locks()
    if verbose and (saves or locks):
        print(f"[port {port}] removed {saves} old world(s)"
              + (f", stale Loom lock(s): {', '.join(locks)}" if locks else ""), flush=True)
    return str(folder)


def refresh_instances() -> None:
    """Delete all copies; they are made again from the installed package on next use."""
    shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    import sys

    for p in (int(a) for a in sys.argv[1:]) if len(sys.argv) > 1 else [8023 + 10 * i for i in range(6)]:
        print(env_path_for(p, verbose=True))
