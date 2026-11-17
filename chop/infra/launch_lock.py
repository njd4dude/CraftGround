"""Start Minecraft instances one at a time instead of all at once.

Every CraftGround instance launches by running Gradle in the same installed
project folder (site-packages/craftground_runtime_mc121). The Fabric Loom plugin
guards its cache of converted Minecraft files with a lock file. When several
Gradle builds start together, one can decide another's lock is left over from
a crash and rebuild the cache while the others are reading it ("Failed to remap
minecraft"), so an instance fails to start. A single launch reuses the cache.

SerialLaunch holds a lock (shared by all processes on this machine) around an
env's first reset, which is when CraftGround starts Minecraft, so instances
launch one after another. Later resets don't take the lock.
"""

import tempfile
import time
from pathlib import Path

import gymnasium as gym
from filelock import FileLock

LOCK_PATH = Path(tempfile.gettempdir()) / "craftground_launch.lock"


class SerialLaunch(gym.Wrapper):
    def __init__(self, env: gym.Env, name: str = ""):
        super().__init__(env)
        self.name = name
        self.launched = False

    def reset(self, **kwargs):
        if self.launched:
            return self.env.reset(**kwargs)
        t = time.perf_counter()
        with FileLock(str(LOCK_PATH)):
            waited = time.perf_counter() - t
            print(f"[launch {self.name}] starting Minecraft (waited {waited:.0f}s for other launches)", flush=True)
            result = self.env.reset(**kwargs)
        self.launched = True
        print(f"[launch {self.name}] up after {time.perf_counter() - t:.0f}s", flush=True)
        return result
