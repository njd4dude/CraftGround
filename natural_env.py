"""Tree chopping in a natural Minecraft world.

Same task as tree_env.py: same image, diamond axe and +1 reward per log. Actions are
the arena's 11 plus jump and forward_jump (13), since the terrain has hills and ledges;
make_env(jump=False) gives the arena's 11 for models trained before jump was added.
The difference is the world. Instead of a walled superflat arena with 8 placed
oaks, the agent plays in a normal generated world (fixed seed) and spawns in a
forest.

Chopped trees don't grow back in a natural world, so every episode starts at a
new spot. On each reset the env drops the player at random points (first within
HOP_RANGE blocks of the last good spot, since forests are contiguous; later
anywhere within FAR_RANGE) and keeps the first landing that is
  - in an oak/birch forest biome (FOREST_CHECK), and
  - on dirt/grass, i.e. under the trees rather than on top of the canopy.

How: CraftGround runs a reset's extra_commands in no reliable order, so the
search instead sends one command per step (env.add_command) and waits for that
command's reply in the chat before sending the next. The forest check is a
single `execute if ...` command, which replies "Test passed" when it passes
(a failure shows nothing, so a miss is a short timeout).
These setup steps are not part of the episode.

Run check_natural.py to see it working (frames go to check_frames/).
"""

import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np

import craftground
from craftground import InitialEnvironmentConfig
from craftground.initial_environment_config import Difficulty, GameMode, WorldType

from tree_env import (ACTION_NAMES, ACTION_NAMES_JUMP, AXE, EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH,
                      to_craftground_action)

# --- Experiment settings -----------------------------------------------------

SEED = "20261114"  # fixed so runs are comparable; every instance builds the same world
# A second world that training never uses (evaluate.py --held-out): the agent has never seen
# any of its forests, so its score there shows how well the agent generalises to new terrain.
HELD_OUT_SEED = "20261117"
# Oak and birch forests: minecraft:is_forest minus dark forest and (snowy, spruce) grove.
FOREST_CHECK = ("if biome ~ ~ ~ #minecraft:is_forest "
                "unless biome ~ ~ ~ minecraft:dark_forest unless biome ~ ~ ~ minecraft:grove")
LOG_KEYS = ["oak_log", "birch_log", "spruce_log", "dark_oak_log",
            "jungle_log", "acacia_log", "mangrove_log", "cherry_log"]
REWARD = "+1 per log chopped (any wood type)"

HOP_RANGE = 150       # first landings: within this many blocks of the last good spot
HOP_TRIES = 12        # landings near the last spot before searching the whole world
FAR_RANGE = 25_000    # then: anywhere in this square around 0, 0
MAX_TRIES = 60
REPLY_TIMEOUT_S = 60.0  # wall-clock wait for one command's reply (new terrain is slow)
# The forest check only replies when it passes, so a miss is a timeout. A pass has always
# replied within 0.31 s, teleport included (results/benchmark_resets.csv), so this leaves
# 2x margin; a pass that came later would only cost one extra try.
CHECK_TIMEOUT_S = 0.75
# After the teleport, idle until the view stops changing: chunks appear a few at a time
# (in wall-clock time, not ticks), so "drawn" = no new terrain for RENDER_STABLE_S.
RENDER_STABLE_S = 1.5
RENDER_MIN_S = 0.5
RENDER_MAX_S = 15.0
RENDER_TOLERANCE = 0.005  # change in non-sky fraction that counts as "still drawing"
MIN_GROUND = 0.15         # and at least this much of the view must be terrain (not just sky)
DEBUG = False
# Fixed evaluation spots (make_eval_spots.py): the same forest spots for every model, so
# evaluations differ by the agent, not by which forests it happened to land in.
EVAL_SPOTS = Path(__file__).parent / "results" / "eval_spots_natural.json"


def eval_spots_path(seed: str = SEED) -> Path:
    """Each world has its own spot file: EVAL_SPOTS for the training world,
    eval_spots_natural_seed<seed>.json for any other."""
    return EVAL_SPOTS if seed == SEED else EVAL_SPOTS.with_name(f"eval_spots_natural_seed{seed}.json")

# Memory: Java grows Minecraft's heap towards its limit as episodes hop to new forests, so with
# CraftGround's defaults (3G heap, render distance 6) 4 instances outgrew this 32 GB machine
# about 15-20 minutes into a run, twice. Each instance uses ~1.5 GB on top of the heap (code,
# graphics, Gradle launcher processes), so a 2G heap caps it at ~3.3-3.7 GB.
# memtest.py, 20 min, 4 instances (results/memtest.csv): 3G -> 12.3 GB, 2G -> 12.0 GB and still
# levelling off, no slowdown or crash. Both at render distance 4 (less terrain per spot; 2G at 6
# is untested). Changed 2026-11-15; earlier runs used 6 / 3G (see each run's config.json).
RENDER_DISTANCE = 4   # chunks drawn around the player (CraftGround's default: 6)
JVM_MAX_MEMORY = "2G"  # Minecraft's Java heap limit, -Xmx (CraftGround's default: 3G)

NOOP = to_craftground_action(0)


def _setup_commands() -> list[str]:
    """Order-independent world rules, run on every reset."""
    return [
        "gamerule doMobSpawning false",
        "gamerule doDaylightCycle false",
        "gamerule doWeatherCycle false",
        "time set noon",
        "weather clear",
        "kill @e[type=item]",
        "effect give @p minecraft:resistance infinite 255 true",  # landing / falling never hurts
    ]


def ground_y(y: float) -> int:
    """The ground surface a recorded height belongs to. Right after a landing the client lets the
    player sink a little (y=63.85 on ground at 64), so most saved spots are a fraction of a block
    low; teleporting there puts the feet inside the top block, the player drops through it and
    suffocates. Forest spots stand on dirt/grass (full blocks), so the surface is y rounded up."""
    return math.ceil(y - 1e-3)


def _ground_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that aren't sky (sky = bright and clearly blue)."""
    img = np.asarray(image, dtype=np.int16)
    r, b = img[..., 0], img[..., 2]
    sky = (b > 200) & (b - r > 40)
    return 1.0 - float(sky.mean())


class NaturalChopTreesEnv(gym.Wrapper):
    """Gymnasium env: image in, discrete action out, +1 reward per log chopped."""

    def __init__(self, port: int = 8023, verbose: bool = False, env_path: Optional[str] = None,
                 jump: bool = True, render_distance: int = RENDER_DISTANCE,
                 jvm_max_memory: str = JVM_MAX_MEMORY, spots: Optional[list[dict]] = None,
                 seed: str = SEED):
        """jump: offer the 13 actions with jump / forward_jump (default) instead of the arena's 11.
        Hills and 1-block ledges are common here; without jump the agent can only walk around them.
        Use jump=False to run models trained before jump was added (2026-11-15).
        spots: play these spots ({x, y, z, yaw}) in order, one per reset, instead of searching
        (evaluate.py's fixed evaluation set); once they run out, resets search as usual.
        seed: the world to generate. Training always uses SEED; HELD_OUT_SEED is for evaluation."""
        self.world_seed = seed
        self.action_names = ACTION_NAMES_JUMP if jump else ACTION_NAMES
        # CraftGround's build.gradle reads this when it launches Minecraft (-Xmx).
        os.environ["CRAFTGROUND_JVM_MAX_MEMORY"] = jvm_max_memory
        config = InitialEnvironmentConfig(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            world_type=WorldType.DEFAULT,
            seed=seed,
            gamemode=GameMode.SURVIVAL,
            difficulty=Difficulty.PEACEFUL,
            generate_structures=False,
            hud_hidden=True,
            render_distance=render_distance,
            simulation_distance=render_distance,
            mined_stat_keys=LOG_KEYS,
            initial_extra_commands=_setup_commands(),
        )
        super().__init__(craftground.make(initial_env_config=config, port=port, verbose=verbose, env_path=env_path))

        self.action_space = gym.spaces.Discrete(len(self.action_names))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8
        )
        self._rng = random.Random()
        self._anchor: Optional[tuple[int, int]] = None  # last good spot
        self._steps = 0
        self._logs_at_start = 0
        self._logs_chopped = 0
        self.spot: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.found = False  # whether the last reset found a forest spot
        self._spots = list(spots or [])
        self.spot_index: Optional[int] = None  # which fixed spot the current episode plays

    # --- running commands one at a time ---------------------------------------

    def _command(self, command: str, replies: tuple[str, ...], timeout: float = REPLY_TIMEOUT_S,
                 quiet: bool = False):
        """Send one command and idle until a chat line containing one of `replies` arrives.
        Returns (obs, matching chat line or None on timeout)."""
        self.env.unwrapped.add_command(command)
        deadline = time.monotonic() + timeout
        while True:
            obs, _, _, _, _ = self.env.step(NOOP)
            for m in obs["full"].chat_messages:
                if any(r in m.message for r in replies):
                    return obs, m.message
            if time.monotonic() > deadline:
                if not quiet:
                    print(f"[natural_env] no reply to: {command}")
                return obs, None

    def _try_spot(self, x: int, z: int, r: int):
        """Drop the player somewhere within r blocks of (x, z); return (obs, is_forest)."""
        obs, reply = self._command(f"spreadplayers {x} {z} 0 {r} false @p",
                                   ("Spread", "Could not spread", "Unable", "Invalid", "rror"))
        if reply is None or "Spread" not in reply:
            return obs, False
        obs, reply = self._command(
            f"execute as @p at @s if block ~ ~-1 ~ #minecraft:dirt {FOREST_CHECK}",
            ("Test passed",), timeout=CHECK_TIMEOUT_S, quiet=True)
        return obs, reply is not None and "Test passed" in reply

    def find_far_spot(self) -> Optional[dict]:
        """One try at a forest spot anywhere in the world (the same test resets use). Returns
        {x, y, z} where the player now stands, or None if this landing wasn't a forest."""
        obs, ok = self._try_spot(0, 0, FAR_RANGE)
        if not ok:
            return None
        f = obs["full"]
        return {"x": round(f.x, 2), "y": ground_y(f.y), "z": round(f.z, 2)}

    def _wait_for_render(self, obs):
        """Idle until the new terrain has finished drawing (see RENDER_STABLE_S)."""
        start = last_change = time.monotonic()
        last = _ground_fraction(obs["pov"])
        while True:
            obs, _, _, _, _ = self.env.step(NOOP)
            now = time.monotonic()
            frac = _ground_fraction(obs["pov"])
            if abs(frac - last) > RENDER_TOLERANCE:
                last, last_change = frac, now
            if (now - start >= RENDER_MIN_S and now - last_change >= RENDER_STABLE_S
                    and frac >= MIN_GROUND):
                return obs
            if now - start >= RENDER_MAX_S:
                if DEBUG:
                    print("[natural_env] terrain still changing after "
                          f"{RENDER_MAX_S:.0f}s; starting anyway")
                return obs

    # --- gym API ---------------------------------------------------------------

    @staticmethod
    def _logs_mined_total(obs) -> int:
        stats = obs["full"].mined_statistics
        return sum(stats.get(key, 0) for key in LOG_KEYS)

    @staticmethod
    def _image(obs) -> np.ndarray:
        return np.ascontiguousarray(obs["pov"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        start = time.perf_counter()
        if seed is not None:
            self._rng.seed(seed)
        obs, _ = self.env.reset(seed=seed)
        t_base = time.perf_counter()

        self.found = False
        near_tries = far_tries = 0
        self.spot_index = None
        fixed = self._spots.pop(0) if self._spots else None
        if fixed is not None:
            # A fixed evaluation spot: teleport there (same seed, so same terrain) and skip the search.
            self.spot_index = fixed.get("index")
            obs, reply = self._command(f"tp @p {fixed['x']} {ground_y(fixed['y'])} {fixed['z']} {fixed['yaw']} 0",
                                       ("Teleported",))
            self.found = reply is not None
        for attempt in range(0 if fixed is not None else MAX_TRIES):
            if self._anchor is not None and attempt < HOP_TRIES:
                near_tries += 1
                obs, self.found = self._try_spot(*self._anchor, HOP_RANGE)
            else:
                far_tries += 1
                obs, self.found = self._try_spot(0, 0, FAR_RANGE)
            if DEBUG:
                f = obs["full"]
                print(f"[natural_env] try {attempt}: forest={self.found} at ({f.x:.0f}, {f.y:.0f}, {f.z:.0f})")
            if self.found:
                break
        else:
            if fixed is None:
                print(f"[natural_env] no forest spot found in {MAX_TRIES} tries; playing anyway")

        t_search = time.perf_counter()
        yaw = fixed["yaw"] if fixed is not None else self._rng.choice(range(-180, 180, 15))
        obs, _ = self._command(f"execute as @p at @s run tp @s ~ ~ ~ {yaw} 0", ("Teleported",))
        obs, _ = self._command("clear @p", ("Removed", "No items"))
        obs, _ = self._command(f"item replace entity @p weapon.mainhand with {AXE}", ("Replaced",))
        t_setup = time.perf_counter()
        obs = self._wait_for_render(obs)
        t_render = time.perf_counter()

        full = obs["full"]
        self.spot = (full.x, full.y, full.z)
        if self.found and fixed is None:
            self._anchor = (int(full.x), int(full.z))
        self._steps = 0
        # Minecraft stats never reset, so count logs relative to the episode start.
        self._logs_at_start = self._logs_mined_total(obs)
        self._logs_chopped = 0
        # Where the reset's time went (benchmark.py logs these per reset).
        timing = {
            "reset_seconds": time.perf_counter() - start,
            "base_reset_s": t_base - start,       # CraftGround's own reset
            "search_s": t_search - t_base,        # spreadplayers + forest checks
            "setup_s": t_setup - t_search,        # turn, clear inventory, give axe
            "render_s": t_render - t_setup,       # waiting for terrain to draw
            "near_tries": near_tries,
            "far_tries": far_tries,
            "found": self.found,
            "spot_index": self.spot_index,
        }
        return self._image(obs), {"logs_chopped": 0, **timing}

    def step(self, action):
        obs, _, _, _, _ = self.env.step(to_craftground_action(int(action)))
        self._steps += 1

        logs_now = self._logs_mined_total(obs) - self._logs_at_start
        reward = float(logs_now - self._logs_chopped)
        self._logs_chopped = logs_now

        terminated = bool(obs["full"].is_dead)
        truncated = self._steps >= EPISODE_STEPS
        info: dict[str, Any] = {"logs_chopped": self._logs_chopped}
        return self._image(obs), reward, terminated, truncated, info


def make_env(port: int = 8023, verbose: bool = False, env_path: Optional[str] = None,
             jump: bool = True, render_distance: int = RENDER_DISTANCE,
             jvm_max_memory: str = JVM_MAX_MEMORY, spots: Optional[list[dict]] = None,
             seed: str = SEED) -> NaturalChopTreesEnv:
    """env_path: game folder to run from (instances.env_path_for(port)); None = CraftGround's shared one.
    jump: 13 actions with jump (default) or the arena's 11 (for older models).
    spots: fixed spots to play in order (see load_eval_spots).
    seed: world seed (SEED for training; HELD_OUT_SEED for the held-out evaluation)."""
    return NaturalChopTreesEnv(port=port, verbose=verbose, env_path=env_path, jump=jump,
                               render_distance=render_distance, jvm_max_memory=jvm_max_memory, spots=spots,
                               seed=seed)


def load_eval_spots(path: Path = EVAL_SPOTS, seed: Optional[str] = None) -> list[dict]:
    """The fixed evaluation spots, each tagged with its "index" in the file. With seed, refuses a
    file made in a different world: its coordinates would land somewhere else entirely."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if seed is not None and str(data.get("world_seed")) != str(seed):
        raise ValueError(f"{path} was made in world {data.get('world_seed')}, not {seed}")
    return [{**s, "index": i} for i, s in enumerate(data["spots"])]
