"""Tree chopping in a natural Minecraft world.

Same task as tree_env.py: same image, actions, diamond axe and +1 reward per log.
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

import random
import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np

import craftground
from craftground import InitialEnvironmentConfig
from craftground.initial_environment_config import Difficulty, GameMode, WorldType

from tree_env import ACTION_NAMES, AXE, EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH, to_craftground_action

# --- Experiment settings -----------------------------------------------------

SEED = "20261114"  # fixed so runs are comparable; every instance builds the same world
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
CHECK_TIMEOUT_S = 3.0   # the forest check only replies when it passes, so a miss is a timeout
# After the teleport, idle until the view stops changing: chunks appear a few at a time
# (in wall-clock time, not ticks), so "drawn" = no new terrain for RENDER_STABLE_S.
RENDER_STABLE_S = 1.5
RENDER_MIN_S = 0.5
RENDER_MAX_S = 15.0
RENDER_TOLERANCE = 0.005  # change in non-sky fraction that counts as "still drawing"
MIN_GROUND = 0.15         # and at least this much of the view must be terrain (not just sky)
DEBUG = False

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


def _ground_fraction(image: np.ndarray) -> float:
    """Fraction of pixels that aren't sky (sky = bright and clearly blue)."""
    img = np.asarray(image, dtype=np.int16)
    r, b = img[..., 0], img[..., 2]
    sky = (b > 200) & (b - r > 40)
    return 1.0 - float(sky.mean())


class NaturalChopTreesEnv(gym.Wrapper):
    """Gymnasium env: image in, discrete action out, +1 reward per log chopped."""

    def __init__(self, port: int = 8023, verbose: bool = False, env_path: Optional[str] = None):
        config = InitialEnvironmentConfig(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            world_type=WorldType.DEFAULT,
            seed=SEED,
            gamemode=GameMode.SURVIVAL,
            difficulty=Difficulty.PEACEFUL,
            generate_structures=False,
            hud_hidden=True,
            mined_stat_keys=LOG_KEYS,
            initial_extra_commands=_setup_commands(),
        )
        super().__init__(craftground.make(initial_env_config=config, port=port, verbose=verbose, env_path=env_path))

        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
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
        for attempt in range(MAX_TRIES):
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
            print(f"[natural_env] no forest spot found in {MAX_TRIES} tries; playing anyway")

        t_search = time.perf_counter()
        yaw = self._rng.choice(range(-180, 180, 15))
        obs, _ = self._command(f"execute as @p at @s run tp @s ~ ~ ~ {yaw} 0", ("Teleported",))
        obs, _ = self._command("clear @p", ("Removed", "No items"))
        obs, _ = self._command(f"item replace entity @p weapon.mainhand with {AXE}", ("Replaced",))
        t_setup = time.perf_counter()
        obs = self._wait_for_render(obs)
        t_render = time.perf_counter()

        full = obs["full"]
        self.spot = (full.x, full.y, full.z)
        if self.found:
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


def make_env(port: int = 8023, verbose: bool = False, env_path: Optional[str] = None) -> NaturalChopTreesEnv:
    """env_path: game folder to run from (instances.env_path_for(port)); None = CraftGround's shared one."""
    return NaturalChopTreesEnv(port=port, verbose=verbose, env_path=env_path)
