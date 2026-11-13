"""Tree-chopping environment for CraftGround.

The agent spawns in the middle of a walled, flat arena with a ring of oak
trees. It holds an unbreakable diamond axe that breaks a log in one hit.
Reward is +1 for every log it breaks. The arena is rebuilt at the start of
every episode, so chopped trees grow back.

Arena layout (seen from above, 1 char = 1 block, P = player spawn):

    ###########################   <- stone brick wall at +/-13
    #                         #
    #          T   T   T      #   <- T = oak tree, ring of radius 5
    #            P            #
    #          T   T   T      #
    #                         #
    ###########################
"""

import random
from typing import Any, Optional

import gymnasium as gym
import numpy as np

import craftground
from craftground import InitialEnvironmentConfig
from craftground.environment.action_space import no_op
from craftground.initial_environment_config import (
    DaylightMode,
    Difficulty,
    GameMode,
    WorldType,
)

# --- Experiment settings -----------------------------------------------------

IMAGE_WIDTH = 114
IMAGE_HEIGHT = 64
EPISODE_STEPS = 1000  # one step = one game tick (1/20 s), so ~50 s of game time

# Superflat ground: grass is at y=-61, so the player stands at y=-60.
GROUND_Y = -60
TRUNK_HEIGHT = 4  # logs per tree
TREE_POSITIONS = [(5, 0), (-5, 0), (0, 5), (0, -5), (4, 4), (-4, 4), (4, -4), (-4, -4)]
ARENA_RADIUS = 13  # walls sit at +/- this many blocks from the centre
WALL_HEIGHT = 3

MAX_LOGS = len(TREE_POSITIONS) * TRUNK_HEIGHT  # 32

# Blocks whose "mined" statistic counts as a chopped log.
LOG_KEYS = ["oak_log"]

# Efficiency X breaks a log in one tick, so a single "attack" step is enough.
AXE = (
    "minecraft:diamond_axe[minecraft:unbreakable={},"
    'minecraft:enchantments={levels:{"minecraft:efficiency":10}}]'
)

CAMERA_STEP = 1  # camera actions turn by CAMERA_STEP * 15 degrees


def arena_commands() -> list[str]:
    """Commands that (re)build the arena. CraftGround runs these on every reset."""
    y0 = GROUND_Y
    r = ARENA_RADIUS
    top = y0 + WALL_HEIGHT - 1
    cmds = [
        # World rules: always day, clear weather, no mobs.
        "gamerule doMobSpawning false",
        "gamerule doDaylightCycle false",
        "gamerule doWeatherCycle false",
        "time set noon",
        "weather clear",
        "gamerule spawnRadius 0",
        f"setworldspawn 0 {y0} 0",
        # Wipe the inside of the arena (old stumps, leaves) and any dropped items.
        f"fill -{r - 1} {y0} -{r - 1} {r - 1} {y0 + 10} {r - 1} air",
        "kill @e[type=item]",
        # Walls.
        f"fill -{r} {y0} -{r} {r} {top} -{r} stone_bricks",
        f"fill -{r} {y0} {r} {r} {top} {r} stone_bricks",
        f"fill -{r} {y0} -{r} -{r} {top} {r} stone_bricks",
        f"fill {r} {y0} -{r} {r} {top} {r} stone_bricks",
    ]
    leaves = "oak_leaves[persistent=true]"
    trunk_top = y0 + TRUNK_HEIGHT - 1
    for x, z in TREE_POSITIONS:
        # Canopy sits above the trunk so every log can be reached from the ground.
        cmds.append(f"fill {x - 2} {trunk_top + 1} {z - 2} {x + 2} {trunk_top + 1} {z + 2} {leaves}")
        cmds.append(f"fill {x - 1} {trunk_top + 2} {z - 1} {x + 1} {trunk_top + 2} {z + 1} {leaves}")
        cmds.append(f"fill {x} {y0} {z} {x} {trunk_top} {z} oak_log")
    cmds += [
        f"tp @p 0.5 {y0} 0.5 0 0",
        "clear @p",
        f"item replace entity @p weapon.mainhand with {AXE}",
    ]
    return cmds


# --- Actions -----------------------------------------------------------------

ACTION_NAMES = [
    "noop",
    "forward",
    "back",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "attack",
    "forward_attack",
]


def to_craftground_action(index: int) -> list[int]:
    """Turn one of ACTION_NAMES into CraftGround's 8-number MineDojo-style action."""
    name = ACTION_NAMES[index]
    act = no_op()  # [move, strafe, jump, pitch=12, yaw=12, functional, ...]
    if name in ("forward", "forward_attack"):
        act[0] = 1
    elif name == "back":
        act[0] = 2
    elif name == "strafe_left":
        act[1] = 1
    elif name == "strafe_right":
        act[1] = 2
    elif name == "turn_left":
        act[4] = 12 - CAMERA_STEP
    elif name == "turn_right":
        act[4] = 12 + CAMERA_STEP
    elif name == "look_up":
        act[3] = 12 - CAMERA_STEP
    elif name == "look_down":
        act[3] = 12 + CAMERA_STEP
    if name in ("attack", "forward_attack"):
        act[5] = 3
    return act


# --- Environment wrapper -----------------------------------------------------


class ChopTreesEnv(gym.Wrapper):
    """Gymnasium env: image in, discrete action out, +1 reward per log chopped."""

    def __init__(self, port: int = 8023, verbose: bool = False):
        config = InitialEnvironmentConfig(
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            world_type=WorldType.SUPERFLAT,
            gamemode=GameMode.SURVIVAL,
            difficulty=Difficulty.PEACEFUL,
            generate_structures=False,
            hud_hidden=True,
            mined_stat_keys=LOG_KEYS,
            initial_extra_commands=arena_commands(),
        )
        config.set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY)
        super().__init__(craftground.make(initial_env_config=config, port=port, verbose=verbose))

        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8
        )
        self._steps = 0
        self._logs_at_start = 0
        self._logs_chopped = 0

    @staticmethod
    def _logs_mined_total(obs) -> int:
        stats = obs["full"].mined_statistics
        return sum(stats.get(key, 0) for key in LOG_KEYS)

    @staticmethod
    def _image(obs) -> np.ndarray:
        return np.ascontiguousarray(obs["pov"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Face a random direction each episode so the agent can't memorise one path.
        yaw = random.choice(range(-180, 180, 15))
        obs, info = self.env.reset(
            seed=seed,
            options={"extra_commands": [f"tp @p 0.5 {GROUND_Y} 0.5 {yaw} 0"]},
        )
        self._steps = 0
        # Minecraft stats never reset, so count logs relative to the episode start.
        self._logs_at_start = self._logs_mined_total(obs)
        self._logs_chopped = 0
        return self._image(obs), {"logs_chopped": 0}

    def step(self, action):
        obs, _, _, _, _ = self.env.step(to_craftground_action(int(action)))
        self._steps += 1

        logs_now = self._logs_mined_total(obs) - self._logs_at_start
        reward = float(logs_now - self._logs_chopped)
        self._logs_chopped = logs_now

        terminated = bool(obs["full"].is_dead)
        truncated = self._steps >= EPISODE_STEPS or self._logs_chopped >= MAX_LOGS
        info: dict[str, Any] = {"logs_chopped": self._logs_chopped}
        return self._image(obs), reward, terminated, truncated, info


def make_env(port: int = 8023, verbose: bool = False) -> ChopTreesEnv:
    return ChopTreesEnv(port=port, verbose=verbose)
