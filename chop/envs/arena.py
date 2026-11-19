"""The arena: tree chopping in a walled superflat world.

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
import time
from typing import Any, Optional

import gymnasium as gym
import numpy as np

import craftground
from craftground import InitialEnvironmentConfig
from craftground.initial_environment_config import (
    DaylightMode,
    Difficulty,
    GameMode,
    WorldType,
)

from craftground.environment.action_space import ActionSpaceVersion, no_op, no_op_v2

from chop.envs.common import (ACTION_NAMES, ACTION_NAMES_JUMP, AXE, EPISODE_STEPS, IMAGE_HEIGHT, IMAGE_WIDTH,
                              multi_nvec, to_craftground_action, to_craftground_multi)

# --- Experiment settings -----------------------------------------------------

# Superflat ground: grass is at y=-61 over two layers of dirt (-63, -62) on bedrock (-64),
# so the player stands at y=-60.
GROUND_Y = -60
TRUNK_HEIGHT = 4  # logs per tree
TREE_POSITIONS = [(5, 0), (-5, 0), (0, 5), (0, -5), (4, 4), (-4, 4), (4, -4), (-4, -4)]
ARENA_RADIUS = 13  # walls sit at +/- this many blocks from the centre
WALL_HEIGHT = 3

MAX_LOGS = len(TREE_POSITIONS) * TRUNK_HEIGHT  # 32
# Describe the reward here whenever you change it in step(): every training run
# records this text in its config.json.
REWARD = "+1 per log chopped"

# Blocks whose "mined" statistic counts as a chopped log.
LOG_KEYS = ["oak_log"]


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
        # Repair the ground, walls' footprint included: the agent can dig (an axe breaks dirt in
        # ~15 ticks of attack) and holes used to stay for the rest of the run. Move the player
        # onto the surface first so a player standing in a hole isn't buried by the refill.
        f"tp @p 0.5 {y0} 0.5",
        f"fill -{r} {y0 - 3} -{r} {r} {y0 - 2} {r} dirt",
        f"fill -{r} {y0 - 1} -{r} {r} {y0 - 1} {r} grass_block",
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


# --- Environment wrapper -----------------------------------------------------


class ChopTreesEnv(gym.Wrapper):
    """Gymnasium env: image in, action out, +1 reward per log chopped.
    actions: "single" = one of ACTION_NAMES per step; "multi" = (move, turn, look) every step
    with attack always held (common.MULTI_NVEC).
    jump: off by default (the arena is flat). On, it offers the natural world's action set instead
    (13 single actions, or a 4th jump choice in multi), so an idea tested here can go to the
    natural world unchanged, and a model trained here can be --resume'd there. The walls are 3
    blocks high, so jumping can't get the agent out.
    action_repeat: game ticks per agent step. Keys (walk, attack, jump) are held for all of them;
    a turn / look is applied once, on the first tick, so it still turns by the amount chosen.
    Rewards are summed and the agent sees the last tick's frame. CraftGround shows an action's
    effects 1-3 ticks late (tools/check_timing.py), so 4 lets each step show its own action.
    An episode is still EPISODE_STEPS ticks, so it's EPISODE_STEPS // action_repeat agent steps."""

    def __init__(self, port: int = 8023, verbose: bool = False, env_path: Optional[str] = None,
                 actions: str = "single", action_repeat: int = 1, jump: bool = False):
        if actions not in ("single", "multi"):
            raise ValueError(f"actions must be 'single' or 'multi', not {actions!r}")
        if action_repeat < 1:
            raise ValueError(f"action_repeat must be at least 1, not {action_repeat}")
        self.multi = actions == "multi"
        self.jump = jump
        self.action_repeat = action_repeat
        self.max_steps = EPISODE_STEPS // action_repeat
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
        # Multi actions turn by exact degrees, which needs CraftGround's V2 action format.
        version = ActionSpaceVersion.V2_MINERL_HUMAN if self.multi else ActionSpaceVersion.V1_MINEDOJO
        super().__init__(craftground.make(initial_env_config=config, port=port, verbose=verbose, env_path=env_path,
                                          action_space_version=version))

        self.action_space = (gym.spaces.MultiDiscrete(multi_nvec(jump)) if self.multi
                             else gym.spaces.Discrete(len(ACTION_NAMES_JUMP if jump else ACTION_NAMES)))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8
        )
        self._steps = 0
        self._logs_at_start = 0
        self._logs_chopped = 0

    def _game_action(self, action, camera: bool):
        """CraftGround's action for one tick; camera=False drops the turn / look (repeat ticks)."""
        if self.multi:
            act = to_craftground_multi(action)
            if not camera:
                act["camera_yaw"] = act["camera_pitch"] = 0.0
        else:
            act = to_craftground_action(int(action))
            if not camera:
                act[3] = act[4] = 12  # 12 = no pitch / yaw change
        return act

    @staticmethod
    def _logs_mined_total(obs) -> int:
        stats = obs["full"].mined_statistics
        return sum(stats.get(key, 0) for key in LOG_KEYS)

    @staticmethod
    def _image(obs) -> np.ndarray:
        return np.ascontiguousarray(obs["pov"])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """options={"extra_commands": [...]}: commands run after the random spawn (tools/ use it to
        teleport the player somewhere exact)."""
        start = time.perf_counter()
        # Face a random direction each episode so the agent can't memorise one path.
        yaw = random.choice(range(-180, 180, 15))
        extra = list((options or {}).get("extra_commands", []))
        obs, info = self.env.reset(
            seed=seed,
            options={"extra_commands": [f"tp @p 0.5 {GROUND_Y} 0.5 {yaw} 0", *extra]},
        )
        # Let go of every key and button for one tick. CraftGround only sends Minecraft a mouse
        # press when attack goes from off to on, and Minecraft lets go of held buttons during a
        # reset, so an attack still held from the last episode would never press again: in multi
        # mode (attack always held) the agent couldn't chop after its first episode.
        obs, _, _, _, _ = self.env.step(no_op_v2() if self.multi else no_op())
        self._steps = 0
        # Minecraft stats never reset, so count logs relative to the episode start.
        self._logs_at_start = self._logs_mined_total(obs)
        self._logs_chopped = 0
        return self._image(obs), {"logs_chopped": 0, "reset_seconds": time.perf_counter() - start}

    def step(self, action):
        reward = 0.0
        terminated = False
        for tick in range(self.action_repeat):
            obs, _, _, _, _ = self.env.step(self._game_action(action, camera=tick == 0))
            logs_now = self._logs_mined_total(obs) - self._logs_at_start
            reward += float(logs_now - self._logs_chopped)
            self._logs_chopped = logs_now
            terminated = bool(obs["full"].is_dead)
            if terminated or self._logs_chopped >= MAX_LOGS:
                break
        self._steps += 1

        truncated = self._steps >= self.max_steps or self._logs_chopped >= MAX_LOGS
        info: dict[str, Any] = {"logs_chopped": self._logs_chopped}
        return self._image(obs), reward, terminated, truncated, info


def make_env(port: int = 8023, verbose: bool = False, env_path: Optional[str] = None,
             actions: str = "single", action_repeat: int = 1, jump: bool = False) -> ChopTreesEnv:
    """env_path: game folder to run from (chop.infra.instances.env_path_for(port)); None = CraftGround's shared one.
    actions: "single" or "multi"; action_repeat: game ticks per agent step; jump: offer the natural
    world's jump actions (see ChopTreesEnv)."""
    return ChopTreesEnv(port=port, verbose=verbose, env_path=env_path, actions=actions,
                        action_repeat=action_repeat, jump=jump)
