"""The tree arena, presented the way the MineRL tree-chopping model expects.

Same arena as tree_env.py (walls, 8 oaks, unbreakable one-hit axe, +1 per log),
but observations and actions copy the MineRL project's train_treechop.py, so an
SB3 PPO model trained there can be loaded here unchanged:

Observation (copies process_pov):
  Minecraft draws 320x180 with the HUD showing, so the hand and held axe look
  like MineRL. Keep the top 138 rows (cutting off hotbar, hearts and hunger),
  then cv2.resize to 64x64 with INTER_AREA. RGB, uint8, shape (64, 64, 3).

Action: MultiDiscrete([3, 11, 9]), one choice from each list:
  move   0 stand, 1 forward, 2 forward+jump
  yaw    YAW_DEGREES[i], positive = turn right
  pitch  PITCH_DEGREES[i], positive = look down (MineRL's camera convention)
  Attack is always held. Each decision lasts FRAME_SKIP ticks; the turn is
  split evenly over those ticks and the logs from all of them are added up.

Episodes last tree_env.EPISODE_STEPS ticks (1000 ticks = 250 decisions), and end
early if the agent dies or every log is chopped.
"""

import random
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np

import craftground
from craftground import InitialEnvironmentConfig
from craftground.environment.action_space import ActionSpaceVersion, no_op_v2
from craftground.initial_environment_config import (
    DaylightMode,
    Difficulty,
    GameMode,
    WorldType,
)

from tree_env import EPISODE_STEPS, GROUND_Y, LOG_KEYS, MAX_LOGS, arena_commands

# --- Copied from the MineRL project (train_treechop.py) ----------------------

RENDER_RES = (320, 180)  # (width, height) Minecraft draws at
HUD_TOP_ROW = 138        # keep rows above this (the HUD starts around row 140)
OBS_W, OBS_H = 64, 64

FRAME_SKIP = 4
MOVES = ["stand", "forward", "forward_jump"]
YAW_DEGREES = [-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40]
PITCH_DEGREES = [-20, -10, -5, -2, 0, 2, 5, 10, 20]

EPISODE_DECISIONS = EPISODE_STEPS // FRAME_SKIP  # 250


def process_pov(pov: np.ndarray) -> np.ndarray:
    """Crop off the HUD and shrink the frame to what the network sees."""
    world = pov[:HUD_TOP_ROW]
    small = cv2.resize(world, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small)


def describe_action(action) -> str:
    move, yaw, pitch = (int(a) for a in action)
    return f"{MOVES[move]} yaw{YAW_DEGREES[yaw]:+d} pitch{PITCH_DEGREES[pitch]:+d}"


class MineRLCompatEnv(gym.Env):
    """Gymnasium env with the MineRL model's observation and action spaces."""

    metadata = {"render_modes": []}

    def __init__(self, port: int = 8023, verbose: bool = False):
        config = InitialEnvironmentConfig(
            image_width=RENDER_RES[0],
            image_height=RENDER_RES[1],
            world_type=WorldType.SUPERFLAT,
            gamemode=GameMode.SURVIVAL,
            difficulty=Difficulty.PEACEFUL,
            generate_structures=False,
            hud_hidden=False,  # MineRL shows the hand; the hotbar is cropped off below
            mined_stat_keys=LOG_KEYS,
            initial_extra_commands=arena_commands(),
        )
        config.set_daylight_cycle_mode(DaylightMode.ALWAYS_DAY)
        self.env = craftground.make(
            initial_env_config=config,
            port=port,
            verbose=verbose,
            action_space_version=ActionSpaceVersion.V2_MINERL_HUMAN,
        )

        self.action_space = gym.spaces.MultiDiscrete([len(MOVES), len(YAW_DEGREES), len(PITCH_DEGREES)])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(OBS_H, OBS_W, 3), dtype=np.uint8)
        self.last_pov: Optional[np.ndarray] = None  # full 320x180 frame, for debugging/videos
        self._ticks = 0
        self._logs_at_start = 0
        self._logs_chopped = 0

    @staticmethod
    def _logs_mined_total(obs) -> int:
        stats = obs["full"].mined_statistics
        return sum(stats.get(key, 0) for key in LOG_KEYS)

    def _observe(self, obs) -> np.ndarray:
        pov = np.asarray(obs["pov"])
        if pov.shape[:2] != (RENDER_RES[1], RENDER_RES[0]):
            raise RuntimeError(f"expected a {RENDER_RES[0]}x{RENDER_RES[1]} frame, got {pov.shape}")
        self.last_pov = pov
        return process_pov(pov)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        yaw = random.choice(range(-180, 180, 15))
        self.env.reset(
            seed=seed,
            options={"extra_commands": [f"tp @p 0.5 {GROUND_Y} 0.5 {yaw} 0"]},
        )
        # CraftGround's fast reset kills and respawns the player. Minecraft blocks
        # mining after the death screen until the attack key is released, and this
        # agent never releases it, so hold one tick with attack up before the
        # episode starts (that tick isn't counted).
        obs, *_ = self.env.step(no_op_v2())
        self._ticks = 0
        self._logs_at_start = self._logs_mined_total(obs)
        self._logs_chopped = 0
        return self._observe(obs), {"logs_chopped": 0}

    def step(self, action):
        move, yaw, pitch = (int(a) for a in action)
        tick = no_op_v2()
        tick["attack"] = True
        tick["forward"] = move >= 1
        tick["jump"] = move == 2
        tick["camera_yaw"] = YAW_DEGREES[yaw] / FRAME_SKIP
        tick["camera_pitch"] = PITCH_DEGREES[pitch] / FRAME_SKIP

        reward = 0.0
        terminated = truncated = False
        for _ in range(FRAME_SKIP):
            obs, _, _, _, _ = self.env.step(dict(tick))
            self._ticks += 1
            logs_now = self._logs_mined_total(obs) - self._logs_at_start
            reward += logs_now - self._logs_chopped
            self._logs_chopped = logs_now
            terminated = bool(obs["full"].is_dead)
            truncated = self._ticks >= EPISODE_STEPS or self._logs_chopped >= MAX_LOGS
            if terminated or truncated:
                break

        info: dict[str, Any] = {"logs_chopped": self._logs_chopped}
        return self._observe(obs), reward, terminated, truncated, info

    def close(self):
        self.env.close()


def make_env(port: int = 8023, verbose: bool = False) -> MineRLCompatEnv:
    return MineRLCompatEnv(port=port, verbose=verbose)
