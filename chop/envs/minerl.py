"""MineRL-compatible view and actions (make_env(actions="minerl") in the arena or the natural world).

Copies the MineRL project's MineRLTreechopAxe320-v0 setup (~/mineRL/train_treechop.py in WSL), so an
SB3 PPO/BC model trained there loads and runs here unchanged:

Observation (process_pov): Minecraft draws 320x180 with the HUD showing, so the hand and held axe
  look like MineRL. Keep the top 138 rows (cuts off hotbar, hearts and hunger), then cv2.resize to
  64x64 with INTER_AREA. RGB uint8 (64, 64, 3).
Action: MultiDiscrete([3, 11, 9]), one choice from each list:
  move   0 stand, 1 forward, 2 forward+jump
  yaw    YAW_DEGREES[i], positive = turn right
  pitch  PITCH_DEGREES[i], positive = look down
  Attack is always held. Each decision lasts FRAME_SKIP ticks; buttons are held every tick and the
  camera turn is split evenly over them (what MineRL does), and the logs from all of them add up.
Axe: a plain iron axe (several hits per log), as in MineRL, unless the env is given another.

Known differences that can't be removed: CraftGround runs a newer Minecraft than MineRL's 1.16
(slightly different textures), worlds are peaceful (MineRL had mobs), and the worlds themselves differ.
"""

import sys
import time

import cv2
import numpy as np
from craftground.environment.action_space import no_op_v2

RENDER_RES = (320, 180)  # (width, height) Minecraft draws at
HUD_TOP_ROW = 138        # keep rows above this (the HUD starts around row 140)
OBS_W, OBS_H = 64, 64
OBS_SHAPE = (OBS_H, OBS_W, 3)

FRAME_SKIP = 4
MOVES = ["stand", "forward", "forward_jump"]
YAW_DEGREES = [-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40]
PITCH_DEGREES = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
NVEC = [len(MOVES), len(YAW_DEGREES), len(PITCH_DEGREES)]

IRON_AXE = "minecraft:iron_axe"
TOAST_SECONDS = 7.0  # a toast stays 5 s; a little margin (see unlock_all_recipes)

SETTINGS = {"source": "MineRL MineRLTreechopAxe320-v0", "render": list(RENDER_RES), "hud_top_row": HUD_TOP_ROW,
            "obs": [OBS_W, OBS_H], "frame_skip": FRAME_SKIP, "move": MOVES, "yaw_degrees": YAW_DEGREES,
            "pitch_degrees": PITCH_DEGREES, "attack": "always held"}


def unlock_all_recipes(craftground_env, noop: dict) -> dict:
    """Stop "New Recipes Unlocked!" pop-ups. MineRL patched toasts out of Minecraft; here every log,
    sapling or apple picked up unlocks recipes, and the toast covers the top-right of the view for
    seconds (tools/check_minerl_toasts.py). Unlocking every recipe once shows one last toast; this
    idles until it has gone (toasts time out in wall-clock seconds, not ticks). The recipe book
    survives CraftGround's reset (which respawns the player), so each instance needs this once.
    Returns the last observation."""
    craftground_env.add_commands(["recipe give @p *"])
    deadline = time.monotonic() + TOAST_SECONDS
    while True:
        obs, *_ = craftground_env.step(dict(noop))
        if time.monotonic() > deadline:
            return obs


def process_pov(pov: np.ndarray) -> np.ndarray:
    """Crop off the HUD and shrink the frame to what the network sees."""
    world = np.asarray(pov)[:HUD_TOP_ROW]
    return np.ascontiguousarray(cv2.resize(world, (OBS_W, OBS_H), interpolation=cv2.INTER_AREA))


def tick_action(action, frame_skip: int = FRAME_SKIP) -> dict:
    """CraftGround's V2 action for each of the frame_skip ticks of one (move, yaw, pitch) decision."""
    move, yaw, pitch = (int(a) for a in action)
    act = no_op_v2()
    act["attack"] = True
    act["forward"] = move >= 1
    act["jump"] = move == 2
    act["camera_yaw"] = YAW_DEGREES[yaw] / frame_skip
    act["camera_pitch"] = PITCH_DEGREES[pitch] / frame_skip
    return act


def label(action) -> str:
    move, yaw, pitch = (int(a) for a in action)
    parts = [MOVES[move]]
    if YAW_DEGREES[yaw]:
        parts.append(f"yaw{YAW_DEGREES[yaw]:+d}")
    if PITCH_DEGREES[pitch]:
        parts.append(f"pitch{PITCH_DEGREES[pitch]:+d}")
    return " ".join(parts)


def is_minerl_model(model) -> bool:
    """A model trained in MineRL's spaces: 64x64 image (either channel order) and MultiDiscrete([3, 11, 9]).
    (Arena multi models also have [3, 11, 9] actions, but see a 114x64 image.)"""
    shape = tuple(model.observation_space.shape)
    nvec = getattr(model.action_space, "nvec", None)
    return shape in ((OBS_H, OBS_W, 3), (3, OBS_H, OBS_W)) and nvec is not None and list(nvec) == NVEC


def register_networks() -> None:
    """MineRL models with the ImpalaCNN were pickled as `networks.ImpalaCNN` (a top-level module in
    the MineRL project); point that name at our copy so PPO.load can find the class."""
    from chop import networks
    sys.modules.setdefault("networks", networks)
