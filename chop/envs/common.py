"""What the arena and the natural world share: the agent's view, episode length, tool and actions."""

from craftground.environment.action_space import no_op

IMAGE_WIDTH = 114
IMAGE_HEIGHT = 64
EPISODE_STEPS = 1000  # one step = one game tick (1/20 s), so ~50 s of game time

# Efficiency X breaks a log in one tick, so a single "attack" step is enough.
AXE = (
    "minecraft:diamond_axe[minecraft:unbreakable={},"
    'minecraft:enchantments={levels:{"minecraft:efficiency":10}}]'
)

CAMERA_STEP = 1  # camera actions turn by CAMERA_STEP * 15 degrees

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

# Extra actions for worlds with terrain (the natural world uses them by default; the flat arena
# doesn't need them). They go after ACTION_NAMES so actions 0-10 mean the same thing in both
# sets, but a model trained with one set can't be run with the other (different output size).
JUMP_ACTIONS = ["jump", "forward_jump"]
ACTION_NAMES_JUMP = ACTION_NAMES + JUMP_ACTIONS


def action_names(n_actions: int) -> list[str]:
    """The action list a model with n_actions outputs was trained with."""
    return ACTION_NAMES_JUMP if n_actions == len(ACTION_NAMES_JUMP) else ACTION_NAMES


def to_craftground_action(index: int) -> list[int]:
    """Turn one of ACTION_NAMES_JUMP (ACTION_NAMES is its first 11) into CraftGround's
    8-number MineDojo-style action."""
    name = ACTION_NAMES_JUMP[index]
    act = no_op()  # [move, strafe, jump, pitch=12, yaw=12, functional, ...]
    if name in ("jump", "forward_jump"):
        act[2] = 1  # slot 2: 0 none, 1 jump, 2 sneak, 3 sprint
    if name in ("forward", "forward_attack", "forward_jump"):
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
