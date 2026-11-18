"""What the arena and the natural world share: the agent's view, episode length, tool and actions."""

from craftground.environment.action_space import no_op, no_op_v2

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


# --- Multi-choice actions (train.py --actions multi) ---------------------------------------
# Every step the agent makes three choices at once (SB3 MultiDiscrete([3, 11, 9])) and always
# holds attack, so it only has to learn to walk and aim. Turns are exact degrees (CraftGround's
# V2 action format), not the 15-degree steps of the single actions above.
# Negative turn = left, negative look = up (Minecraft's pitch), same as turn_left / look_up.
# The natural world adds a 4th choice, jump yes/no (MULTI_NVEC_JUMP), for its hills and ledges.
MOVES = ["none", "forward", "back"]
TURN_DEGREES = [-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45]
LOOK_DEGREES = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
MULTI_NVEC = [len(MOVES), len(TURN_DEGREES), len(LOOK_DEGREES)]
MULTI_NVEC_JUMP = MULTI_NVEC + [2]
MULTI_SETTINGS = {"move": MOVES, "turn_degrees": TURN_DEGREES, "look_degrees": LOOK_DEGREES,
                  "attack": "always held"}
MULTI_SETTINGS_JUMP = {**MULTI_SETTINGS, "jump": ["no", "yes"]}


def multi_nvec(jump: bool) -> list[int]:
    return MULTI_NVEC_JUMP if jump else MULTI_NVEC


def to_craftground_multi(action) -> dict:
    """Turn a (move, turn, look[, jump]) choice into CraftGround's V2 (MineRL-style) action."""
    move, turn, look = (int(a) for a in action[:3])
    act = no_op_v2()
    act["attack"] = True
    act["forward"] = MOVES[move] == "forward"
    act["back"] = MOVES[move] == "back"
    act["camera_yaw"] = float(TURN_DEGREES[turn])
    act["camera_pitch"] = float(LOOK_DEGREES[look])
    act["jump"] = len(action) > 3 and bool(int(action[3]))
    return act


def multi_label(action) -> str:
    """Short text for one (move, turn, look[, jump]) choice, e.g. 'forward turn+5 look-2 jump'."""
    move, turn, look = (int(a) for a in action[:3])
    parts = [MOVES[move] if MOVES[move] != "none" else "stay"]
    if TURN_DEGREES[turn]:
        parts.append(f"turn{TURN_DEGREES[turn]:+d}")
    if LOOK_DEGREES[look]:
        parts.append(f"look{LOOK_DEGREES[look]:+d}")
    if len(action) > 3 and int(action[3]):
        parts.append("jump")
    return " ".join(parts)
