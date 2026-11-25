"""Training defaults, and the environment settings saved with every run (config.json and W&B)."""

from stable_baselines3.common.monitor import Monitor

from chop.envs import ENVS, common, minerl
from chop.envs import arena as arena_env
from chop.envs import natural as natural_env
from chop.infra.vec import game_folder

WANDB_PROJECT = "craftground-chop-trees"

TRAIN_FIELDS = ["started", "timesteps_done", "episodes", "minutes", "last20_avg_logs",
                "best_episode_logs", "status", "resumed_from", "model", "wandb_url"]

PPO_SETTINGS = dict(
    n_steps=2048,        # steps collected before each update, summed over all envs (~2 episodes)
    batch_size=256,
    n_epochs=4,
    learning_rate=2.5e-4,
    gamma=0.99,
    ent_coef=0.01,       # keeps the agent exploring
    clip_range=0.2,
)


def env_factory(env_name: str, port: int, monitor_file: str, shared_folder: bool = False, jump: bool = False,
                multi: bool = False, action_repeat: int = 1, spawn: str = "forest", use_minerl: bool = False,
                axe: str | None = None, stuck_penalty: float = 0.0, replay: float = 0.0):
    """Builds one Monitor-wrapped env. Called inside each SubprocVecEnv worker.
    jump: use the 13 actions with jump instead of 11 (natural world default; arena with --jump).
    multi: (move, turn, look) every step with attack held (common.MULTI_SETTINGS); jump adds a
    jump choice.
    action_repeat: arena: game ticks per agent step (arena.ChopTreesEnv); MineRL mode: ticks per decision
    (natural.NaturalChopTreesEnv's minerl_ticks, default minerl.FRAME_SKIP).
    spawn: natural only, where resets drop the player (natural.SPAWN_CHECKS).
    use_minerl: the MineRL project's view and actions (chop/envs/minerl.py); jump doesn't apply.
    axe: the item the player holds (None: the env's default, the one-hit axe, or iron in MineRL mode).
    stuck_penalty: natural only, reward off per stuck tick (natural.STUCK_WINDOW).
    replay: natural only, the share of resets that replay a saved hard spawn (natural.REPLAY_LOGS)."""
    def _make():
        make_env = ENVS[env_name].make_env
        actions = "minerl" if use_minerl else "multi" if multi else "single"
        if use_minerl:
            kwargs = {"actions": actions, **({"spawn": spawn} if env_name == "natural" else {})}
            if action_repeat != minerl.FRAME_SKIP:
                kwargs["minerl_ticks"] = action_repeat  # natural only (train.py checks)
        else:
            kwargs = ({"jump": jump, "actions": actions, "spawn": spawn} if env_name == "natural"
                      else {"actions": actions, "action_repeat": action_repeat, "jump": jump})
        if axe:
            kwargs["axe"] = axe
        if stuck_penalty:
            kwargs["stuck_penalty"] = stuck_penalty
        if replay:
            kwargs["replay"] = replay
        # Replay runs also log whether each episode was a replayed spawn.
        keys = (("logs_chopped", "stuck_ticks") + (("replay",) if replay else ())
                if env_name == "natural" else ("logs_chopped",))
        return Monitor(make_env(port=port, env_path=game_folder(port, shared_folder), **kwargs), filename=monitor_file,
                       info_keywords=keys)
    return _make


def env_settings(env_name: str, jump: bool = False, multi: bool = False, action_repeat: int = 1,
                 spawn: str = "forest", use_minerl: bool = False, axe: str | None = None,
                 stuck_penalty: float = 0.0, replay: float = 0.0) -> dict:
    """The environment's settings, saved in config.json (and W&B), so runs can be compared later."""
    shared = {
        "world": env_name,
        "episode_steps": common.EPISODE_STEPS,
        "image": f"{common.IMAGE_WIDTH}x{common.IMAGE_HEIGHT}",
        "actions": ((common.MULTI_SETTINGS_JUMP if jump else common.MULTI_SETTINGS) if multi
                    else common.ACTION_NAMES_JUMP if jump else common.ACTION_NAMES),
    }
    if use_minerl:
        # episode_steps stays in game ticks; an agent step is action_repeat of them (MineRL's 4 by default).
        # actions.frame_skip records it (chop.reports.agents.minerl_ticks reads it back).
        actions = dict(minerl.SETTINGS, frame_skip=action_repeat)
        if action_repeat != minerl.FRAME_SKIP:
            actions["turn_per_tick"] = f"yaw / pitch degrees / {minerl.FRAME_SKIP} (MineRL's turning speed)"
        shared.update(image=f"{minerl.OBS_W}x{minerl.OBS_H} (320x180 with hand, HUD cropped)",
                      actions=actions, mode="minerl", axe=axe or minerl.IRON_AXE)
    elif axe:
        shared["axe"] = axe
    if env_name == "natural":
        return {
            **shared,
            "reward": natural_env.REWARD,
            "seed": natural_env.SEED,
            # Forest runs save what every run before 2026-11-18 did; other spawns name their rule.
            **({"forest_check": natural_env.FOREST_CHECK} if spawn == "forest"
               else {"spawn": spawn, "spawn_checks": natural_env.SPAWN_CHECKS[spawn],
                     "max_hop_episodes": natural_env.MAX_HOP_EPISODES[spawn]}),
            "hop_range": natural_env.HOP_RANGE,
            "far_range": natural_env.FAR_RANGE,
            "render_distance": natural_env.RENDER_DISTANCE,
            "jvm_max_memory": natural_env.JVM_MAX_MEMORY,
            # Only runs with the penalty record it, so earlier configs stay as they were.
            **({"stuck_penalty": stuck_penalty, "stuck_window_ticks": natural_env.STUCK_WINDOW,
                "stuck_distance": natural_env.STUCK_DISTANCE} if stuck_penalty else {}),
            **({"replay": replay, "replay_logs_below": natural_env.REPLAY_LOGS,
                "replay_max_tries": natural_env.REPLAY_MAX_TRIES, "replay_buffer": natural_env.REPLAY_BUFFER,
                "replay_exclude_radius": natural_env.REPLAY_EXCLUDE_RADIUS} if replay else {}),
        }
    return {
        **shared,
        "reward": arena_env.REWARD,
        "trees": len(arena_env.TREE_POSITIONS),
        "logs_per_tree": arena_env.TRUNK_HEIGHT,
        "max_logs": arena_env.MAX_LOGS,
        "action_repeat": action_repeat,  # game ticks per agent step; episode_steps counts ticks
        "ground_repaired_on_reset": True,  # before 2026-11-17 dug holes stayed for the whole run
        "arena_radius": arena_env.ARENA_RADIUS,
    }
