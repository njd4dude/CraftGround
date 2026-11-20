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
                multi: bool = False, action_repeat: int = 1, spawn: str = "forest", use_minerl: bool = False):
    """Builds one Monitor-wrapped env. Called inside each SubprocVecEnv worker.
    jump: use the 13 actions with jump instead of 11 (natural world default; arena with --jump).
    multi: (move, turn, look) every step with attack held (common.MULTI_SETTINGS); jump adds a
    jump choice.
    action_repeat: arena only, game ticks per agent step (arena.ChopTreesEnv).
    spawn: natural only, where resets drop the player (natural.SPAWN_CHECKS).
    use_minerl: the MineRL project's view and actions (chop/envs/minerl.py); jump / action_repeat don't apply."""
    def _make():
        make_env = ENVS[env_name].make_env
        actions = "minerl" if use_minerl else "multi" if multi else "single"
        if use_minerl:
            kwargs = {"actions": actions, **({"spawn": spawn} if env_name == "natural" else {})}
        else:
            kwargs = ({"jump": jump, "actions": actions, "spawn": spawn} if env_name == "natural"
                      else {"actions": actions, "action_repeat": action_repeat, "jump": jump})
        return Monitor(make_env(port=port, env_path=game_folder(port, shared_folder), **kwargs), filename=monitor_file,
                       info_keywords=("logs_chopped",))
    return _make


def env_settings(env_name: str, jump: bool = False, multi: bool = False, action_repeat: int = 1,
                 spawn: str = "forest", use_minerl: bool = False) -> dict:
    """The environment's settings, saved in config.json (docs/natural/ flags changes between runs)."""
    shared = {
        "world": env_name,
        "episode_steps": common.EPISODE_STEPS,
        "image": f"{common.IMAGE_WIDTH}x{common.IMAGE_HEIGHT}",
        "actions": ((common.MULTI_SETTINGS_JUMP if jump else common.MULTI_SETTINGS) if multi
                    else common.ACTION_NAMES_JUMP if jump else common.ACTION_NAMES),
    }
    if use_minerl:
        # episode_steps stays in game ticks; an agent step is minerl.FRAME_SKIP of them.
        shared.update(image=f"{minerl.OBS_W}x{minerl.OBS_H} (320x180 with hand, HUD cropped)",
                      actions=minerl.SETTINGS, mode="minerl", axe=minerl.IRON_AXE)
        action_repeat = minerl.FRAME_SKIP
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
