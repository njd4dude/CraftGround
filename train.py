"""Train a PPO agent to chop as many trees as possible.

    python train.py                       # train with defaults
    python train.py --timesteps 500000    # train longer
    python train.py --resume runs/chop_ppo_20261113_131800/checkpoints/ppo_chop_100000_steps.zip
    python train.py --no-wandb            # train without logging to Weights & Biases
    python train.py --learning-rate 1e-4 --n-epochs 8 --notes "slower lr, more epochs"
    python train.py --n-envs 4            # run 4 Minecraft instances instead of 6
    python train.py --env natural --n-envs 4                 # train in a forest in a normal world
    python train.py --env natural --resume runs/chop_ppo_20261113_135620/model.zip   # fine-tune an arena model there
    python train.py --env natural --no-jump                  # natural world with the arena's 11 actions (no jump)

--env picks the world: arena (tree_env.py, walled superflat arena, the default)
or natural (natural_env.py, a fresh forest spot every episode). Both have the
same images, but the natural world adds jump and forward_jump (13 actions, see
tree_env.JUMP_ACTIONS) because of its hills and ledges. 11-action models still --resume
there (jump is turned off automatically to match); --no-jump forces the 11 actions.
Without jump the two worlds share the same actions, so --resume works across them. Natural runs get their
own folders (chop_ppo_natural_...), CSV (results/training_runs_natural.csv) and
report (EXPERIMENTS_NATURAL.md). The natural world is slower (about a third of
the arena's steps/s in evaluation) and needs more memory per instance, so
--n-envs 4 is a safer start than 6 there.

Training runs --n-envs Minecraft instances side by side (default 6). Each needs
~2.4 GB of RAM (~2.8 GB in the natural world) and a lot of CPU. Measured raw env
speed on this machine (8-core 7700X, BENCHMARK.md): arena 1 -> 487, 2 -> 713,
4 -> 977, 6 -> 1046 steps/s; natural 4 -> 235 steps/s with the current 0.75 s
forest-check timeout (1 -> 48, 2 -> 120, 6 -> 263 were measured with the old 3 s
timeout, so they understate it). Training runs ~20% below these raw speeds
(model inference each step, PPO updates): 4 natural instances trained at ~173.

Each instance runs from its own copy of CraftGround's game folder
(~/craftground_instances/mc121_port<port>, see instances.py), because instances
started together in CraftGround's one shared folder sometimes collide and one
fails to start, which hangs the whole run. --shared-folder goes back to the
shared folder. The first run on a new port is ~75 s slower while its copy builds.
--n-steps is the rollout size summed over all instances, so changing --n-envs
keeps each PPO update the same size (each instance collects n_steps / n_envs).

Every PPO setting in PPO_SETTINGS has a flag (--n-steps, --batch-size, --n-epochs,
--learning-rate, --gamma, --ent-coef, --clip-range). Use --notes to say what's
special about a run (e.g. a reward change); it shows up in EXPERIMENTS.md.

Live graphs (logs chopped per episode, reward, losses) go to Weights & Biases,
project "craftground-chop-trees". Log in once first with:  wandb login

Every run gets one folder with everything in it:

    runs/chop_ppo_20261113_131800/
        config.json            this run's settings, changes from the defaults, notes
        model.zip              final model (saved on finish, Ctrl+C or crash)
        checkpoints/           a model every 20,000 steps
        monitor_0.monitor.csv  one line per training episode (one file per instance)
        wandb/                 W&B's local copy of the run

Press Ctrl+C at any time: the current model is saved before exiting.
Each run is added to the Training runs table in EXPERIMENTS.md (or
EXPERIMENTS_NATURAL.md), with a link to its W&B page.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

import natural_env
import tree_env
from experiments import NATURAL_TRAIN_CSV, TRAIN_CSV, append_row, update_report
from instances import clear_stale_loom_locks, close_vec_env, env_path_for, kill_instance_processes

ENVS = {"arena": tree_env, "natural": natural_env}

RUNS_DIR = Path("runs")
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


def env_factory(env_name: str, port: int, monitor_file: str, shared_folder: bool = False, jump: bool = False):
    """Builds one Monitor-wrapped env. Called inside each SubprocVecEnv worker.
    jump: natural world only, use the 13 actions with jump instead of the arena's 11."""
    def _make():
        make_env = ENVS[env_name].make_env
        env_path = None if shared_folder else env_path_for(port, verbose=True)
        kwargs = {"jump": jump} if env_name == "natural" else {}
        return Monitor(make_env(port=port, env_path=env_path, **kwargs), filename=monitor_file,
                       info_keywords=("logs_chopped",))
    return _make


def env_settings(env_name: str, jump: bool = False) -> dict:
    """The environment's settings, saved in config.json (EXPERIMENTS.md flags changes between runs)."""
    common = {
        "world": env_name,
        "episode_steps": tree_env.EPISODE_STEPS,
        "image": f"{tree_env.IMAGE_WIDTH}x{tree_env.IMAGE_HEIGHT}",
        "actions": tree_env.ACTION_NAMES_JUMP if jump else tree_env.ACTION_NAMES,
    }
    if env_name == "natural":
        return {
            **common,
            "reward": natural_env.REWARD,
            "seed": natural_env.SEED,
            "forest_check": natural_env.FOREST_CHECK,
            "hop_range": natural_env.HOP_RANGE,
            "far_range": natural_env.FAR_RANGE,
        }
    return {
        **common,
        "reward": tree_env.REWARD,
        "trees": len(tree_env.TREE_POSITIONS),
        "logs_per_tree": tree_env.TRUNK_HEIGHT,
        "max_logs": tree_env.MAX_LOGS,
        "arena_radius": tree_env.ARENA_RADIUS,
    }


class WandbOutputFormat(KVWriter):
    """Sends SB3's training stats (losses, rewards, fps...) to W&B instead of TensorBoard."""

    def write(self, key_values, key_excluded, step=0):
        data = {k: v for k, v in key_values.items() if isinstance(v, (int, float, np.number))}
        if data and wandb.run is not None:
            wandb.log({"global_step": step, **data})

    def close(self):
        pass


class LogsChoppedCallback(BaseCallback):
    """Logs every finished episode's log count to W&B and the console."""

    def __init__(self):
        super().__init__()
        self.recent: list[int] = []
        self.episodes = 0
        self.best = 0

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                logs = info.get("logs_chopped", 0)
                self.episodes += 1
                self.best = max(self.best, logs)
                self.recent = (self.recent + [logs])[-20:]
                avg = float(np.mean(self.recent))
                if wandb.run is not None:
                    wandb.log({
                        "global_step": self.num_timesteps,
                        "chop/logs_per_episode": logs,
                        "chop/logs_per_episode_avg20": avg,
                        "chop/episode": self.episodes,
                    })
                print(f"[{self.num_timesteps:>8} steps] episode done: {logs:2d} logs (last-20 avg {avg:.1f})")
        return True


class StepTimer(VecEnvWrapper):
    """Adds up the wall time spent inside env.step (all Minecraft instances, including resets)."""

    def __init__(self, venv):
        super().__init__(venv)
        self.seconds = 0.0
        self._t = 0.0

    def step_async(self, actions):
        self._t = time.perf_counter()
        self.venv.step_async(actions)

    def step_wait(self):
        result = self.venv.step_wait()
        self.seconds += time.perf_counter() - self._t
        return result

    def reset(self):
        return self.venv.reset()


class SpeedCallback(BaseCallback):
    """Logs current training speed per PPO iteration (SB3's time/fps averages over the whole run).

    time/env_fps              env steps/s counting only time inside env.step: pure Minecraft speed
    time/rollout_fps          env steps/s while collecting, summed over all instances
                              (Minecraft + policy inference + callbacks, incl. checkpoint saves)
    time/rollout_fps_per_env  rollout_fps / n_envs (one instance's effective speed)
    time/iter_fps             env steps/s over rollout + PPO update (real training throughput);
                              measured start-to-start, so it describes the previous iteration
    """

    def __init__(self, n_envs: int, step_timer: StepTimer):
        super().__init__()
        self.n_envs = n_envs
        self.step_timer = step_timer
        self.rollout_start = None
        self.rollout_start_steps = 0

    def _on_rollout_start(self) -> None:
        now = time.perf_counter()
        if self.rollout_start is not None:
            steps = self.num_timesteps - self.rollout_start_steps
            self.logger.record("time/iter_fps", steps / (now - self.rollout_start))
        self.rollout_start = now
        self.rollout_start_steps = self.num_timesteps
        self.step_timer.seconds = 0.0

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps - self.rollout_start_steps
        if self.step_timer.seconds > 0:
            self.logger.record("time/env_fps", steps / self.step_timer.seconds)
        fps = steps / (time.perf_counter() - self.rollout_start)
        self.logger.record("time/rollout_fps", fps)
        self.logger.record("time/rollout_fps_per_env", fps / self.n_envs)

    def _on_step(self) -> bool:
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=sorted(ENVS), default="arena",
                        help="arena = walled superflat arena (tree_env.py); natural = forest in a normal world (natural_env.py)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .zip to continue from")
    parser.add_argument("--port", type=int, default=8023, help="first instance's port (others go up by 10)")
    parser.add_argument("--n-envs", type=int, default=6, help="Minecraft instances to run in parallel")
    parser.add_argument("--shared-folder", action="store_true",
                        help="run all instances from CraftGround's one shared game folder (old behaviour)")
    parser.add_argument("--name", type=str, default=None,
                        help="run folder prefix (default chop_ppo, or chop_ppo_natural with --env natural)")
    parser.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
    parser.add_argument("--notes", type=str, default="", help="what's special about this run")
    parser.add_argument("--no-jump", action="store_true",
                        help="natural world: use the arena's 11 actions instead of 13 with jump "
                             "(automatic when --resume is an 11-action model)")
    for key, default in PPO_SETTINGS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=None,
                            help=f"PPO {key} (default {default})")
    args = parser.parse_args()
    # Only the flags given on the command line, so a resumed model keeps its own saved settings.
    overrides = {k: getattr(args, k) for k in PPO_SETTINGS if getattr(args, k) is not None}
    ppo_settings = {**PPO_SETTINGS, **overrides}
    saved = load_from_zip_file(args.resume, load_data=True, device="cpu")[0] if args.resume else None
    if saved and "n_steps" not in overrides:
        # Keep the resumed model's own rollout size (stored per env, so multiply back up).
        ppo_settings["n_steps"] = saved["n_steps"] * saved.get("n_envs", 1)
    # Natural world: 13 actions with jump unless --no-jump; a resumed model keeps the set it was trained with.
    jump = args.env == "natural" and not args.no_jump
    if saved and args.env == "natural":
        jump = saved["action_space"].n == len(tree_env.ACTION_NAMES_JUMP)
        if args.no_jump and jump:
            parser.error("--no-jump: the --resume model was trained with jump (13 actions)")
    elif saved and saved["action_space"].n != len(tree_env.ACTION_NAMES):
        parser.error(f"--resume model has {saved['action_space'].n} actions; the {args.env} world has "
                     f"{len(tree_env.ACTION_NAMES)} (jump models only run in the natural world)")
    # SB3's n_steps is per env; ours is the total, so split it across the instances.
    steps_per_env = max(ppo_settings["n_steps"] // args.n_envs, 1)
    sb3_settings = {**ppo_settings, "n_steps": steps_per_env}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    name = args.name or ("chop_ppo" if args.env == "arena" else f"chop_ppo_{args.env}")
    started = datetime.now()
    run_dir = RUNS_DIR / f"{name}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"Run folder: {run_dir}")

    run_config = {
        "run": run_dir.name,
        "started": started.isoformat(timespec="seconds"),
        "timesteps_target": args.timesteps,
        "resumed_from": args.resume or "",
        "notes": args.notes,
        "n_envs": args.n_envs,
        "rollout_steps": steps_per_env * args.n_envs,
        "ppo": ppo_settings,
        "changed_from_defaults": {k: {"default": PPO_SETTINGS[k], "used": v}
                                  for k, v in overrides.items() if v != PPO_SETTINGS[k]},
        "env": env_settings(args.env, jump),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    wandb.init(
        project=WANDB_PROJECT,
        name=run_dir.name,
        dir=str(run_dir),
        notes=args.notes or None,
        tags=[args.env],
        mode="disabled" if args.no_wandb else "online",
        config={
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "timesteps": args.timesteps,
            "resumed_from": args.resume,
            "n_envs": args.n_envs,
            **ppo_settings,
            **{f"env/{k}": v for k, v in env_settings(args.env, jump).items()},
        },
    )
    # Plot everything against training steps.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb_url = wandb.run.url if wandb.run is not None and not args.no_wandb else ""

    ports = [args.port + 10 * i for i in range(args.n_envs)]
    if not args.shared_folder:
        kill_instance_processes(ports, verbose=True)  # leftovers of a killed run would block these ports
    clear_stale_loom_locks()  # also needed with --shared-folder; see instances.py
    factories = [env_factory(args.env, args.port + 10 * i, str(run_dir / f"monitor_{i}"), args.shared_folder, jump)
                 for i in range(args.n_envs)]
    step_timer = StepTimer(SubprocVecEnv(factories) if args.n_envs > 1 else DummyVecEnv(factories))
    env = step_timer
    print(f"{args.env} world, {args.n_envs} Minecraft instance(s), {steps_per_env} steps each per update "
          f"({steps_per_env * args.n_envs} total), {env.action_space.n} actions{' (with jump)' if jump else ''}")

    if args.resume:
        model = PPO.load(args.resume, env=env, device=device, **{**overrides, "n_steps": steps_per_env})
        print(f"Resumed from {args.resume}" + (f" with {overrides}" if overrides else ""))
    else:
        model = PPO("CnnPolicy", env, device=device, verbose=1, **sb3_settings)
    model.set_logger(Logger(folder=str(run_dir), output_formats=[HumanOutputFormat(sys.stdout), WandbOutputFormat()]))

    chop_stats = LogsChoppedCallback()
    callbacks = [
        chop_stats,
        SpeedCallback(args.n_envs, step_timer),
        # save_freq counts vec-env steps, each of which is n_envs env steps.
        CheckpointCallback(save_freq=max(20_000 // args.n_envs, 1), save_path=str(run_dir / "checkpoints"), name_prefix="ppo_chop"),
    ]

    t0 = time.time()
    status = "crashed"
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=args.resume is None,
        )
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        print("\nInterrupted, saving model...")
    finally:
        model_path = run_dir / "model.zip"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        close_vec_env(env, ports, args.shared_folder)

        summary = {
            "timesteps_done": model.num_timesteps,
            "episodes": chop_stats.episodes,
            "minutes": round((time.time() - t0) / 60, 1),
            "last20_avg_logs": round(float(np.mean(chop_stats.recent)), 2) if chop_stats.recent else None,
            "best_episode_logs": chop_stats.best,
            "status": status,
        }
        if wandb.run is not None:
            wandb.run.summary.update(summary)
            wandb.finish(exit_code=0 if status != "crashed" else 1)

        append_row(NATURAL_TRAIN_CSV if args.env == "natural" else TRAIN_CSV, TRAIN_FIELDS, {
            "started": started.isoformat(timespec="seconds"),
            **{k: ("" if v is None else v) for k, v in summary.items()},
            "resumed_from": args.resume or "",
            "model": model_path.as_posix(),
            "wandb_url": wandb_url,
        })
        update_report(args.env)


if __name__ == "__main__":
    main()
