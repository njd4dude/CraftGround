"""Train a PPO agent to chop as many trees as possible.

    python train.py                       # train with defaults
    python train.py --timesteps 500000    # train longer
    python train.py --resume runs/chop_ppo_20261113_131800/checkpoints/ppo_chop_100000_steps.zip
    python train.py --no-wandb            # train without logging to Weights & Biases
    python train.py --learning-rate 1e-4 --n-epochs 8 --notes "slower lr, more epochs"

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
        monitor.monitor.csv    one line per training episode
        wandb/                 W&B's local copy of the run

Press Ctrl+C at any time: the current model is saved before exiting.
Each run is added to the Training runs table in EXPERIMENTS.md, with a link
to its W&B page.
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

import tree_env
from experiments import TRAIN_CSV, append_row, update_report
from tree_env import make_env

RUNS_DIR = Path("runs")
WANDB_PROJECT = "craftground-chop-trees"
TRAIN_FIELDS = ["started", "timesteps_done", "episodes", "minutes", "last20_avg_logs",
                "best_episode_logs", "status", "resumed_from", "model", "wandb_url"]

PPO_SETTINGS = dict(
    n_steps=2048,        # steps collected before each update (~2 episodes)
    batch_size=256,
    n_epochs=4,
    learning_rate=2.5e-4,
    gamma=0.99,
    ent_coef=0.01,       # keeps the agent exploring
    clip_range=0.2,
)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .zip to continue from")
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--name", type=str, default="chop_ppo", help="run folder prefix")
    parser.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
    parser.add_argument("--notes", type=str, default="", help="what's special about this run")
    for key, default in PPO_SETTINGS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=None,
                            help=f"PPO {key} (default {default})")
    args = parser.parse_args()
    # Only the flags given on the command line, so a resumed model keeps its own saved settings.
    overrides = {k: getattr(args, k) for k in PPO_SETTINGS if getattr(args, k) is not None}
    ppo_settings = {**PPO_SETTINGS, **overrides}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    started = datetime.now()
    run_dir = RUNS_DIR / f"{args.name}_{started:%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    print(f"Run folder: {run_dir}")

    run_config = {
        "run": run_dir.name,
        "started": started.isoformat(timespec="seconds"),
        "timesteps_target": args.timesteps,
        "resumed_from": args.resume or "",
        "notes": args.notes,
        "ppo": ppo_settings,
        "changed_from_defaults": {k: {"default": PPO_SETTINGS[k], "used": v}
                                  for k, v in overrides.items() if v != PPO_SETTINGS[k]},
        "env": {
            "reward": tree_env.REWARD,
            "episode_steps": tree_env.EPISODE_STEPS,
            "image": f"{tree_env.IMAGE_WIDTH}x{tree_env.IMAGE_HEIGHT}",
            "trees": len(tree_env.TREE_POSITIONS),
            "logs_per_tree": tree_env.TRUNK_HEIGHT,
            "max_logs": tree_env.MAX_LOGS,
            "arena_radius": tree_env.ARENA_RADIUS,
            "actions": tree_env.ACTION_NAMES,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    wandb.init(
        project=WANDB_PROJECT,
        name=run_dir.name,
        dir=str(run_dir),
        notes=args.notes or None,
        mode="disabled" if args.no_wandb else "online",
        config={
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "timesteps": args.timesteps,
            "resumed_from": args.resume,
            **ppo_settings,
            "env/reward": tree_env.REWARD,
            "env/image": f"{tree_env.IMAGE_WIDTH}x{tree_env.IMAGE_HEIGHT}",
            "env/episode_steps": tree_env.EPISODE_STEPS,
            "env/trees": len(tree_env.TREE_POSITIONS),
            "env/max_logs": tree_env.MAX_LOGS,
            "env/actions": tree_env.ACTION_NAMES,
        },
    )
    # Plot everything against training steps.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb_url = wandb.run.url if wandb.run is not None and not args.no_wandb else ""

    env = Monitor(make_env(port=args.port), filename=str(run_dir / "monitor"), info_keywords=("logs_chopped",))

    if args.resume:
        model = PPO.load(args.resume, env=env, device=device, **overrides)
        print(f"Resumed from {args.resume}" + (f" with {overrides}" if overrides else ""))
    else:
        model = PPO("CnnPolicy", env, device=device, verbose=1, **ppo_settings)
    model.set_logger(Logger(folder=str(run_dir), output_formats=[HumanOutputFormat(sys.stdout), WandbOutputFormat()]))

    chop_stats = LogsChoppedCallback()
    callbacks = [
        chop_stats,
        CheckpointCallback(save_freq=20_000, save_path=str(run_dir / "checkpoints"), name_prefix="ppo_chop"),
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
        env.close()

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

        append_row(TRAIN_CSV, TRAIN_FIELDS, {
            "started": started.isoformat(timespec="seconds"),
            **{k: ("" if v is None else v) for k, v in summary.items()},
            "resumed_from": args.resume or "",
            "model": model_path.as_posix(),
            "wandb_url": wandb_url,
        })
        update_report()


if __name__ == "__main__":
    main()
