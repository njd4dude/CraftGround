"""train.py's command line (see train.py for usage)."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import HumanOutputFormat, Logger
from stable_baselines3.common.save_util import load_from_zip_file

from chop.envs import ENVS, common, minerl
from chop.infra.vec import DEFAULT_N_ENVS, close_vec_env, instance_ports, launch_vec_env
from chop.paths import RUNS_DIR
from chop.reports import update_report
from chop.reports.agents import run_action_repeat
from chop.reports.csvio import MINERL_TRAIN_CSV, NATURAL_TRAIN_CSV, TRAIN_CSV, append_row
from chop.training.callbacks import LogsChoppedCallback, SpeedCallback, StepTimer, ValueWarmup, WandbOutputFormat
from chop.training.lineage import group_of, history, lineage_tag, tag_runs
from chop.training.lines import natural_line_dir
from chop.training.schedule import linear_lr
from chop.training.settings import PPO_SETTINGS, TRAIN_FIELDS, WANDB_PROJECT, env_factory, env_settings


def run_dir_name(model: str) -> str:
    """The run folder name of a model path (model.zip, final.zip or a checkpoint), trimmed of its date."""
    p = Path(model)
    run = p.parent.parent if p.parent.name == "checkpoints" else p.parent
    return run.name.removeprefix("model_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=sorted(ENVS), default="arena",
                        help="arena = walled superflat arena (chop/envs/arena.py); natural = forest in a normal world (chop/envs/natural.py)")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint .zip to continue from")
    parser.add_argument("--port", type=int, default=8023, help="first instance's port (others go up by 10)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Minecraft instances to run in parallel (default: 6 in the arena, 3 in the natural world)")
    parser.add_argument("--shared-folder", action="store_true",
                        help="run all instances from CraftGround's one shared game folder (old behaviour)")
    parser.add_argument("--name", type=str, default=None,
                        help="arena: run folder prefix (default chop_ppo)")
    parser.add_argument("--line", type=str, default=None,
                        help="natural world: line folder under runs/natural/ (default: the --resume model's "
                             "line, else a new lineN). Give a new name to branch, e.g. line4_adventure-mode")
    parser.add_argument("--no-wandb", action="store_true", help="don't log to Weights & Biases")
    parser.add_argument("--notes", type=str, default="", help="what's special about this run")
    parser.add_argument("--no-jump", action="store_true",
                        help="natural world: use the arena's 11 actions instead of 13 with jump "
                             "(automatic when --resume is an 11-action model)")
    parser.add_argument("--jump", action="store_true",
                        help="arena: offer the natural world's jump actions (13 single, or a 4th jump choice in "
                             "multi) so the model can later be --resume'd in the natural world. "
                             "A resumed model keeps its own")
    parser.add_argument("--actions", choices=["single", "multi", "minerl"], default=None,
                        help="single (default): one action per step; multi: move + turn + look (+ jump in the "
                             "natural world or with --jump) every step with attack always held; minerl: the "
                             "MineRL project's view and actions (chop/envs/minerl.py). A resumed model keeps its own "
                             "(MineRL-trained models are detected), so --resume runs/minerl/<run>/final.zip fine-tunes one")
    parser.add_argument("--axe", choices=["default", "iron", "instant"], default="default",
                        help="the axe the player holds: instant (one hit per log, every run before 2026-11-20) or "
                             "iron (MineRL's). Default: iron in MineRL mode, instant otherwise")
    parser.add_argument("--value-warmup", type=int, default=0, metavar="N",
                        help="train only the value head for the model's first N PPO updates, policy frozen: for "
                             "starting PPO from a behaviour-cloned model, whose value head was never trained "
                             "(the MineRL project used 5). Counted from the model's own update count, so a "
                             "model that already had N updates skips it")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="PPO target_kl: stop an update's epochs early once the policy has moved this far "
                             "(default: none, or the resumed model's own; MineRL fine-tunes used 0.05)")
    parser.add_argument("--action-repeat", type=int, default=None,
                        help="arena only: game ticks per agent step (default 1; 4 lets each step show its own "
                             "action, see tools/check_timing.py). A resumed model keeps its own unless given")
    parser.add_argument("--spawn", choices=sorted(ENVS["natural"].SPAWNS), default="forest",
                        help="natural world: where resets drop the player. forest (default): oak/birch "
                             "forests; wooded: any biome with trees, snowy ones included")
    parser.add_argument("--lr-schedule", choices=["linear", "constant"], default="linear",
                        help="linear (default): --learning-rate falls to 0 by the end of this run; "
                             "constant: stays fixed (all runs before 2026-11-15)")
    for key, default in PPO_SETTINGS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(default), default=None,
                            help=f"PPO {key} (default {default})")
    args = parser.parse_args()
    if args.n_envs is None:
        args.n_envs = DEFAULT_N_ENVS[args.env]
    # Only the flags given on the command line, so a resumed model keeps its own saved settings.
    overrides = {k: getattr(args, k) for k in PPO_SETTINGS if getattr(args, k) is not None}
    ppo_settings = {**PPO_SETTINGS, **overrides}
    minerl.register_networks()  # MineRL models with the ImpalaCNN need it to load
    saved = load_from_zip_file(args.resume, load_data=True, device="cpu")[0] if args.resume else None
    use_minerl = args.actions == "minerl"
    if saved:
        saved_minerl = (tuple(saved["observation_space"].shape) in ((3, minerl.OBS_H, minerl.OBS_W), minerl.OBS_SHAPE)
                        and list(getattr(saved["action_space"], "nvec", [])) == minerl.NVEC)
        if args.actions is not None and use_minerl != saved_minerl:
            parser.error(f"--actions {args.actions}: the --resume model "
                         f"{'was' if saved_minerl else 'was not'} trained in MineRL's spaces")
        use_minerl = saved_minerl
    iron = args.axe == "iron" or (args.axe == "default" and use_minerl)
    # None for a normal run with the default axe, so its config.json stays as before.
    axe = (minerl.IRON_AXE if iron else common.AXE) if (use_minerl or args.axe != "default") else None
    if use_minerl and (args.jump or args.no_jump or args.action_repeat):
        parser.error("MineRL mode has its own jump (forward+jump) and 4-tick frame skip")
    if use_minerl and saved and "learning_rate" not in overrides:
        # The linear schedule starts from the model's own rate (MineRL fine-tunes: 1e-4), not ours.
        ppo_settings["learning_rate"] = float(saved["learning_rate"]) if isinstance(
            saved["learning_rate"], float) else 1e-4
    if args.target_kl is not None:
        overrides["target_kl"] = ppo_settings["target_kl"] = args.target_kl
    if saved and "n_steps" not in overrides:
        # Keep the resumed model's own rollout size (stored per env, so multiply back up).
        ppo_settings["n_steps"] = saved["n_steps"] * saved.get("n_envs", 1)
    if saved:
        # A resumed model keeps its own PPO settings unless a flag overrides them, so record those
        # (config.json / W&B), not our defaults: a MineRL model keeps batch 64, 10 epochs, ent 0.001.
        # learning_rate is left alone: the linear schedule restarts from ppo_settings' value.
        for key in ("batch_size", "n_epochs", "gamma", "ent_coef"):
            if key not in overrides and isinstance(saved.get(key), (int, float)):
                ppo_settings[key] = saved[key]
        if "target_kl" not in overrides and saved.get("target_kl") is not None:
            ppo_settings["target_kl"] = saved["target_kl"]
    # Multi actions: from --actions, or the resumed model's own action space.
    multi = args.actions == "multi"
    if saved and not use_minerl:
        saved_multi = isinstance(saved["action_space"], gym.spaces.MultiDiscrete)
        if args.actions is not None and multi != saved_multi:
            parser.error(f"--actions {args.actions}: the --resume model uses "
                         f"{'multi' if saved_multi else 'single'} actions")
        multi = saved_multi
    action_repeat = (minerl.FRAME_SKIP if use_minerl
                     else args.action_repeat or (run_action_repeat(args.resume) if args.resume else 1))
    if action_repeat < 1:
        parser.error("--action-repeat must be at least 1")
    if action_repeat > 1 and args.env != "arena" and not use_minerl:
        parser.error("--action-repeat is arena only for now")
    # Jump: on in the natural world unless --no-jump, off in the arena unless --jump. A resumed
    # model keeps the set it was trained with, in either world, so arena models can move to the
    # natural world (and back) as long as they were trained with the same set.
    if args.jump and args.env != "arena":
        parser.error("--jump is for the arena; the natural world has jump unless --no-jump")
    if args.no_jump and args.env != "natural":
        parser.error("--no-jump is for the natural world; the arena has no jump unless --jump")
    if args.spawn != "forest" and args.env != "natural":
        parser.error("--spawn is for the natural world")
    jump = args.jump if args.env == "arena" else not args.no_jump
    if use_minerl:
        jump = False  # its move choice has forward+jump
    elif saved:
        # A multi model's jump choice is its 4th dimension; a single model's is 13 actions instead of 11.
        jump = (len(saved["action_space"].nvec) == len(common.MULTI_NVEC_JUMP) if multi
                else saved["action_space"].n == len(common.ACTION_NAMES_JUMP))
        if args.no_jump and jump:
            parser.error("--no-jump: the --resume model was trained with jump")
        if args.jump and not jump:
            parser.error("--jump: the --resume model was trained without jump")
    # SB3's n_steps is per env; ours is the total, so split it across the instances.
    steps_per_env = max(ppo_settings["n_steps"] // args.n_envs, 1)
    sb3_settings = {**ppo_settings, "n_steps": steps_per_env}
    start_steps = int(saved["num_timesteps"]) if saved else 0
    if args.lr_schedule == "linear":
        # A resumed model restarts the schedule from --learning-rate (default 2.5e-4), not from
        # wherever its own (constant) rate was.
        lr = linear_lr(ppo_settings["learning_rate"], start_steps, start_steps + args.timesteps)
        sb3_settings["learning_rate"] = lr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    started = datetime.now()
    if use_minerl:
        # MineRL-mode runs (fine-tunes and from-scratch runs in MineRL's spaces) are kept apart from
        # the natural lines and the arena runs: different view, actions and frame skip.
        default = ("ft_" + run_dir_name(args.resume)) if args.resume else "scratch"
        run_dir = RUNS_DIR / "minerl_mode" / args.env / f"{args.name or default}_{started:%Y%m%d_%H%M%S}"
        run_name = f"minerl_mode/{args.env}/{run_dir.name}"
    elif args.env == "natural":
        line_dir = natural_line_dir(args.resume, args.line)
        run_dir = line_dir / f"{started:%Y%m%d_%H%M%S}"
        run_name = f"{line_dir.name}/{run_dir.name}"
    else:
        run_dir = RUNS_DIR / "arena" / f"{args.name or ('chop_ppo_multi' if multi else 'chop_ppo')}_{started:%Y%m%d_%H%M%S}"
        run_name = run_dir.name
    run_dir.mkdir(parents=True)
    print(f"Run folder: {run_dir}")

    run_config = {
        "run": run_dir.relative_to(RUNS_DIR).as_posix(),
        "started": started.isoformat(timespec="seconds"),
        "timesteps_target": args.timesteps,
        "resumed_from": args.resume or "",
        "notes": args.notes,
        "n_envs": args.n_envs,
        "rollout_steps": steps_per_env * args.n_envs,
        "ppo": ppo_settings,
        "lr_schedule": args.lr_schedule,
        "value_warmup_updates": args.value_warmup,
        "changed_from_defaults": {k: {"default": PPO_SETTINGS.get(k), "used": v}  # target_kl has no default here
                                  for k, v in overrides.items() if v != PPO_SETTINGS.get(k)},
        "env": env_settings(args.env, jump, multi, action_repeat, args.spawn, use_minerl, axe),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Group the line's pieces on W&B, and tag this run and every run it continues with its lineage.
    chain = history(args.resume)
    group, lineage = group_of(run_dir, chain), lineage_tag(run_dir, chain)
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=group,
        dir=str(run_dir),
        notes=args.notes or None,
        tags=[args.env, *(["minerl-mode"] if use_minerl else []), *([lineage] if lineage else [])],
        mode="disabled" if args.no_wandb else "online",
        config={
            "algorithm": "PPO",
            "policy": "CnnPolicy",
            "timesteps": args.timesteps,
            "resumed_from": args.resume,
            "n_envs": args.n_envs,
            "lr_schedule": args.lr_schedule,
            "actions": "minerl" if use_minerl else "multi" if multi else "single",
            **ppo_settings,
            **{f"env/{k}": v for k, v in env_settings(args.env, jump, multi, action_repeat, args.spawn,
                                                      use_minerl, axe).items()},
        },
    )
    # Plot everything against training steps.
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    wandb_url = wandb.run.url if wandb.run is not None and not args.no_wandb else ""
    if wandb_url and chain:
        try:
            tag_runs(chain, lineage, group)
        except Exception as e:  # never stop training over a tag
            print(f"Couldn't tag the earlier runs on W&B ({e}); tools/wandb_lineage.py can")

    ports = instance_ports(args.port, args.n_envs)
    factories = [env_factory(args.env, port, str(run_dir / f"monitor_{i}"), args.shared_folder, jump, multi,
                             1 if use_minerl else action_repeat, args.spawn, use_minerl, axe)
                 for i, port in enumerate(ports)]
    step_timer = StepTimer(launch_vec_env(factories, ports, args.shared_folder))
    env = step_timer
    print(f"{args.env} world, {args.n_envs} Minecraft instance(s), {steps_per_env} steps each per update "
          f"({steps_per_env * args.n_envs} total), "
          + ("MineRL actions: move x yaw x pitch 3x11x9, attack held" if use_minerl else
             "multi actions: move x turn x look" + (" x jump " if jump else " ")
             + "x".join(map(str, common.multi_nvec(jump))) + ", attack held" if multi
             else f"{env.action_space.n} actions{' (with jump)' if jump else ''}")
          + (f", {action_repeat} ticks per step" if action_repeat > 1 else "")
          + f", {'iron' if iron else 'one-hit'} axe")

    if args.resume:
        load_settings = {**overrides, "n_steps": steps_per_env}
        if args.lr_schedule == "linear":
            load_settings["learning_rate"] = sb3_settings["learning_rate"]
        model = PPO.load(args.resume, env=env, device=device, **load_settings)
        if use_minerl:
            # A MineRL model's saved episode stats are from MineRL (4000-tick episodes, up to 64 logs);
            # without this the first ~100 episodes' ep_rew_mean here would be averaged with them.
            model.ep_info_buffer.clear()
            model.ep_success_buffer.clear()
        print(f"Resumed from {args.resume}" + (f" with {overrides}" if overrides else ""))
    else:
        model = PPO("CnnPolicy", env, device=device, verbose=1, **sb3_settings)
    model.set_logger(Logger(folder=str(run_dir), output_formats=[HumanOutputFormat(sys.stdout), WandbOutputFormat()]))

    chop_stats = LogsChoppedCallback(action_repeat)
    callbacks = [
        *([ValueWarmup(args.value_warmup)] if args.value_warmup else []),
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

        train_csv = MINERL_TRAIN_CSV if use_minerl else NATURAL_TRAIN_CSV if args.env == "natural" else TRAIN_CSV
        append_row(train_csv, TRAIN_FIELDS, {
            "started": started.isoformat(timespec="seconds"),
            **{k: ("" if v is None else v) for k, v in summary.items()},
            "resumed_from": args.resume or "",
            "model": model_path.as_posix(),
            "wandb_url": wandb_url,
        })
        update_report("minerl" if use_minerl else args.env)
