"""evaluate.py's command line and play loop (see evaluate.py for usage)."""

import argparse
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from chop.envs import ENVS
from chop.envs import natural as natural_env
from chop.envs.common import ACTION_NAMES, MULTI_NVEC_JUMP, action_names, multi_label
from chop.evaluation.results import episode_shares, save_results, timing_summary
from chop.evaluation.video import RECORD_PER_GROUP, VIDEO_FPS, EpisodeRecorder
from chop.infra.vec import DEFAULT_N_ENVS, close_vec_env, game_folder, instance_ports, launch_vec_env
from chop.reports.agents import run_action_repeat
from chop.reports.csvio import normalize_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="path to a saved model .zip")
    parser.add_argument("--random", action="store_true", help="use random actions instead of a model")
    parser.add_argument("--actions", choices=["single", "multi"], default="single",
                        help="--random only: the action set to play randomly in (a model uses its own). "
                             "multi = move + turn + look (+ jump in the natural world) every step, attack held. "
                             "Models trained with jump can run in either world")
    parser.add_argument("--action-repeat", type=int, default=None,
                        help="arena: game ticks per agent step (default: the model's own from its run's "
                             "config.json; 1 for --random)")
    parser.add_argument("--episodes", type=int, default=50, help="episodes to run (default 50: mean within about ±0.5 logs; use 100 to separate close agents)")
    parser.add_argument("--port", type=int, default=8023, help="first instance's port (others go up by 10)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Minecraft instances to run in parallel (default: 6 in the arena, 3 in the natural world)")
    parser.add_argument("--shared-folder", action="store_true",
                        help="run all instances from CraftGround's one shared game folder (old behaviour)")
    parser.add_argument("--record", type=int, default=RECORD_PER_GROUP, metavar="N",
                        help=f"videos of the N best, N worst and N median episodes "
                             f"(default {RECORD_PER_GROUP}, so {3 * RECORD_PER_GROUP} videos; 0 = none)")
    parser.add_argument("--env", choices=sorted(ENVS), default="arena",
                        help="arena = walled superflat arena (chop/envs/arena.py); "
                             "natural = forest in a normal world (chop/envs/natural.py)")
    parser.add_argument("--video-fps", type=int, nargs="+", default=[VIDEO_FPS],
                        help=f"one video per playback rate given (default real speed only); "
                             f"{VIDEO_FPS} = real speed, 5 = quarter speed, e.g. --video-fps {VIDEO_FPS} 5 for both")
    parser.add_argument("--random-spots", action="store_true",
                        help="natural world: random spawns instead of the fixed evaluation spots")
    parser.add_argument("--spots", type=Path, default=None,
                        help="natural world: fixed spot file (default: the world's own, "
                             "results/eval_spots_natural.json for the training world)")
    parser.add_argument("--seed", default=natural_env.SEED,
                        help=f"natural world: world seed (default: the training world, {natural_env.SEED})")
    parser.add_argument("--held-out", action="store_true",
                        help=f"natural world: play the held-out world training never sees "
                             f"(seed {natural_env.HELD_OUT_SEED})")
    args = parser.parse_args()
    if args.held_out:
        args.seed = natural_env.HELD_OUT_SEED
    if args.env != "natural" and (args.held_out or args.seed != natural_env.SEED):
        parser.error("--seed / --held-out only apply to --env natural")
    if args.spots is None:
        args.spots = natural_env.eval_spots_path(args.seed)
    if args.n_envs is None:
        args.n_envs = DEFAULT_N_ENVS[args.env]
    if not args.random and not args.model:
        parser.error("give a model path, or --random for the baseline")
    args.error = parser.error
    return args


def fixed_spots(args, n_envs: int) -> tuple[list[list[dict] | None], str]:
    """Natural world: every model plays the same fixed spots (make_eval_spots.py), one per episode.
    Instance i gets spots i, i+n, i+2n, ..., which matches its share of the episodes exactly.
    Returns each instance's spots (None = search as usual) and the label saved with the results."""
    if args.env != "natural":
        return [None] * n_envs, ""
    if args.random_spots:
        return [None] * n_envs, "random"
    if not args.spots.exists():
        make = "make_eval_spots.py" + ("" if args.seed == natural_env.SEED else f" --seed {args.seed}")
        args.error(f"{args.spots} not found: run {make} first, or use --random-spots")
    try:
        all_spots = natural_env.load_eval_spots(args.spots, seed=args.seed)
    except ValueError as e:
        args.error(str(e))
    if len(all_spots) < args.episodes:
        args.error(f"{args.spots} has {len(all_spots)} spots, fewer than --episodes {args.episodes}")
    print(f"Fixed spots from {args.spots}: spots 1-{args.episodes}")
    return [all_spots[:args.episodes][i::n_envs] for i in range(n_envs)], f"fixed:{args.spots.name}"


def video_folder(stamp: str, spot_label: str) -> str:
    """Each evaluation run's videos get their own folder: its start time plus the spots it played,
    e.g. 20261119_013033_eval_spots_natural_wooded, 20261119_014237_random-spots, or just the time in the arena."""
    if spot_label.startswith("fixed:"):
        return f"{stamp}_{Path(spot_label.removeprefix('fixed:')).stem}"
    return f"{stamp}_{spot_label}-spots" if spot_label else stamp


def main():
    args = parse_args()
    # Load the model first: its number of actions says which action set the env must offer
    # (natural world: 13 with jump, or the arena's 11 for models trained before jump was added).
    # The random baseline keeps the 11 actions so it stays comparable with earlier runs.
    model = None if args.random else PPO.load(args.model, device="cpu")
    multi = (args.actions == "multi" if model is None
             else isinstance(model.action_space, gym.spaces.MultiDiscrete))
    if multi:
        n_actions, label = None, multi_label
        # A multi model's jump choice is its 4th dimension; --random multi plays with jump in the natural world.
        jump = (args.env == "natural" if model is None
                else len(model.action_space.nvec) == len(MULTI_NVEC_JUMP))
    else:
        n_actions = len(ACTION_NAMES) if model is None else model.action_space.n
        names = action_names(n_actions)
        jump = n_actions != len(ACTION_NAMES)
        label = lambda action: names[int(action)]  # noqa: E731
    action_repeat = args.action_repeat or (1 if model is None else run_action_repeat(args.model))
    if action_repeat > 1 and args.env != "arena":
        args.error("--action-repeat / repeat-trained models are arena only for now")
    actions = "multi" if multi else "single"
    env_kwargs = ({"jump": jump, "seed": args.seed, "actions": actions}
                  if args.env == "natural" else {"actions": actions, "action_repeat": action_repeat, "jump": jump})
    if action_repeat > 1:
        print(f"{action_repeat} game ticks per step")
    make_env = ENVS[args.env].make_env
    max_logs = getattr(ENVS[args.env], "MAX_LOGS", None)  # shown on video frames; the natural world has no cap
    n_envs = max(1, min(args.n_envs, args.episodes))
    shares = episode_shares(args.episodes, n_envs)
    per_instance_spots, spot_label = fixed_spots(args, n_envs)
    fixed = spot_label.startswith("fixed")
    if args.env == "natural":
        print(f"World seed {args.seed}" + (" (held out: never trained in)" if args.seed != natural_env.SEED
                                          else " (the training world)"))

    t_launch = time.perf_counter()
    ports = instance_ports(args.port, n_envs)
    factories = [(lambda port=port, extra=({"spots": per_instance_spots[i]} if fixed else {}):
                  make_env(port=port, env_path=game_folder(port, args.shared_folder), **env_kwargs, **extra))
                 for i, port in enumerate(ports)]
    env = launch_vec_env(factories, ports, args.shared_folder)
    random_name = "random" + ("-multi" if multi else "") + (f"-repeat{action_repeat}" if action_repeat > 1 else "")
    agent = random_name if args.random else normalize_agent(args.model)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = ((Path("runs/baselines") / random_name if args.random else Path(args.model).parent)
                 / "videos" / args.env / video_folder(stamp, spot_label))
    recorder = EpisodeRecorder(video_dir, stamp, list(dict.fromkeys(args.video_fps)), label, max_logs, n_envs,
                               step_ticks=action_repeat,
                               enabled=args.record > 0)
    print(f"{args.episodes} episodes over {n_envs} Minecraft instance(s): {shares}")

    done_count = [0] * n_envs        # counted episodes each instance has finished
    episode_id = [0] * n_envs        # number of the episode each instance is playing
    next_id = 0
    for i in range(n_envs):          # every instance starts its first counted episode now
        episode_id[i], next_id = next_id, next_id + 1
    steps = [0] * n_envs
    results: dict[int, int] = {}
    spot_of: dict[int, int | None] = {}  # episode -> fixed spot it played (from reset_infos)
    env_steps = resets = 0
    reset_s = 0.0
    t_play = None
    played_all = False
    try:
        obs = env.reset()
        t_play = time.perf_counter()  # every instance is up and in its first episode
        for i in range(n_envs):
            spot_of[episode_id[i]] = env.reset_infos[i].get("spot_index")
        while any(done < share for done, share in zip(done_count, shares)):
            if model is None:
                actions = np.array([env.action_space.sample() for _ in range(n_envs)])
            else:
                actions, _ = model.predict(obs, deterministic=False)
            obs, _, dones, infos = env.step(actions)
            env_steps += n_envs
            for i in np.flatnonzero(dones):  # those instances just reset; reset_infos says how long it took
                resets += 1
                reset_s += env.reset_infos[i].get("reset_seconds", 0.0)
            for i in range(n_envs):
                if done_count[i] >= shares[i]:
                    continue  # this instance has played its share; whatever it does now doesn't count
                steps[i] += 1
                ep = episode_id[i]
                # On the last step obs[i] is already the next episode's first frame.
                image = infos[i]["terminal_observation"] if dones[i] else obs[i]
                recorder.add_frame(i, image, steps[i], infos[i]["logs_chopped"], actions[i])
                if not dones[i]:
                    continue
                logs = infos[i]["logs_chopped"]
                results[ep] = logs
                done_count[i] += 1
                print(f"episode {ep + 1} (instance {i}): {logs}{f' / {max_logs}' if max_logs else ''} logs "
                      f"[{len(results)}/{args.episodes} done]")
                recorder.end_episode(i, ep)
                steps[i] = 0
                if done_count[i] < shares[i]:
                    episode_id[i], next_id = next_id, next_id + 1
                    spot_of[episode_id[i]] = env.reset_infos[i].get("spot_index")  # this step's auto-reset
            recorder.throttle()
        played_all = True
    finally:
        close_vec_env(env, ports, args.shared_folder)
        recorder.finish(played_all)
    t_end = time.perf_counter()

    episodes_done = sorted(results)
    scores = [results[k] for k in episodes_done]
    if not scores or t_play is None:
        recorder.discard()
        return
    spots_played = [spot_of.get(k) for k in episodes_done] if fixed else None
    timing = timing_summary(n_envs, launch_s=t_play - t_launch, play_s=t_end - t_play,
                            total_s=t_end - t_launch, env_steps=env_steps, resets=resets, reset_s=reset_s)
    save_results(agent, scores, env=args.env, timing=timing, spots=spot_label, spot_per_episode=spots_played,
                 world_seed=args.seed if args.env == "natural" else "")
    print(f"\n{agent} [{args.env}]: {np.mean(scores):.1f} logs per episode on average "
          f"(min {min(scores)}, max {max(scores)}, {len(scores)} episodes)")
    recorder.encode(results, args.record)
