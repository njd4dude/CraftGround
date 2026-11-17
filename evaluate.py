"""Measure how many logs an agent chops per episode.

    python evaluate.py --random                              # baseline: random actions
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip               # a trained model
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --episodes 10 # quick check (default 100)
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --record 0    # no videos
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --record 34   # video of every episode (3 x 34 >= 100)
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --env natural # forest in a normal world
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --env natural --held-out  # a world never trained in

Runs 100 episodes by default so the mean is stable. Every run counts toward the
agent's score in EXPERIMENTS.md (EXPERIMENTS_NATURAL.md for --env natural), so to just grab a couple of videos use a small
--episodes, e.g. --episodes 2.

Episodes are spread over --n-envs Minecraft instances running side by side
(default 6, natural world 3, never more than --episodes), each running from its own copy of the
game folder (chop/infra/instances.py; --shared-folder for CraftGround's one shared folder,
where instances started together sometimes collide and hang the run). Each instance gets a fixed share of the
episodes decided up front (100 over 6 = 17,17,17,17,16,16) and only its first
share counts, so no half-played episode is ever dropped. (Stopping at "the first
100 to finish" would favour short episodes, and here short means good, because
an episode ends early once every log is chopped.)

(The model path for each training run is listed in EXPERIMENTS.md.)

21 episodes are saved as MP4 videos by default, picked by score once every episode is
done so they show where the agent succeeds and fails: the 7 best (file names start
top1..top7, top1 = most logs), the 7 worst (low1..low7, low1 = fewest) and the 7 around
the median (mid1..mid7, lowest first). --record N picks N per group (--record 0 for no
videos); when 3N covers every episode, all of them are saved. Ties go to the earlier
episode. All episodes still count toward the score; only the videos are limited.
Choosing after the fact means every episode's frames are kept until the end: they're
written as compressed .npz files to a temporary _frames_<time> folder next to the videos
(roughly 5-15 MB per episode) and deleted once the chosen ones are encoded. The encoding
happens after Minecraft has closed, so it doesn't slow the playing or the timing.

Each video is ~5 MB at real speed. The videos show what the agent sees,
enlarged 4x, with the step, logs chopped and current action written on each frame.
They open in any video player and go in the model's run folder under
videos/arena/ or videos/natural/, depending on --env (random agent:
runs/baselines/random/videos/<env>/).

Videos play at real speed by default. The agent acts every tick, so that looks frantic;
--video-fps adds slower copies: --video-fps 20 5 also saves a quarter-speed copy
(each step shown for 1/5 s, file name ends in _5fps).

At the end it prints how long the run took and how fast it went: launch time
(starting Minecraft), playing time, env steps per second (total and per
instance), and how much of each instance's time went into resets. Those numbers
are saved with the results too.

Compare the trained model against the random baseline: that gap is what the
agent actually learned. Every run is appended to results/eval_results.csv
and the tables in EXPERIMENTS.md are refreshed.
"""

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from chop.envs import natural as natural_env
from chop.envs import arena as arena_env
from chop.reports import update_report
from chop.reports.csvio import EVAL_CSV, NATURAL_EVAL_CSV, append_row, normalize_agent
from chop.infra.instances import clear_stale_loom_locks, close_vec_env, env_path_for, kill_instance_processes
from chop.envs.common import ACTION_NAMES, EPISODE_STEPS, action_names

ENVS = {"arena": arena_env, "natural": natural_env}
# Natural-world scores aren't comparable with the arena, so they get their own CSV
# and their own report (EXPERIMENTS_NATURAL.md).
MAX_LOGS = arena_env.MAX_LOGS  # shown on video frames; None in the natural world (no cap)

VIDEO_SCALE = 4
VIDEO_FPS = 20  # one step is one game tick (1/20 s), so videos play at real Minecraft speed
MAX_PENDING_VIDEOS = 8  # episodes waiting to be written before playing pauses to let writing catch up
RECORD_PER_GROUP = 7    # videos of the best, worst and median episodes, this many each

FIELDS = ["timestamp", "agent", "episodes", "episode_steps", "mean_logs", "min_logs", "max_logs", "per_episode",
          "n_envs", "minutes", "launch_minutes", "steps_per_s", "reset_share", "spots", "spot_per_episode",
          "world_seed"]


def save_results(agent: str, results: list[int], timestamp: str | None = None, env: str = "arena",
                 timing: dict | None = None, spots: str = "", spot_per_episode: list | None = None,
                 world_seed: str = "") -> None:
    """spots: "fixed:<file>" for a fixed-spot natural evaluation, "random" for random spawns
    (blank in rows from before fixed spots existed, which were all random).
    world_seed: natural world only (blank in older rows, which all used natural_env.SEED)."""
    csv_path = NATURAL_EVAL_CSV if env == "natural" else EVAL_CSV
    append_row(csv_path, FIELDS, {
        **(timing or {}),
        "spots": spots,
        "world_seed": world_seed,
        "spot_per_episode": " ".join(map(str, spot_per_episode)) if spot_per_episode else "",
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "agent": agent,
        "episodes": len(results),
        "episode_steps": EPISODE_STEPS,
        "mean_logs": f"{np.mean(results):.2f}",
        "min_logs": min(results),
        "max_logs": max(results),
        "per_episode": " ".join(map(str, results)),
    })
    print(f"Saved to {csv_path}")
    update_report(env)


NAMES = ACTION_NAMES  # action labels for videos; main() switches to the model's action set


def video_frame(image: np.ndarray, step: int, logs: int, action: int) -> np.ndarray:
    """Enlarge the agent's view and write step / logs / action across the top."""
    h, w = image.shape[:2]
    frame = Image.fromarray(image).resize((w * VIDEO_SCALE, h * VIDEO_SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(frame)
    draw.rectangle([0, 0, frame.width, 16], fill=(0, 0, 0))
    draw.text((4, 2), f"step {step:4d}   logs {logs:2d}{f'/{MAX_LOGS}' if MAX_LOGS else ''}   {NAMES[action]}", fill=(255, 255, 255))
    return np.asarray(frame)


def save_video(frames: list[np.ndarray], path: Path, fps: int = VIDEO_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 + yuv420p plays in Windows' Media Player, VS Code, browsers and VLC.
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  saved video {path} ({path.stat().st_size / 1e6:.1f} MB)")


def save_episode_videos(raw: list[tuple], base: Path, speeds: list[int]) -> None:
    """Build the annotated frames from the raw (image, step, logs, action) records and write
    one video per speed."""
    frames = [video_frame(*r) for r in raw]
    for fps in speeds:
        speed = "" if fps == VIDEO_FPS else f"_{fps}fps"
        save_video(frames, base.with_name(f"{base.name}{speed}.mp4"), fps=fps)


def save_raw_episode(raw: list[tuple], path: Path) -> None:
    """Keep one episode's small frames + labels on disk until we know whether it gets a video.
    Runs in a background thread so Minecraft keeps playing meanwhile."""
    images, steps, logs, actions = zip(*raw)
    np.savez_compressed(path, images=np.stack(images), steps=np.array(steps),
                        logs=np.array(logs), actions=np.array(actions))


def encode_saved_episode(path: Path, base: Path, speeds: list[int]) -> None:
    with np.load(path) as d:
        raw = list(zip(d["images"], d["steps"].tolist(), d["logs"].tolist(), d["actions"].tolist()))
    save_episode_videos(raw, base, speeds)


def pick_video_episodes(results: dict[int, int], per_group: int) -> list[tuple[str, int]]:
    """(label, episode) for the per_group best, worst and median-most episodes.
    Labels: top1 = most logs, low1 = fewest, mid1..midN = around the median, lowest first.
    Ties go to the earlier episode; an episode in two groups is saved once (top, then low, then mid)."""
    if per_group <= 0:
        return []
    order = sorted(results, key=lambda e: (results[e], e))           # fewest logs first
    best = sorted(results, key=lambda e: (-results[e], e))[:per_group]
    worst = order[:per_group]
    start = max(0, min(len(order) - per_group, len(order) // 2 - per_group // 2))
    middle = order[start:start + per_group]
    picked, seen = [], set()
    for prefix, group in (("top", best), ("low", worst), ("mid", middle)):
        for rank, ep in enumerate(group, 1):
            if ep not in seen:
                seen.add(ep)
                picked.append((f"{prefix}{rank}", ep))
    return picked


def timing_summary(n_envs: int, launch_s: float, play_s: float, total_s: float,
                   env_steps: int, resets: int, reset_s: float) -> dict:
    """Print how long the run took and how fast it went; return the numbers for the CSV."""
    total_rate = env_steps / play_s if play_s > 0 else 0.0
    per_instance = total_rate / n_envs
    # Share of all instances' playing time spent inside reset() (spawn search, terrain loading...).
    # Instances step in lockstep, so a slow reset also stalls the others: the real cost is higher.
    reset_share = reset_s / (n_envs * play_s) if play_s > 0 else 0.0
    busy = n_envs * play_s - reset_s
    no_reset_rate = env_steps / busy if busy > 0 else 0.0
    avg_reset = reset_s / resets if resets else 0.0
    print(f"\nTiming: {total_s / 60:.1f} min total = {launch_s / 60:.1f} min launching Minecraft "
          f"+ {play_s / 60:.1f} min playing (incl. saving frames for videos)")
    print(f"  {env_steps:,} env steps over {n_envs} instance(s): {total_rate:.0f} steps/s total, "
          f"{per_instance:.0f} per instance")
    print(f"  {resets} resets, {avg_reset:.1f} s each on average = {reset_share:.0%} of instance time; "
          f"outside resets each instance ran ~{no_reset_rate:.0f} steps/s")
    return {
        "n_envs": n_envs,
        "minutes": f"{total_s / 60:.1f}",
        "launch_minutes": f"{launch_s / 60:.1f}",
        "steps_per_s": f"{total_rate:.0f}",
        "reset_share": f"{reset_share:.2f}",
    }


def episode_shares(episodes: int, n_envs: int) -> list[int]:
    """Split the episodes as evenly as possible: 100 over 6 -> [17, 17, 17, 17, 16, 16]."""
    return [episodes // n_envs + (i < episodes % n_envs) for i in range(n_envs)]


# Natural-world instances grow to ~4.2 GB each (2G heap + ~1 GB native + Gradle launchers,
# measured with memprobe.py). 4 of them plus training leave ~0.3 GB of this 32 GB machine free;
# 3 leave ~5 GB.
N_ENVS_NATURAL = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="path to a saved model .zip")
    parser.add_argument("--random", action="store_true", help="use random actions instead of a model")
    parser.add_argument("--episodes", type=int, default=100, help="episodes to run (default 100, for a stable mean)")
    parser.add_argument("--port", type=int, default=8023, help="first instance's port (others go up by 10)")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Minecraft instances to run in parallel (default: 6 in the arena, 3 in the natural world)")
    parser.add_argument("--shared-folder", action="store_true",
                        help="run all instances from CraftGround's one shared game folder (old behaviour)")
    parser.add_argument("--record", type=int, default=RECORD_PER_GROUP, metavar="N",
                        help=f"videos of the N best, N worst and N median episodes "
                             f"(default {RECORD_PER_GROUP}, so {3 * RECORD_PER_GROUP} videos; 0 = none)")
    parser.add_argument("--env", choices=sorted(ENVS), default="arena",
                        help="arena = walled superflat arena (chop/envs/arena.py); natural = forest in a normal world (chop/envs/natural.py)")
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
        args.n_envs = N_ENVS_NATURAL if args.env == "natural" else 6
    if not args.random and not args.model:
        parser.error("give a model path, or --random for the baseline")

    global MAX_LOGS, NAMES
    # Load the model first: its number of actions says which action set the env must offer
    # (natural world: 13 with jump, or the arena's 11 for models trained before jump was added).
    # The random baseline keeps the 11 actions so it stays comparable with earlier runs.
    model = None if args.random else PPO.load(args.model, device="cpu")
    n_actions = len(ACTION_NAMES) if model is None else model.action_space.n
    NAMES = action_names(n_actions)
    if args.env != "natural" and n_actions != len(ACTION_NAMES):
        parser.error(f"this model has {n_actions} actions (with jump); only --env natural offers those")
    env_kwargs = ({"jump": n_actions != len(ACTION_NAMES), "seed": args.seed}
                  if args.env == "natural" else {})
    make_env = ENVS[args.env].make_env
    MAX_LOGS = getattr(ENVS[args.env], "MAX_LOGS", None)
    n_envs = max(1, min(args.n_envs, args.episodes))
    shares = episode_shares(args.episodes, n_envs)
    # Natural world: every model plays the same fixed spots (make_eval_spots.py), one per episode.
    # Instance i gets spots i, i+n, i+2n, ..., which matches its share of the episodes exactly.
    fixed = args.env == "natural" and not args.random_spots
    spot_label = "random" if args.env == "natural" else ""
    per_instance_spots: list[list[dict] | None] = [None] * n_envs
    if fixed:
        if not args.spots.exists():
            make = "make_eval_spots.py" + ("" if args.seed == natural_env.SEED else f" --seed {args.seed}")
            parser.error(f"{args.spots} not found: run {make} first, or use --random-spots")
        try:
            all_spots = natural_env.load_eval_spots(args.spots, seed=args.seed)
        except ValueError as e:
            parser.error(str(e))
        if len(all_spots) < args.episodes:
            parser.error(f"{args.spots} has {len(all_spots)} spots, fewer than --episodes {args.episodes}")
        per_instance_spots = [all_spots[:args.episodes][i::n_envs] for i in range(n_envs)]
        spot_label = f"fixed:{args.spots.name}"
        print(f"Fixed spots from {args.spots}: spots 1-{args.episodes}")
    if args.env == "natural":
        print(f"World seed {args.seed}" + (" (held out: never trained in)" if args.seed != natural_env.SEED
                                          else " (the training world)"))
    t_launch = time.perf_counter()
    shared = args.shared_folder
    ports = [args.port + 10 * i for i in range(n_envs)]
    if not shared:
        kill_instance_processes(ports, verbose=True)  # leftovers of a killed run would block these ports
    clear_stale_loom_locks()  # also needed with --shared-folder; see chop/infra/instances.py
    factories = [(lambda port=args.port + 10 * i, extra=({"spots": per_instance_spots[i]} if fixed else {}):
                  make_env(port=port, env_path=None if shared else env_path_for(port, verbose=True),
                           **env_kwargs, **extra))
                 for i in range(n_envs)]
    env = SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)
    label = "random" if args.random else normalize_agent(args.model)
    video_dir = (Path("runs/baselines/random") if args.random else Path(args.model).parent) / "videos" / args.env
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    speeds = list(dict.fromkeys(args.video_fps))
    recording = args.record > 0 and bool(speeds)
    frames_dir = video_dir / f"_frames_{stamp}"  # every episode's frames until the videos are chosen
    if recording:
        frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.episodes} episodes over {n_envs} Minecraft instance(s): {shares}")

    done_count = [0] * n_envs        # counted episodes each instance has finished
    episode_id = [0] * n_envs        # number of the episode each instance is playing
    next_id = 0
    for i in range(n_envs):          # every instance starts its first counted episode now
        episode_id[i], next_id = next_id, next_id + 1
    steps = [0] * n_envs
    raw = [[] for _ in range(n_envs)]   # small frames + labels; enlarged only when encoding
    results: dict[int, int] = {}
    spot_of: dict[int, int | None] = {}  # episode -> fixed spot it played (from reset_infos)
    encoder = ThreadPoolExecutor(max_workers=2)
    pending = []
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
                if recording:
                    # On the last step obs[i] is already the next episode's first frame.
                    image = infos[i]["terminal_observation"] if dones[i] else obs[i]
                    raw[i].append((np.array(image), steps[i], infos[i]["logs_chopped"], int(actions[i])))
                if not dones[i]:
                    continue
                logs = infos[i]["logs_chopped"]
                results[ep] = logs
                done_count[i] += 1
                print(f"episode {ep + 1} (instance {i}): {logs}{f' / {MAX_LOGS}' if MAX_LOGS else ''} logs "
                      f"[{len(results)}/{args.episodes} done]")
                if recording:
                    pending.append(encoder.submit(save_raw_episode, raw[i], frames_dir / f"ep{ep + 1}.npz"))
                raw[i], steps[i] = [], 0
                if done_count[i] < shares[i]:
                    episode_id[i], next_id = next_id, next_id + 1
                    spot_of[episode_id[i]] = env.reset_infos[i].get("spot_index")  # this step's auto-reset
            # Don't let unencoded episodes pile up in memory.
            while sum(not f.done() for f in pending) > MAX_PENDING_VIDEOS:
                pending[next(k for k, f in enumerate(pending) if not f.done())].result()
        played_all = True
    finally:
        close_vec_env(env, ports, shared)
        for f in pending:
            if played_all:
                f.result()  # finish writing frames (and surface any error)
            else:
                f.exception()  # just wait; the error that stopped the run is the one to report
        encoder.shutdown()
        if not played_all:
            shutil.rmtree(frames_dir, ignore_errors=True)  # no score is saved, so no videos either
    t_end = time.perf_counter()

    scores = dict(results)
    episodes_done = sorted(results)
    results = [results[k] for k in episodes_done]
    if not results or t_play is None:
        shutil.rmtree(frames_dir, ignore_errors=True)
        return
    spots_played = [spot_of.get(k) for k in episodes_done] if fixed else None
    timing = timing_summary(n_envs, launch_s=t_play - t_launch, play_s=t_end - t_play,
                            total_s=t_end - t_launch, env_steps=env_steps, resets=resets, reset_s=reset_s)
    save_results(label, results, env=args.env, timing=timing, spots=spot_label, spot_per_episode=spots_played,
                 world_seed=args.seed if args.env == "natural" else "")
    print(f"\n{label} [{args.env}]: {np.mean(results):.1f} logs per episode on average "
          f"(min {min(results)}, max {max(results)}, {len(results)} episodes)")

    if recording:
        picked = pick_video_episodes(scores, args.record)
        print(f"\nEncoding {len(picked)} videos (best, worst and median episodes) to {video_dir}")
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:  # Minecraft is closed, so the CPU is free
                jobs = [pool.submit(encode_saved_episode, frames_dir / f"ep{ep + 1}.npz",
                                    video_dir / f"eval_{stamp}_{tag}_ep{ep + 1}_{scores[ep]}logs", speeds)
                        for tag, ep in picked]
                for job in jobs:
                    job.result()
        finally:
            shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
