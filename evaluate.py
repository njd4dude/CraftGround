"""Measure how many logs an agent chops per episode.

    python evaluate.py --random                              # baseline: random actions
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip               # a trained model
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip --episodes 10 # quick check (default 100)
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip --record 0    # no videos
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip --record 100  # video of every episode

Runs 100 episodes by default so the mean is stable. Every run counts toward the
agent's score in EXPERIMENTS.md, so to just grab a couple of videos use a small
--episodes, e.g. --episodes 2.

Episodes are spread over --n-envs Minecraft instances running side by side
(default 6, never more than --episodes). Each instance gets a fixed share of the
episodes decided up front (100 over 6 = 17,17,17,17,16,16) and only its first
share counts, so no half-played episode is ever dropped. (Stopping at "the first
100 to finish" would favour short episodes, and here short means good, because
an episode ends early once every log is chopped.)

(The model path for each training run is listed in EXPERIMENTS.md.)

The first 20 episodes are saved as MP4 videos by default (--record N for the
first N, --record 0 for none). All episodes still count toward the score; only
the videos are limited. Each video is ~5 MB at real speed and ~7 MB at 5 fps,
so the default 20 recordings take ~240 MB. The videos show what the agent sees,
enlarged 4x, with the step, logs chopped and current action written on each frame.
They open in any video player and go in the model's run folder under videos/
(random agent: runs/random/videos/).

The agent acts every tick, so real speed looks frantic. By default each
recorded episode is saved twice from the same frames: at real speed and at
quarter speed (5 fps, each step shown for 1/5 s, file name ends in _5fps).
--video-fps picks the speeds: --video-fps 20 for real speed only, --video-fps 5 10 20 for three.

Compare the trained model against the random baseline: that gap is what the
agent actually learned. Every run is appended to results/eval_results.csv
and the tables in EXPERIMENTS.md are refreshed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from experiments import EVAL_CSV, append_row, normalize_agent, update_report
from tree_env import ACTION_NAMES, EPISODE_STEPS, MAX_LOGS, make_env

VIDEO_SCALE = 4
VIDEO_FPS = 20  # one step is one game tick (1/20 s), so videos play at real Minecraft speed
MAX_PENDING_VIDEOS = 8  # episodes waiting to be encoded before playing pauses to let encoding catch up

FIELDS = ["timestamp", "agent", "episodes", "episode_steps", "mean_logs", "min_logs", "max_logs", "per_episode"]


def save_results(agent: str, results: list[int], timestamp: str | None = None) -> None:
    append_row(EVAL_CSV, FIELDS, {
        "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
        "agent": agent,
        "episodes": len(results),
        "episode_steps": EPISODE_STEPS,
        "mean_logs": f"{np.mean(results):.2f}",
        "min_logs": min(results),
        "max_logs": max(results),
        "per_episode": " ".join(map(str, results)),
    })
    print(f"Saved to {EVAL_CSV}")
    update_report()


def video_frame(image: np.ndarray, step: int, logs: int, action: int) -> np.ndarray:
    """Enlarge the agent's view and write step / logs / action across the top."""
    h, w = image.shape[:2]
    frame = Image.fromarray(image).resize((w * VIDEO_SCALE, h * VIDEO_SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(frame)
    draw.rectangle([0, 0, frame.width, 16], fill=(0, 0, 0))
    draw.text((4, 2), f"step {step:4d}   logs {logs:2d}/{MAX_LOGS}   {ACTION_NAMES[action]}", fill=(255, 255, 255))
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
    one video per speed. Runs in a background thread so Minecraft keeps playing meanwhile."""
    frames = [video_frame(*r) for r in raw]
    for fps in speeds:
        speed = "" if fps == VIDEO_FPS else f"_{fps}fps"
        save_video(frames, base.with_name(f"{base.name}{speed}.mp4"), fps=fps)


def episode_shares(episodes: int, n_envs: int) -> list[int]:
    """Split the episodes as evenly as possible: 100 over 6 -> [17, 17, 17, 17, 16, 16]."""
    return [episodes // n_envs + (i < episodes % n_envs) for i in range(n_envs)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="path to a saved model .zip")
    parser.add_argument("--random", action="store_true", help="use random actions instead of a model")
    parser.add_argument("--episodes", type=int, default=100, help="episodes to run (default 100, for a stable mean)")
    parser.add_argument("--port", type=int, default=8023, help="first instance's port (others go up by 10)")
    parser.add_argument("--n-envs", type=int, default=6, help="Minecraft instances to run in parallel")
    parser.add_argument("--record", type=int, default=20, metavar="N",
                        help="save the first N episodes as MP4 videos (default 20; 0 = none)")
    parser.add_argument("--video-fps", type=int, nargs="+", default=[VIDEO_FPS, 5],
                        help=f"one video per playback rate given; {VIDEO_FPS} = real speed, 5 = quarter speed")
    args = parser.parse_args()
    if not args.random and not args.model:
        parser.error("give a model path, or --random for the baseline")

    n_envs = max(1, min(args.n_envs, args.episodes))
    shares = episode_shares(args.episodes, n_envs)
    factories = [(lambda port=args.port + 10 * i: make_env(port=port)) for i in range(n_envs)]
    env = SubprocVecEnv(factories) if n_envs > 1 else DummyVecEnv(factories)
    model = None if args.random else PPO.load(args.model, device="cpu")
    label = "random" if args.random else normalize_agent(args.model)
    video_dir = (Path("runs/random") if args.random else Path(args.model).parent) / "videos"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    speeds = list(dict.fromkeys(args.video_fps))
    print(f"{args.episodes} episodes over {n_envs} Minecraft instance(s): {shares}")

    done_count = [0] * n_envs        # counted episodes each instance has finished
    episode_id = [0] * n_envs        # number of the episode each instance is playing
    next_id = 0
    for i in range(n_envs):          # every instance starts its first counted episode now
        episode_id[i], next_id = next_id, next_id + 1
    steps = [0] * n_envs
    raw = [[] for _ in range(n_envs)]   # small frames + labels; enlarged only when encoding
    results: dict[int, int] = {}
    encoder = ThreadPoolExecutor(max_workers=2)
    pending = []
    try:
        obs = env.reset()
        while any(done < share for done, share in zip(done_count, shares)):
            if model is None:
                actions = np.array([env.action_space.sample() for _ in range(n_envs)])
            else:
                actions, _ = model.predict(obs, deterministic=False)
            obs, _, dones, infos = env.step(actions)
            for i in range(n_envs):
                if done_count[i] >= shares[i]:
                    continue  # this instance has played its share; whatever it does now doesn't count
                steps[i] += 1
                ep = episode_id[i]
                recording = ep < args.record
                if recording:
                    # On the last step obs[i] is already the next episode's first frame.
                    image = infos[i]["terminal_observation"] if dones[i] else obs[i]
                    raw[i].append((np.array(image), steps[i], infos[i]["logs_chopped"], int(actions[i])))
                if not dones[i]:
                    continue
                logs = infos[i]["logs_chopped"]
                results[ep] = logs
                done_count[i] += 1
                print(f"episode {ep + 1} (instance {i}): {logs} / {MAX_LOGS} logs "
                      f"[{len(results)}/{args.episodes} done]")
                if recording and speeds:
                    pending.append(encoder.submit(save_episode_videos, raw[i],
                                                  video_dir / f"eval_{stamp}_ep{ep + 1}_{logs}logs", speeds))
                raw[i], steps[i] = [], 0
                if done_count[i] < shares[i]:
                    episode_id[i], next_id = next_id, next_id + 1
            # Don't let unencoded episodes pile up in memory.
            while sum(not f.done() for f in pending) > MAX_PENDING_VIDEOS:
                pending[next(k for k, f in enumerate(pending) if not f.done())].result()
    finally:
        env.close()
        for f in pending:
            f.result()  # finish writing videos (and surface any encoding error)
        encoder.shutdown()

    results = [results[k] for k in sorted(results)]
    if not results:
        return
    save_results(label, results)
    print(f"\n{label}: {np.mean(results):.1f} logs per episode on average "
          f"(min {min(results)}, max {max(results)}, {len(results)} episodes)")


if __name__ == "__main__":
    main()
