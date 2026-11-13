"""Measure how many logs an agent chops per episode.

    python evaluate.py --random                              # baseline: random actions
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip               # a trained model
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip --episodes 10
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip --record 2    # also save videos of 2 episodes

(The model path for each training run is listed in EXPERIMENTS.md.)

--record N saves the first N episodes as MP4 videos of what the agent sees, enlarged
4x, with the step, logs chopped and current action written on each frame. They play
at real Minecraft speed (20 steps per second) and open in any video player, so you
can pause, scrub and slow them down. They go in the model's run folder under videos/
(random agent: runs/random/videos/).

The agent acts every tick, so real speed looks frantic. --video-fps 5 makes the
videos play at quarter speed (each step shown for 1/5 s).

Compare the trained model against the random baseline: that gap is what the
agent actually learned. Every run is appended to results/eval_results.csv
and the tables in EXPERIMENTS.md are refreshed.
"""

import argparse
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from experiments import EVAL_CSV, append_row, normalize_agent, update_report
from tree_env import ACTION_NAMES, EPISODE_STEPS, MAX_LOGS, make_env

VIDEO_SCALE = 4
VIDEO_FPS = 20  # one step is one game tick (1/20 s), so videos play at real Minecraft speed

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="path to a saved model .zip")
    parser.add_argument("--random", action="store_true", help="use random actions instead of a model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--port", type=int, default=8023)
    parser.add_argument("--record", type=int, default=0, metavar="N", help="save the first N episodes as MP4 videos")
    parser.add_argument("--video-fps", type=int, default=VIDEO_FPS,
                        help=f"video playback rate; {VIDEO_FPS} = real speed, 5 = quarter speed")
    args = parser.parse_args()
    if not args.random and not args.model:
        parser.error("give a model path, or --random for the baseline")

    env = make_env(port=args.port)
    model = None if args.random else PPO.load(args.model, device="cpu")
    label = "random" if args.random else normalize_agent(args.model)
    video_dir = (Path("runs/random") if args.random else Path(args.model).parent) / "videos"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            recording = ep < args.record
            frames = []
            done = False
            step = 0
            while not done:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1
                done = terminated or truncated
                if recording:
                    frames.append(video_frame(obs, step, info["logs_chopped"], int(action)))
            results.append(info["logs_chopped"])
            print(f"episode {ep + 1}: {info['logs_chopped']} / {MAX_LOGS} logs")
            if frames:
                speed = "" if args.video_fps == VIDEO_FPS else f"_{args.video_fps}fps"
                save_video(frames, video_dir / f"eval_{stamp}_ep{ep + 1}_{info['logs_chopped']}logs{speed}.mp4",
                           fps=args.video_fps)
    finally:
        env.close()

    if not results:
        return
    save_results(label, results)
    print(f"\n{label}: {np.mean(results):.1f} logs per episode on average "
          f"(min {min(results)}, max {max(results)}, {len(results)} episodes)")


if __name__ == "__main__":
    main()
