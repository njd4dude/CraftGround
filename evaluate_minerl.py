"""Run a model trained in the MineRL project in this project's tree arena.

    python evaluate_minerl.py path/to/minerl_model.zip                 # 20 episodes over 4 instances
    python evaluate_minerl.py path/to/minerl_model.zip --episodes 100
    python evaluate_minerl.py path/to/minerl_model.zip --deterministic # always pick the likeliest action
    python evaluate_minerl.py --random                                 # random actions, same env, for comparison

Uses minerl_compat_env.py, which gives the model exactly the input and actions it
was trained with (64x64 crop of a 320x180 frame, MultiDiscrete([3, 11, 9]),
attack held, 4-tick frame skip). Episodes are 1000 ticks (250 decisions), the
same game time as tree_env.py episodes, so logs per episode are comparable with
this project's own agents.

Results are appended to results/eval_results_minerl.csv (kept separate from the
EXPERIMENTS.md tables, since it is a different agent setup).
The first --record episodes are saved as MP4s of the full 320x180 view next to
the model, under minerl_eval_videos/.
"""

import argparse
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import minerl_compat_env
from experiments import append_row

CSV = Path("results/eval_results_minerl.csv")
FIELDS = ["timestamp", "agent", "deterministic", "episodes", "mean_logs", "min_logs", "max_logs", "per_episode"]


def episode_shares(episodes: int, n_envs: int) -> list[int]:
    return [episodes // n_envs + (i < episodes % n_envs) for i in range(n_envs)]


def frame_for_video(pov: np.ndarray, decision: int, logs: int, action) -> np.ndarray:
    img = Image.fromarray(pov).resize((pov.shape[1] * 2, pov.shape[0] * 2), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 14], fill=(0, 0, 0))
    draw.text((4, 1), f"decision {decision:3d}  logs {logs:2d}  {minerl_compat_env.describe_action(action)}",
              fill=(255, 255, 255))
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="SB3 PPO .zip trained in the MineRL project")
    parser.add_argument("--random", action="store_true", help="random actions instead of a model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--port", type=int, default=8123, help="first instance's port (others go up by 10)")
    parser.add_argument("--deterministic", action="store_true", help="pick the likeliest action instead of sampling")
    parser.add_argument("--record", type=int, default=3, metavar="N", help="save the first N episodes as MP4 (0 = none)")
    args = parser.parse_args()
    if not args.random and not args.model:
        parser.error("give a model path, or --random")

    n_envs = max(1, min(args.n_envs, args.episodes))
    makers = [lambda p=args.port + 10 * i: minerl_compat_env.make_env(port=p) for i in range(n_envs)]
    venv = SubprocVecEnv(makers) if n_envs > 1 else DummyVecEnv(makers)

    model = None
    if not args.random:
        model = PPO.load(args.model, device="auto")
        if model.observation_space.shape not in [(64, 64, 3), (3, 64, 64)]:
            raise SystemExit(f"model expects observations {model.observation_space.shape}, not 64x64x3")
        if list(model.action_space.nvec) != [3, 11, 9]:
            raise SystemExit(f"model's action space is {model.action_space}, expected MultiDiscrete([3, 11, 9])")

    agent = "random" if args.random else str(args.model)
    out_dir = (Path("runs/random") if args.random else Path(args.model).parent) / "minerl_eval_videos"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    shares = episode_shares(args.episodes, n_envs)
    done_counts = [0] * n_envs
    results: list[int] = []
    decisions = [0] * n_envs
    # Only instance 0 records, so its episodes are videos 1..N.
    frames: list[np.ndarray] = []
    recorded = 0

    obs = venv.reset()
    while any(d < s for d, s in zip(done_counts, shares)):
        if model is None:
            actions = np.array([venv.action_space.sample() for _ in range(n_envs)])
        else:
            actions, _ = model.predict(obs, deterministic=args.deterministic)
        obs, _, dones, infos = venv.step(actions)
        if recorded < args.record and done_counts[0] < shares[0]:
            pov = venv.get_attr("last_pov", indices=[0])[0]
            frames.append(frame_for_video(pov, decisions[0], infos[0]["logs_chopped"], actions[0]))
        for i, (done, info) in enumerate(zip(dones, infos)):
            decisions[i] += 1
            if not done:
                continue
            decisions[i] = 0
            if done_counts[i] >= shares[i]:
                continue  # this instance already played its share; ignore extra episodes
            done_counts[i] += 1
            logs = info["logs_chopped"]
            results.append(logs)
            print(f"episode {len(results):3d}/{args.episodes} (instance {i}): {logs} logs")
            if i == 0 and recorded < args.record:
                recorded += 1
                path = out_dir / f"{stamp}_ep{recorded}.mp4"
                path.parent.mkdir(parents=True, exist_ok=True)
                imageio.mimwrite(path, frames, fps=5, codec="libx264", quality=8,
                                 pixelformat="yuv420p", macro_block_size=8)
                print(f"  saved video {path}")
                frames = []
    venv.close()

    print(f"\n{agent}: mean {np.mean(results):.2f} logs/episode over {len(results)} episodes "
          f"(min {min(results)}, max {max(results)})")
    append_row(CSV, FIELDS, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "agent": agent,
        "deterministic": args.deterministic,
        "episodes": len(results),
        "mean_logs": f"{np.mean(results):.2f}",
        "min_logs": min(results),
        "max_logs": max(results),
        "per_episode": " ".join(map(str, results)),
    })
    print(f"Saved to {CSV}")


if __name__ == "__main__":
    main()
