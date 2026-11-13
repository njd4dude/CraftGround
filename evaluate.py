"""Measure how many logs an agent chops per episode.

    python evaluate.py --random                              # baseline: random actions
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip               # a trained model
    python evaluate.py runs/chop_ppo_20261113_131800/model.zip --episodes 10

(The model path for each training run is listed in EXPERIMENTS.md.)

Compare the trained model against the random baseline: that gap is what the
agent actually learned. Every run is appended to results/eval_results.csv
and the tables in EXPERIMENTS.md are refreshed.
"""

import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO

from experiments import EVAL_CSV, append_row, update_report
from tree_env import EPISODE_STEPS, MAX_LOGS, make_env

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="path to a saved model .zip")
    parser.add_argument("--random", action="store_true", help="use random actions instead of a model")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--port", type=int, default=8023)
    args = parser.parse_args()
    if not args.random and not args.model:
        parser.error("give a model path, or --random for the baseline")

    env = make_env(port=args.port)
    model = None if args.random else PPO.load(args.model, device="cpu")
    label = "random" if args.random else args.model

    results = []
    try:
        for ep in range(args.episodes):
            obs, info = env.reset()
            done = False
            while not done:
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            results.append(info["logs_chopped"])
            print(f"episode {ep + 1}: {info['logs_chopped']} / {MAX_LOGS} logs")
    finally:
        env.close()

    if not results:
        return
    save_results(label, results)
    print(f"\n{label}: {np.mean(results):.1f} logs per episode on average "
          f"(min {min(results)}, max {max(results)}, {len(results)} episodes)")


if __name__ == "__main__":
    main()
