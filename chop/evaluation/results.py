"""Recording an evaluation: its row in the results CSV, and how long it took."""

from datetime import datetime

import numpy as np

from chop.envs.common import EPISODE_STEPS
from chop.reports import update_report
from chop.reports.csvio import EVAL_CSV, NATURAL_EVAL_CSV, append_row

# Natural-world scores aren't comparable with the arena, so they get their own CSV
# and their own report (EXPERIMENTS_NATURAL.md).
FIELDS = ["timestamp", "agent", "episodes", "episode_steps", "mean_logs", "min_logs", "max_logs", "per_episode",
          "n_envs", "minutes", "launch_minutes", "steps_per_s", "reset_share", "spots", "spot_per_episode",
          "world_seed"]


def save_results(agent: str, results: list[int], timestamp: str | None = None, env: str = "arena",
                 timing: dict | None = None, spots: str = "", spot_per_episode: list | None = None,
                 world_seed: str = "") -> None:
    """spots: "fixed:<file>" for a fixed-spot natural evaluation, "random" for random spawns
    (blank in rows from before fixed spots existed, which were all random).
    world_seed: natural world only (blank in older rows, which all used the training world)."""
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
