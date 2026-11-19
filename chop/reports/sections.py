"""Report sections both worlds use: one row per evaluation run, one row per training run."""

from chop.reports.csvio import is_fixed, is_held_out, is_wooded
from chop.reports.markdown import table


def evals_section(evals: list[dict]) -> str:
    if not evals:
        return "## Evaluation runs\n\n_None yet._"
    with_spots = any(r.get("spots") for r in evals)  # natural world: fixed or random spawns
    with_world = any(is_held_out(r) for r in evals)  # natural world: training or held-out world
    rows = [[r["timestamp"], f"`{r['agent']}`",
             *([("wooded" if is_wooded(r) else "fixed" if is_fixed(r) else "random")] if with_spots else []),
             *([("held-out" if is_held_out(r) else "training")] if with_world else []),
             r["episodes"], r["mean_logs"], r["min_logs"],
             r["max_logs"], r.get("n_envs") or "–", r.get("minutes") or "–", r.get("steps_per_s") or "–"]
            for r in reversed(evals)]
    return ("## Evaluation runs\n\nOne row per `evaluate.py` run, newest first. **Minutes** includes "
            "launching Minecraft; **Steps/s** is env steps per second over all instances while playing "
            "(runs from before timing was added show –)."
            + (" **Spots**: fixed = the saved forest spots, wooded = the saved wooded spots (any biome with "
               "trees), random = random spawns (all runs before 2026-11-16)." if with_spots else "")
            + (" **World**: training = the world agents train in, held-out = one they never see "
               "(`--held-out`)." if with_world else "") + "\n\n"
            + table(["When", "Agent", *(["Spots"] if with_spots else []), *(["World"] if with_world else []),
                     "Episodes", "Mean logs", "Min", "Max",
                     "Instances", "Minutes", "Steps/s"], rows)
            + "\n\nEvery episode's score is in the CSV (`per_episode`, with `spot_per_episode` saying which "
              "spot each one played).")


def training_section(runs: list[dict], start_cmd: str = "run.bat train.py") -> str:
    if not runs:
        return f"## Training runs\n\n_None yet. Start one with `{start_cmd}`._"
    def wandb_link(r: dict) -> str:
        url = r.get("wandb_url") or ""
        return f"[open]({url})" if url else "–"

    rows = [[r["started"], r["timesteps_done"], r["episodes"], r["minutes"], r["last20_avg_logs"],
             r["best_episode_logs"], r["status"], f"`{r['model']}`", wandb_link(r)] for r in reversed(runs)]
    return ("## Training runs\n\nOne row per `train.py` run, newest first.\n\n"
            + table(["Started", "Steps", "Episodes", "Minutes", "Last-20 avg logs", "Best episode",
                     "Status", "Saved model", "W&B"], rows))
