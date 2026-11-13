"""Keeps EXPERIMENTS.md up to date.

evaluate.py and train.py append rows to the CSV files in results/, then call
update_report(), which rewrites the tables in EXPERIMENTS.md. Only the part
between the AUTO-GENERATED markers is replaced, so notes you write anywhere
else in the file are kept.

Run it by hand to rebuild the tables:  python experiments.py
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import tree_env

RESULTS_DIR = Path("results")
EVAL_CSV = RESULTS_DIR / "eval_results.csv"
TRAIN_CSV = RESULTS_DIR / "training_runs.csv"
REPORT = Path("EXPERIMENTS.md")

START = "<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->"
END = "<!-- AUTO-GENERATED:END -->"

TEMPLATE = f"""# CraftGround Experiments: Chop Trees

Goal: train an agent to chop as many oak logs as possible in one episode.

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""


def append_row(path: Path, fields: list[str], row: dict) -> None:
    """Append a row. If `fields` has columns the file doesn't, the file is rewritten
    with the extra columns (left blank for older rows) so nothing gets misaligned."""
    path.parent.mkdir(exist_ok=True)
    old_rows = read_rows(path)
    old_fields = list(old_rows[0].keys()) if old_rows else []
    if path.exists() and old_rows and not set(fields) <= set(old_fields):
        merged = old_fields + [f for f in fields if f not in old_fields]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged, restval="")
            writer.writeheader()
            writer.writerows(old_rows + [row])
        return
    new_file = not path.exists() or not old_fields
    with path.open("a" if not new_file else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields or fields, restval="")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def setup_section() -> str:
    rows = [
        ["World", "Superflat arena, stone-brick walls at ±%d blocks" % tree_env.ARENA_RADIUS],
        ["Trees", f"{len(tree_env.TREE_POSITIONS)} oaks × {tree_env.TRUNK_HEIGHT} logs = {tree_env.MAX_LOGS} logs max"],
        ["Rebuilt each episode", "yes (trees regrow on reset)"],
        ["Tool", "Unbreakable diamond axe, Efficiency X (1 hit per log)"],
        ["Spawn", "Arena centre, random facing (15° steps)"],
        ["Observation", f"RGB image {tree_env.IMAGE_WIDTH}×{tree_env.IMAGE_HEIGHT}"],
        ["Actions", f"{len(tree_env.ACTION_NAMES)} discrete: " + ", ".join(tree_env.ACTION_NAMES)],
        ["Reward", tree_env.REWARD],
        ["Episode length", f"{tree_env.EPISODE_STEPS} steps (or all logs chopped)"],
    ]
    return "## Environment setup\n\n" + table(["Setting", "Value"], rows)


def normalize_agent(agent: str) -> str:
    """'runs\\x\\model.zip', './runs/x/model.zip' and 'runs/x/model.zip' are the same agent."""
    return agent if agent == "random" else Path(agent).as_posix().removeprefix("./")


def summary_section(evals: list[dict], runs: list[dict]) -> str:
    by_agent: dict[str, list[int]] = {}
    for r in evals:
        by_agent.setdefault(normalize_agent(r["agent"]), []).extend(int(x) for x in r["per_episode"].split())

    random_eps = by_agent.get("random", [])
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None

    def eval_cells(eps: list[int], is_random: bool) -> list:
        mean = sum(eps) / len(eps)
        vs = "baseline" if is_random else (f"**{mean / random_mean:.1f}×**" if random_mean else "no baseline yet")
        return [len(eps), f"**{mean:.2f}**", min(eps), max(eps), vs]

    rows = []
    if random_eps:
        rows.append(["Random actions (baseline)", "–", *eval_cells(random_eps, True)])
    else:
        rows.append(["Random actions (baseline)", "–", "_not evaluated: `run.bat evaluate.py --random --episodes 20`_",
                     "", "", "", ""])

    listed = {"random"}
    for run in reversed(runs):  # newest first
        model = normalize_agent(run["model"])
        listed.add(model)
        name = Path(model).parent.name
        eps = by_agent.get(model)
        if eps:
            rows.append([f"`{name}`", run["last20_avg_logs"], *eval_cells(eps, False)])
        else:
            rows.append([f"`{name}`", run["last20_avg_logs"],
                         f"_not evaluated: `run.bat evaluate.py {model} --episodes 20`_", "", "", "", ""])

    # Evaluated models that aren't from a recorded training run (e.g. a checkpoint).
    for agent, eps in by_agent.items():
        if agent not in listed:
            rows.append([f"`{agent}`", "–", *eval_cells(eps, False)])

    return (
        "## Summary\n\n"
        "One row per agent. **Mean logs** is the real score: the average over all `evaluate.py` "
        f"episodes for that agent (max {tree_env.MAX_LOGS}). **Training avg** is the last-20-episode "
        "average seen during training. It's a useful hint, but the agent was still exploring then, "
        "so evaluate before trusting it. **vs random** = how many times more logs than random actions.\n\n"
        + table(["Agent", "Training avg", "Eval episodes", "Mean logs", "Min", "Max", "vs random"], rows)
    )


def load_config(model: str) -> dict | None:
    """The config.json train.py saves next to each run's model.zip."""
    path = Path(model).parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def agents_section(evals: list[dict], runs: list[dict]) -> str:
    """One subsection per trained agent: how long it trained, its settings, and what was
    different about it (hyperparameters changed from the defaults, environment/reward changes
    since the previous run, and the run's notes)."""
    if not runs:
        return "## Agents\n\n_None yet._"
    by_agent: dict[str, list[int]] = {}
    for r in evals:
        by_agent.setdefault(normalize_agent(r["agent"]), []).extend(int(x) for x in r["per_episode"].split())

    blocks = []
    prev_env = None
    for run in runs:  # oldest first, so environment changes are compared with the run before
        model = normalize_agent(run["model"])
        cfg = load_config(model)
        lines = [f"### `{Path(model).parent.name}`", ""]
        target = f" of {int(cfg['timesteps_target']):,} planned" if cfg else ""
        lines.append(f"- **Trained:** {int(run['timesteps_done']):,} steps{target}, {run['episodes']} episodes, "
                     f"{run['minutes']} min ({run['status']})")
        if run.get("resumed_from"):
            lines.append(f"- **Resumed from:** `{run['resumed_from']}`")
        eps = by_agent.get(model)
        score = (f"{sum(eps) / len(eps):.2f} logs/episode over {len(eps)} eval episodes" if eps
                 else "not evaluated yet")
        lines.append(f"- **Result:** {score} (training last-20 avg {run['last20_avg_logs']})")

        if cfg is None:
            lines.append("- _No config.json for this run, so its settings weren't recorded._")
        else:
            changed = cfg.get("changed_from_defaults", {})
            ppo = ", ".join(f"**{k}={v}**" if k in changed else f"{k}={v}" for k, v in cfg["ppo"].items())
            lines.append(f"- **PPO settings:** {ppo}")
            special = [f"{k}: {c['default']} → **{c['used']}**" for k, c in changed.items()]
            env = cfg.get("env", {})
            if prev_env is not None:
                special += [f"env {k}: {prev_env.get(k)} → **{v}** (changed since previous run)"
                            for k, v in env.items() if prev_env.get(k) != v]
            same = ("first recorded run, default settings (the baseline)" if prev_env is None
                    else "nothing, default settings and same environment as the previous run")
            prev_env = env
            lines.append("- **What's special:** " + ("; ".join(special) if special else same))
            if cfg.get("notes"):
                lines.append(f"- **Notes:** {cfg['notes']}")
        blocks.append("\n".join(lines))

    return ("## Agents\n\nOne section per trained agent, newest first. Hyperparameters changed from the "
            "defaults in `train.py` are in bold. Environment changes (reward, episode length, trees...) are "
            "flagged on the first run that used them. Add notes with `train.py --notes \"...\"`, or edit "
            "`notes` in the run's `config.json` and rerun `python experiments.py`.\n\n"
            + "\n\n".join(reversed(blocks)))


def evals_section(evals: list[dict]) -> str:
    if not evals:
        return "## Evaluation runs\n\n_None yet._"
    rows = [[r["timestamp"], f"`{r['agent']}`", r["episodes"], r["mean_logs"], r["min_logs"],
             r["max_logs"], r["per_episode"]] for r in reversed(evals)]
    return ("## Evaluation runs\n\nOne row per `evaluate.py` run, newest first.\n\n"
            + table(["When", "Agent", "Episodes", "Mean logs", "Min", "Max", "Per episode"], rows))


def training_section(runs: list[dict]) -> str:
    if not runs:
        return "## Training runs\n\n_None yet. Start one with `run.bat train.py`._"
    def wandb_link(r: dict) -> str:
        url = r.get("wandb_url") or ""
        return f"[open]({url})" if url else "–"

    rows = [[r["started"], r["timesteps_done"], r["episodes"], r["minutes"], r["last20_avg_logs"],
             r["best_episode_logs"], r["status"], f"`{r['model']}`", wandb_link(r)] for r in reversed(runs)]
    return ("## Training runs\n\nOne row per `train.py` run, newest first.\n\n"
            + table(["Started", "Steps", "Episodes", "Minutes", "Last-20 avg logs", "Best episode",
                     "Status", "Saved model", "W&B"], rows))


def update_report() -> None:
    evals, runs = read_rows(EVAL_CSV), read_rows(TRAIN_CSV)
    generated = "\n\n".join([
        f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._",
        summary_section(evals, runs),
        agents_section(evals, runs),
        training_section(runs),
        evals_section(evals),
        setup_section(),
    ])
    text = REPORT.read_text(encoding="utf-8") if REPORT.exists() else TEMPLATE
    if START not in text or END not in text:
        text = TEMPLATE + "\n" + text  # markers removed by hand: put a fresh block on top
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    REPORT.write_text(f"{before}{START}\n{generated}\n{END}{after}", encoding="utf-8")
    print(f"Updated {REPORT}")


if __name__ == "__main__":
    update_report()
