"""EXPERIMENTS.md: the arena's report."""

import json
from datetime import datetime
from pathlib import Path

from chop.envs import arena as arena_env
from chop.envs import common
from chop.reports.csvio import EVAL_CSV, TRAIN_CSV, normalize_agent, read_rows
from chop.reports.markdown import END, START, table, write_report
from chop.reports.sections import evals_section, training_section

REPORT = Path("EXPERIMENTS.md")


TEMPLATE = f"""# CraftGround Experiments: Chop Trees

Goal: train an agent to chop as many oak logs as possible in one episode.

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""


def setup_section() -> str:
    rows = [
        ["World", "Superflat arena, stone-brick walls at ±%d blocks" % arena_env.ARENA_RADIUS],
        ["Trees", f"{len(arena_env.TREE_POSITIONS)} oaks × {arena_env.TRUNK_HEIGHT} logs = {arena_env.MAX_LOGS} logs max"],
        ["Rebuilt each episode", "yes (trees regrow on reset)"],
        ["Tool", "Unbreakable diamond axe, Efficiency X (1 hit per log)"],
        ["Spawn", "Arena centre, random facing (15° steps)"],
        ["Observation", f"RGB image {common.IMAGE_WIDTH}×{common.IMAGE_HEIGHT}"],
        ["Actions", f"{len(common.ACTION_NAMES)} discrete: " + ", ".join(common.ACTION_NAMES)],
        ["Reward", arena_env.REWARD],
        ["Episode length", f"{common.EPISODE_STEPS} steps (or all logs chopped)"],
    ]
    return "## Environment setup\n\n" + table(["Setting", "Value"], rows)


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
        f"episodes for that agent (max {arena_env.MAX_LOGS}). **Training avg** is the last-20-episode "
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
        n_envs = cfg.get("n_envs", 1) if cfg else 1
        envs = f", {n_envs} parallel envs" if n_envs > 1 else ""
        lines.append(f"- **Trained:** {int(run['timesteps_done']):,} steps{target}, {run['episodes']} episodes, "
                     f"{run['minutes']} min{envs} ({run['status']})")
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
                            # A setting older runs didn't record yet is not a change.
                            for k, v in env.items() if k in prev_env and prev_env[k] != v]
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


def update(stamp: str | None = None) -> None:
    """Rebuild EXPERIMENTS.md's tables from the arena CSVs."""
    stamp = stamp or f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"
    evals, runs = read_rows(EVAL_CSV), read_rows(TRAIN_CSV)
    generated = "\n\n".join([
        stamp,
        summary_section(evals, runs),
        agents_section(evals, runs),
        training_section(runs),
        evals_section(evals),
        setup_section(),
    ])
    write_report(REPORT, TEMPLATE, generated)
