"""EXPERIMENTS_ARENA.md: the arena's report.

Organised by experiment (results/experiments_arena.json): each one's question, what it found,
and its runs with their setup and evaluation score. Runs no experiment lists go under
"Other runs". Everything per run (PPO settings, notes, every episode) stays in the CSVs and in
each run's config.json; the report only shows what you compare runs by."""

import json
import re
from datetime import datetime
from pathlib import Path

from chop.envs import arena as arena_env
from chop.envs import common
from chop.paths import RESULTS_DIR
from chop.reports.csvio import EVAL_CSV, TRAIN_CSV, episodes_by_agent, normalize_agent, read_rows
from chop.reports.markdown import END, START, mean_ci, table, write_report

REPORT = Path("EXPERIMENTS_ARENA.md")
EXPERIMENTS_FILE = RESULTS_DIR / "experiments_arena.json"

TEMPLATE = f"""# CraftGround Experiments: Chop Trees (arena)

Goal: train an agent to chop as many oak logs as possible in one episode.

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""


def load_experiments() -> list[dict]:
    try:
        return json.loads(EXPERIMENTS_FILE.read_text(encoding="utf-8"))["experiments"]
    except (OSError, ValueError, KeyError):
        return []


def load_config(model: str) -> dict | None:
    """The config.json train.py saves next to each run's model.zip (a checkpoint: its run's)."""
    run_dir = Path(model).parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def setup_of(cfg: dict | None) -> tuple[str, int, bool]:
    """(actions "single"/"multi", game ticks per step, ground repaired on reset) from a run's config."""
    env = (cfg or {}).get("env", {})
    actions = "multi" if isinstance(env.get("actions"), dict) else "single"
    return actions, int(env.get("action_repeat", 1)), bool(env.get("ground_repaired_on_reset"))


def checkpoint_steps(path: str) -> int | None:
    m = re.search(r"_(\d+)_steps\.zip$", path)
    return int(m.group(1)) if m else None


def short_steps(n: int) -> str:
    """1.0M, 200k (nearest 10k from 100k up, so 200,508 and 202,488 steps both read 200k), 51k."""
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return f"{round(n, -4) // 1000}k" if n >= 1e5 else f"{round(n / 1e3)}k"


def score_cells(eps: list[int] | None) -> list:
    if not eps:
        return ["_not evaluated_", "", ""]
    mean, ci = mean_ci(eps)
    return [f"**{mean:.2f}** ± {ci:.2f} ({len(eps)})", min(eps), max(eps)]


class Runs:
    """Training runs and evaluation scores, looked up by run folder name or model path."""

    def __init__(self):
        self.train = {Path(normalize_agent(r["model"])).parent.name: r for r in read_rows(TRAIN_CSV)}
        self.evals = episodes_by_agent(read_rows(EVAL_CSV))

    def model_of(self, entry: str) -> str:
        """A run folder name means that run's final model.zip; anything else is a model path."""
        if entry in self.train:
            return normalize_agent(self.train[entry]["model"])
        return normalize_agent(entry)

    def row(self, entry: str) -> tuple[list, tuple[str, int, bool], list[int] | None]:
        model = self.model_of(entry)
        cfg = load_config(model)
        setup = setup_of(cfg)
        ckpt = checkpoint_steps(model)
        run = self.train.get(Path(model).parent.parent.name if ckpt else Path(model).parent.name)
        name = Path(model).parent.parent.name if ckpt else Path(model).parent.name
        if ckpt:
            label, steps, minutes = f"`{name}` @ {short_steps(ckpt)} (checkpoint)", short_steps(ckpt), "–"
        else:
            label = f"`{name}`"
            steps = short_steps(int(run["timesteps_done"])) if run else "?"
            minutes = run["minutes"] if run else "–"
            if run and run.get("resumed_from"):
                at = checkpoint_steps(run["resumed_from"])
                minutes += f" (resumed at {short_steps(at)})" if at else " (resumed)"
        wandb = f"[open]({run['wandb_url']})" if run and run.get("wandb_url") and not ckpt else "–"
        actions, ticks, repaired = setup
        eps = self.evals.get(model)
        cells = [label, actions, ticks, "yes" if repaired else "**no**", steps, minutes, *score_cells(eps), wandb]
        return cells, setup, eps


HEADERS = ["Run", "Actions", "Ticks/step", "Ground fixed", "Steps", "Train min", "Eval mean ± 95% (episodes)",
           "Min", "Max", "W&B"]


def by_setup_table(rows: list[tuple[tuple[str, int, bool], str, list[int] | None, bool]]) -> str:
    """When an experiment repeats a setup (same actions, ticks, ground and training steps), the
    average of those runs' scores (final models only)."""
    groups: dict[tuple[str, int, bool, str], list[float]] = {}
    for setup, steps, eps, is_final in rows:
        if eps and is_final:
            groups.setdefault((*setup, steps), []).append(sum(eps) / len(eps))
    if not any(len(v) > 1 for v in groups.values()):
        return ""
    out = [[f"{a}, {t} tick{'s' if t > 1 else ''}" + ("" if fixed else ", ground not fixed") + f", {steps}",
            len(v), f"**{sum(v) / len(v):.2f}**", " ".join(f"{x:.1f}" for x in v)]
           for (a, t, fixed, steps), v in groups.items()]
    return "\n\n" + table(["Setup", "Runs", "Average of runs", "Each run"], out)


def experiments_section(runs: Runs) -> tuple[str, set[str]]:
    listed: set[str] = set()
    blocks = []
    for exp in load_experiments():
        rows, setups = [], []
        for entry in exp.get("runs", []):
            listed.add(runs.model_of(entry))
            cells, setup, eps = runs.row(entry)
            rows.append(cells)
            setups.append((setup, cells[4], eps, checkpoint_steps(runs.model_of(entry)) is None))
        found = exp.get("found") or "_not yet_"
        blocks.append(f"### {exp['id']} · {exp['name']}\n\n**Question:** {exp.get('tested', '')}\n\n"
                      f"**Found:** {found}\n\n" + table(HEADERS, rows) + by_setup_table(setups))
    if not blocks:
        return "", listed
    return ("## Experiments\n\n**Eval** is the mean over `evaluate.py` episodes (max "
            f"{arena_env.MAX_LOGS}), ± its 95% range. **Ground fixed** = no: holes the agent dug "
            "stayed for the rest of the run (fixed 2026-11-17), so those scores are rough. Edit the "
            "questions and findings in `results/experiments_arena.json`.\n\n" + "\n\n".join(blocks)), listed


def other_section(runs: Runs, listed: set[str]) -> str:
    rows = [runs.row(name)[0] for name in reversed(list(runs.train))
            if runs.model_of(name) not in listed]
    rows += [runs.row(agent)[0] for agent in runs.evals
             if agent != "random" and agent not in listed and Path(agent).parent.name not in runs.train]
    if not rows:
        return ""
    return ("## Other runs\n\nRuns no experiment lists yet (add them to `results/experiments_arena.json`).\n\n"
            + table(HEADERS, rows))


def setup_section(runs: Runs) -> str:
    random_eps = runs.evals.get("random")
    random_text = (f"{sum(random_eps) / len(random_eps):.2f} logs over {len(random_eps)} episodes"
                   + (" (too few to trust)" if len(random_eps) < 20 else "") if random_eps
                   else "not evaluated: `run.bat evaluate.py --random --episodes 50`")
    turns = ", ".join(map(str, common.TURN_DEGREES))
    looks = ", ".join(map(str, common.LOOK_DEGREES))
    rows = [
        ["World", f"Superflat arena, stone-brick walls at ±{arena_env.ARENA_RADIUS} blocks"],
        ["Trees", f"{len(arena_env.TREE_POSITIONS)} oaks × {arena_env.TRUNK_HEIGHT} logs = "
                  f"{arena_env.MAX_LOGS} logs max, rebuilt every episode"],
        ["Ground", "Refilled every episode (since 2026-11-17; before, dug holes stayed)"],
        ["Tool", "Unbreakable diamond axe, Efficiency X (1 hit per log)"],
        ["Spawn", "Arena centre, random facing (15° steps)"],
        ["Observation", f"RGB image {common.IMAGE_WIDTH}×{common.IMAGE_HEIGHT}"],
        ["Single actions (default)", f"{len(common.ACTION_NAMES)}, one per step: " + ", ".join(common.ACTION_NAMES)
                                     + "; turns are 15°"],
        ["Multi actions (`--actions multi`)", f"every step: move (none, forward, back) × turn ({turns}°) × "
                                              f"look ({looks}°), attack always held"],
        ["Ticks per step", "1 (default); `--action-repeat N` holds each action for N ticks"],
        ["Reward", arena_env.REWARD],
        ["Episode length", f"{common.EPISODE_STEPS} ticks (or all logs chopped)"],
        ["Random actions", random_text],
    ]
    return "## Setup\n\n" + table(["Setting", "Value"], rows)


def update(stamp: str | None = None) -> None:
    """Rebuild EXPERIMENTS_ARENA.md's generated part from the arena CSVs and results/experiments_arena.json."""
    stamp = stamp or f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"
    runs = Runs()
    experiments, listed = experiments_section(runs)
    parts = [stamp, setup_section(runs), experiments, other_section(runs, listed)]
    write_report(REPORT, TEMPLATE, "\n\n".join(p for p in parts if p))
