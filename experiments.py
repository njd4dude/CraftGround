"""Keeps EXPERIMENTS.md up to date.

evaluate.py and train.py append rows to the CSV files in results/, then call
update_report(), which rewrites the tables in EXPERIMENTS.md. Only the part
between the AUTO-GENERATED markers is replaced, so notes you write anywhere
else in the file are kept.

Run it by hand to rebuild the tables:  python experiments.py
"""

import csv
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
        ["Reward", "+1 per log chopped"],
        ["Episode length", f"{tree_env.EPISODE_STEPS} steps (or all logs chopped)"],
    ]
    return "## Environment setup\n\n" + table(["Setting", "Value"], rows)


def summary_section(evals: list[dict]) -> str:
    if not evals:
        return "## Summary\n\n_No evaluations yet. Run `run.bat evaluate.py --random` for a baseline._"
    by_agent: dict[str, list[dict]] = {}
    for r in evals:
        by_agent.setdefault(r["agent"], []).append(r)

    def pooled(rows: list[dict]) -> list[int]:
        return [int(x) for r in rows for x in r["per_episode"].split()]

    random_eps = pooled(by_agent.get("random", []))
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None

    out = []
    for agent, rows in by_agent.items():
        eps = pooled(rows)
        mean = sum(eps) / len(eps)
        vs = "baseline" if agent == "random" else (
            f"{mean / random_mean:.1f}× random" if random_mean else "n/a")
        out.append([f"`{agent}`", len(eps), f"{mean:.2f}", min(eps), max(eps), vs])
    out.sort(key=lambda r: -float(r[2]))
    return ("## Summary\n\nAll evaluation episodes pooled per agent, best first.\n\n"
            + table(["Agent", "Episodes", "Mean logs", "Min", "Max", "vs random"], out))


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
        summary_section(evals),
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
