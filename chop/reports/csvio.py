"""The CSV files in results/ that train.py and evaluate.py append to.

Each row stores the model path relative to the repository root ("runs/..."), which is also
how agents are told apart in the reports (normalize_agent)."""

import csv
from pathlib import Path

from chop.envs import natural as natural_env
from chop.paths import RESULTS_DIR


EVAL_CSV = RESULTS_DIR / "eval_results.csv"

TRAIN_CSV = RESULTS_DIR / "training_runs.csv"

NATURAL_EVAL_CSV = RESULTS_DIR / "eval_results_natural.csv"

NATURAL_TRAIN_CSV = RESULTS_DIR / "training_runs_natural.csv"

MINERL_EVAL_CSV = RESULTS_DIR / "eval_results_minerl.csv"  # MineRL-mode evaluations, either world

MINERL_TRAIN_CSV = RESULTS_DIR / "training_runs_minerl.csv"  # MineRL-mode training runs, either world


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


def normalize_agent(agent: str) -> str:
    """'runs\\x\\model.zip', './runs/x/model.zip' and 'runs/x/model.zip' are the same agent.
    "" (no model, e.g. a run from scratch) stays "", not Path's "."."""
    return agent if agent in ("random", "") else Path(agent).as_posix().removeprefix("./")


def episodes_by_agent(rows: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in rows:
        out.setdefault(normalize_agent(r["agent"]), []).extend(int(x) for x in r["per_episode"].split())
    return out


def is_standard_natural(row: dict) -> bool:
    """A MineRL-CSV row evaluated exactly like the natural report's own rows: natural world,
    1000-tick episodes, no log cap, our one-hit axe. Those belong in the natural tables too."""
    return (row.get("env") == "natural" and str(row.get("episode_ticks")) == "1000"
            and not row.get("log_cap") and row.get("axe") == "instant")


def is_fixed(row: dict) -> bool:
    return (row.get("spots") or "").startswith("fixed")


def is_wooded(row: dict) -> bool:
    """Played the spot set in every biome with trees (eval_spots_natural_wooded.json, made with
    make_eval_spots.py --spawn wooded), not the oak/birch one. Not comparable with the oak/birch spots,
    so it gets its own page."""
    return is_fixed(row) and (row.get("spots") or "").endswith("_wooded.json")


def is_anywhere(row: dict) -> bool:
    """Played the spot set anywhere on land (eval_spots_natural_anywhere.json, made with
    make_eval_spots.py --spawn anywhere): plains, desert, mountains... included. Its own page."""
    return is_fixed(row) and (row.get("spots") or "").endswith("_anywhere.json")


def is_held_out(row: dict) -> bool:
    """Played in a world other than the training one (blank = older rows, all in the training world)."""
    return (row.get("world_seed") or natural_env.SEED) != natural_env.SEED
