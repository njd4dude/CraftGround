"""Set the W&B group and lineage tags (chop/training/lineage.py) on runs logged before train.py did.

Every run gets its group, plus a "lineage:<group>" tag for its own lineage and for each later run
that continues it. Filter the workspace by e.g. lineage:line2b_linear-lr-from-601k to see Line 2b
and the Line 2 pieces it came from. Prints the changes; --apply makes them.

    run.bat tools\\wandb_lineage.py
    run.bat tools\\wandb_lineage.py --apply
"""

import argparse
from collections import defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
import wandb

from chop.paths import ROOT, RUNS_DIR
from chop.training.lineage import group_of, history, lineage_tag, resumed_from, wandb_run


def all_runs() -> list[Path]:
    """Every run folder with a config.json, as relative paths like runs/natural/<line>/<time>."""
    configs = [*(ROOT / RUNS_DIR / "natural").glob("*/*/config.json"), *(ROOT / RUNS_DIR / "arena").glob("*/config.json")]
    return sorted(c.parent.relative_to(ROOT) for c in configs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="make the changes (default: only print them)")
    args = parser.parse_args()

    groups: dict[str, str | None] = {}
    tags: dict[str, set[str]] = defaultdict(set)
    chains = {run_dir.as_posix(): history(resumed_from(run_dir)) for run_dir in all_runs()}
    for key, chain in chains.items():
        groups[key] = group_of(Path(key), chain)
    for key, chain in chains.items():
        lineage = lineage_tag(Path(key), chain)
        if lineage is None:
            continue
        for r in [*chain, Path(key)]:
            tags[r.as_posix()].add(lineage)
            if groups.get(r.as_posix()) is None:
                groups[r.as_posix()] = groups[key]  # an arena run a later run continues

    api = wandb.Api()
    changed = 0
    for key, group in groups.items():
        try:
            run = wandb_run(api, Path(key))
        except wandb.errors.CommError:  # deleted on W&B
            run = None
        if run is None:
            print(f"{key}: not on W&B, skipped")
            continue
        new_tags = sorted(tags[key] - set(run.tags))
        if run.group == group and not new_tags:
            continue
        changed += 1
        print(f"{key} ({run.id}): group {run.group!r} -> {group!r}" + (f", + tags {new_tags}" if new_tags else ""))
        if args.apply:
            run.group = group
            run.tags = [*run.tags, *new_tags]
            run.update()
    print(f"{changed} run(s) {'updated' if args.apply else 'to update; --apply to make the changes'}")


if __name__ == "__main__":
    main()
