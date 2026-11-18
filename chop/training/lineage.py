"""Which earlier runs a run continues, and how W&B groups and tags them to show it.

W&B doesn't know that a resumed run continues another one, so each run gets:
- a group: its natural-world line folder (e.g. line2_jump-constant-lr), or in the arena the folder
  of the first run in its chain. A line's pieces then chart as one series;
- a "lineage:<group>" tag, also added to every run it continues. Filtering by Line 2b's tag
  shows the whole history behind it, including the Line 2 pieces it branched from.

train.py sets these on new runs; tools/wandb_lineage.py sets them on runs logged before.
"""

import json
from pathlib import Path

from chop.paths import NATURAL_RUNS, ROOT

TAG_LIMIT = 64  # W&B's longest tag


def run_dir_of(model_path: str | Path) -> Path:
    """The run folder a model .zip (its model.zip or a checkpoint) belongs to."""
    p = Path(model_path)
    return p.parent.parent if p.parent.name == "checkpoints" else p.parent


def resumed_from(run_dir: Path) -> str:
    """The model a run started from, from its config.json ("" if from scratch or unknown)."""
    config = ROOT / run_dir / "config.json"
    if not config.exists():
        return ""
    return json.loads(config.read_text(encoding="utf-8")).get("resumed_from") or ""


def history(resume: str | None) -> list[Path]:
    """The run folders a model started from `resume` continues, oldest first."""
    chain: list[Path] = []
    while resume:
        run = run_dir_of(resume)
        if any(r.as_posix() == run.as_posix() for r in chain):
            break  # a loop in the configs; shouldn't happen
        chain.insert(0, run)
        resume = resumed_from(run)
    return chain


def group_of(run_dir: Path, chain: list[Path]) -> str | None:
    """W&B group: the natural-world line folder, else the first run of the chain (None for an
    arena run that continues nothing; it gets the group if a later run continues it)."""
    if run_dir.parent.parent.as_posix() == NATURAL_RUNS.as_posix():  # runs/natural/<line>/<time>
        return run_dir.parent.name
    return chain[0].name if chain else None


def lineage_tag(run_dir: Path, chain: list[Path]) -> str | None:
    group = group_of(run_dir, chain)
    return ("lineage:" + group)[:TAG_LIMIT] if group else None


def wandb_id(run_dir: Path) -> str | None:
    """The W&B run id, from the run-<time>-<id> folder wandb.init left in the run folder."""
    runs = sorted((ROOT / run_dir / "wandb").glob("run-*"))
    return runs[-1].name.rsplit("-", 1)[-1] if runs else None


def wandb_run(api, run_dir: Path):
    """The run on W&B (wandb.apis.public.Run), or None if it never logged there."""
    from chop.training.settings import WANDB_PROJECT

    run_id = wandb_id(run_dir)
    return api.run(f"{api.default_entity}/{WANDB_PROJECT}/{run_id}") if run_id else None


def tag_runs(run_dirs: list[Path], tag: str, group: str) -> None:
    """Add `tag` on W&B to each run that logged there, and `group` to those that have none."""
    import wandb

    api = wandb.Api()
    for run_dir in run_dirs:
        run = wandb_run(api, run_dir)
        if run is None or (tag in run.tags and run.group):
            continue
        if tag not in run.tags:
            run.tags = [*run.tags, tag]
        run.group = run.group or group
        run.update()
