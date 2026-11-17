"""Which natural-world training line (runs/natural/<line folder>/) a run belongs to.

A line is one training history (see results/lines_natural.json): --resume continues the line its
model came from, unless --line names a different one (a branch); a fresh run starts a new line.
Arena runs go to runs/arena/<name>_<time>/ instead.
"""

import re
from pathlib import Path

from chop.paths import NATURAL_RUNS


def natural_line_dir(resume: str | None, line: str | None) -> Path:
    """The line folder a natural-world run belongs in."""
    if line:
        return NATURAL_RUNS / line
    if resume:
        src = Path(resume)
        run = src.parent.parent if src.parent.name == "checkpoints" else src.parent
        if run.parent.parent.resolve() == NATURAL_RUNS.resolve():
            return run.parent  # continue that model's line
    taken = [int(m.group(1)) for d in NATURAL_RUNS.glob("line*")
             if (m := re.match(r"line(\d+)", d.name))]
    return NATURAL_RUNS / f"line{max(taken, default=0) + 1}"
