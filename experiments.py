"""Rebuild docs/EXPERIMENTS_ARENA.md (arena) and docs/natural/ (natural world, one page per table) from results/.

    python experiments.py

train.py and evaluate.py already do this after every run; run it by hand after editing
results/lines_natural.json or a run's config.json notes. Only the part between the
AUTO-GENERATED markers is replaced, so notes you write anywhere else in the file are kept.
The code is in chop/reports/.
"""

from chop.reports import update_report

if __name__ == "__main__":
    update_report("arena")
    update_report("natural")
