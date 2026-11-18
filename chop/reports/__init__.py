"""The CSVs in results/ and the Markdown reports built from them.

    csvio     the result CSVs: appending and reading rows
    markdown  tables, and rewriting a report's AUTO-GENERATED block (your Notes are kept)
    agents    which model is which: labels, steps, natural-world training lines
    sections  sections both reports use (evaluation runs, training runs)
    arena     docs/EXPERIMENTS_ARENA.md
    natural   docs/EXPERIMENTS_NATURAL.md

train.py and evaluate.py append a row, then call update_report(env).
"""

from chop.reports import arena, natural


def update_report(env: str = "arena") -> None:
    """Rebuild the tables of one environment's report: "arena" or "natural"."""
    {"arena": arena.update, "natural": natural.update}[env]()
