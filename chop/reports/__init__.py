"""The CSVs in results/ and the Markdown reports built from them.

    csvio     the result CSVs: appending and reading rows
    markdown  tables, and rewriting a report's AUTO-GENERATED block (your Notes are kept)
    agents    which model is which: labels, steps, natural-world training lines
    sections  sections both reports use (evaluation runs, training runs)
    arena     docs/arena/ (README.md index + one page per table)
    natural   docs/natural/ (README.md index + one page per table)
    minerl    docs/minerl/ (MineRL models in CraftGround, MineRL-style scoring)

train.py and evaluate.py append a row, then call update_report(env).
"""

from chop.reports import arena, minerl, natural


def update_report(env: str = "arena") -> None:
    """Rebuild the tables of one report: "arena", "natural" or "minerl"."""
    {"arena": arena.update, "natural": natural.update, "minerl": minerl.update}[env]()
