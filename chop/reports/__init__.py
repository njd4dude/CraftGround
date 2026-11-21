"""The CSVs in results/ and the docs/ pages built from them.

    csvio     the result CSVs: appending and reading rows
    markdown  tables, and rewriting a page's AUTO-GENERATED block (notes outside it are kept)
    agents    which model is which: labels, steps, natural-world training lines, MineRL models
    compare   spot-by-spot comparisons on the fixed evaluation spots
    sections  log sections (every evaluation run, every training run)
    arena     arena sections: results, experiments, setup
    natural   natural-world sections: fixed-spot results, training lines, random spawns, setup
    minerl    MineRL-mode sections: other episode rules, logs, setup
    site      which section goes on which page of docs/; rebuilds them all

train.py and evaluate.py append a row, then call update_report().
"""

from chop.reports import site


def update_report(env: str = "") -> None:
    """Rebuild every generated page in docs/. (env is ignored: every page can draw on every world's
    results, so they're always rebuilt together; it stays so older calls keep working.)"""
    site.update()
