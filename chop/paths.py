"""Where results and runs live.

RESULTS_DIR and RUNS_DIR are relative to the working directory (the repository root, which
run.bat changes to): model paths are stored in the CSVs as "runs/...", so they must stay relative.
ROOT is absolute, for files that must be found wherever the code is run from.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path("results")
DOCS_DIR = Path("docs")  # the reports (EXPERIMENTS_*.md, BENCHMARK.md) and FINDINGS.md
RUNS_DIR = Path("runs")
NATURAL_RUNS = RUNS_DIR / "natural"
