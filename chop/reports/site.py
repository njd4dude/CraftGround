"""The docs/ site: which generated section goes on which page. update() rebuilds every page.

    docs/results/      comparison tables, one per set of rules, every agent in one place
    docs/experiments/  why each run was made (arena experiments, natural training lines); the MineRL
                       transfer page there is written by hand
    docs/reference/    how the environments are set up (speed and parallel launch are separate,
                       hand-written / benchmark.py pages)
    docs/logs/         every evaluation and training run, newest first

A page keeps whatever is written outside its AUTO-GENERATED block (notes), so only the tables
are rebuilt. docs/FINDINGS.md and the root README.md are written by hand."""

from datetime import datetime
from pathlib import Path

from chop.paths import DOCS_DIR
from chop.reports import arena, minerl, natural
from chop.reports.agents import model_label
from chop.reports.csvio import (EVAL_CSV, MINERL_EVAL_CSV, MINERL_TRAIN_CSV, NATURAL_EVAL_CSV, NATURAL_TRAIN_CSV,
                                TRAIN_CSV, is_fixed, is_held_out, is_standard_natural, is_wooded, read_rows)
from chop.reports.markdown import END, START, body, write_report
from chop.reports.sections import evals_section, training_section


def page_template(title: str, about: str, depth: int = 1) -> str:
    up = "../" * depth
    return f"""# {title}

{about}

[← Project README]({up}../README.md) · [Findings]({up}FINDINGS.md)

{START}
{END}
"""


def write_page(rel: str, title: str, about: str, generated: str, stamp: str) -> None:
    path = DOCS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    write_report(path, page_template(title, about, depth=rel.count("/")), f"{stamp}\n\n{generated}")


def section(heading: str, text: str) -> str:
    """A section under its own heading (the section functions name themselves differently)."""
    return f"## {heading}\n\n{body(text) or '_None yet._'}"


def update(stamp: str | None = None) -> None:
    stamp = stamp or f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"
    natural_evals, natural_runs = read_rows(NATURAL_EVAL_CSV), read_rows(NATURAL_TRAIN_CSV)
    arena_evals, minerl_evals = read_rows(EVAL_CSV), read_rows(MINERL_EVAL_CSV)
    models = minerl.load_models()

    # MineRL rows played under our standard rules join our agents' tables.
    minerl_natural = [r for r in minerl_evals if is_standard_natural(r) and is_fixed(r) and not is_held_out(r)]
    minerl_arena = [r for r in minerl_evals if r.get("env") == "arena" and r.get("axe") == "instant"
                    and str(r.get("episode_ticks")) == "1000"]
    training_world = [r for r in natural_evals if not is_held_out(r)]
    forest = [r for r in training_world + minerl_natural if is_fixed(r) and not is_wooded(r)]
    wooded = [r for r in training_world + minerl_natural if is_wooded(r)]
    random_forest = [r for r in training_world if not is_fixed(r) and r.get("spots") != "random_wooded"]
    random_wooded = [r for r in training_world if r.get("spots") == "random_wooded"]
    arena_runs = arena.Runs()

    def minerl_label(agent: str) -> str:
        return minerl.label(agent, models, natural_runs)

    pages = [
        ("results/forest_spots.md", "Results: forest spots",
         "The main comparison: every agent on the same 100 oak/birch forest spots in the natural world.",
         natural.fixed_spots_section(forest, natural_runs)),
        ("results/wooded_spots.md", "Results: wooded spots",
         "Every agent on 100 spots in any biome with trees, next to its forest score.",
         natural.wooded_section(wooded, forest, natural_runs)),
        ("results/arena.md", "Results: arena",
         "Every agent in the walled arena (8 trees, 32 logs at most).",
         arena.results_section(arena_runs, minerl_arena, minerl_label)),
        ("results/other_rules.md", "Results: other rules",
         "Scores that don't compare with the main tables: MineRL's own scoring and other episode rules, "
         "random spawns, and the held-out world.",
         "\n\n".join([
             minerl.results_section(minerl_evals, models, natural_runs),
             natural.natural_random_spawn_section(random_forest, random_wooded, arena_evals, natural_runs)
             or "## Random spawns\n\n_None yet._",
             "## Held-out world\n\nArchived 2026-11-19: agents score the same in a world they never trained in. "
             "See [Held-out world](archive/held_out.md).",
         ])),
        ("experiments/natural_lines.md", "Experiments: natural-world training lines",
         "Every natural-world training line: what it tested, what it found, and its pieces.",
         natural.lines_section(forest, random_forest, natural_runs)),
        ("experiments/arena_experiments.md", "Experiments: arena",
         "Every arena experiment: its question, what it found, and its runs.",
         "\n\n".join(s for s in [arena.experiments_section(arena_runs)[0],
                                 arena.other_section(arena_runs, arena.experiments_section(arena_runs)[1])] if s)),
        ("reference/environments.md", "Reference: environments",
         "How the arena, the natural world and MineRL mode are set up.",
         "\n\n".join([arena.setup_section(arena_runs), natural.natural_setup_section(), minerl.setup_section()])),
        ("logs/evaluations.md", "Log: evaluation runs",
         "Every `evaluate.py` run, newest first. Each run's every episode is in the CSVs in `results/`.",
         "\n\n".join([section("Natural world", evals_section(natural_evals)),
                      section("Arena", evals_section(arena_evals)),
                      section("MineRL mode and other rules", minerl.evals_section(minerl_evals, models,
                                                                                  natural_runs))])),
        ("logs/training.md", "Log: training runs",
         "Every `train.py` run that ended normally, newest first. Stopped pieces are listed on the "
         "experiment pages.",
         "\n\n".join([section("Natural world", training_section(natural_runs)),
                      section("Arena", training_section(read_rows(TRAIN_CSV))),
                      section("MineRL mode", minerl.training_section(read_rows(MINERL_TRAIN_CSV)))])),
    ]
    for rel, title, about, generated in pages:
        write_page(rel, title, about, generated, stamp)


if __name__ == "__main__":
    update()
