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
    """Write a page: its title and intro (always from here, so renaming a page here renames it there),
    the generated block, then whatever was written by hand below the block."""
    path = DOCS_DIR / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    template = page_template(title, about, depth=rel.count("/"))
    text = None
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if START in text and END in text:
            text = template.split(START)[0] + START + text.split(START, 1)[1]
    write_report(path, template, f"{stamp}\n\n{generated}", text)


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
        ("results/natural_oak_birch_forests.md", "Natural world: oak and birch forests (100 fixed spots)",
         "The main comparison: every agent on the same 100 spots in the natural world's oak and birch forests "
         "(`results/eval_spots_natural.json`).",
         natural.fixed_spots_section(forest, natural_runs)),
        ("results/natural_all_tree_biomes.md", "Natural world: every biome with trees (100 fixed spots)",
         "Every agent on 100 spots spread over every biome with trees (taiga, snowy taiga, jungle, dark forest, "
         "swamp, cherry grove...; no savanna; `results/eval_spots_natural_wooded.json`), next to its "
         "oak/birch forest score.",
         natural.wooded_section(wooded, forest, natural_runs)),
        ("results/arena.md", "Arena: walled flat arena with 8 oaks",
         "Every agent in the one arena: superflat, stone-brick walls, 8 oaks of 4 logs (32 logs at most), "
         "rebuilt every episode. Only the agents' actions differ between runs.",
         arena.results_section(arena_runs, minerl_arena, minerl_label)),
        ("results/other_evaluations.md", "Other evaluations: different rules",
         "Scores that don't compare with the three main pages because the rules differ: MineRL's own scoring "
         "(4,000 ticks, iron axe, ends at 64 logs), the iron axe, long episodes, random spawns instead of fixed "
         "spots, and the held-out world.",
         "\n\n".join([
             minerl.results_section(minerl_evals, models, natural_runs),
             natural.natural_random_spawn_section(random_forest, random_wooded, arena_evals, natural_runs)
             or "## Random spawns\n\n_None yet._",
             "## Held-out world\n\nArchived 2026-11-19: agents score the same in a world they never trained in. "
             "See [held-out world](archive/held_out.md).",
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
                      section("MineRL mode and other evaluations", minerl.evals_section(minerl_evals, models,
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

    # One README.md per folder, listing its pages: editors can't open a folder from a link, and GitHub
    # shows a folder's README.md under its file list. Hand-written pages are listed here too.
    listed = [(rel, title, about) for rel, title, about, _ in pages] + HAND_WRITTEN
    for folder, (title, about) in FOLDERS.items():
        items = [f"- [{t}]({Path(rel).relative_to(folder).as_posix()}): {a}"
                 for rel, t, a in listed if Path(rel).parent.as_posix().startswith(folder)]
        write_page(f"{folder}/README.md", title, about, "\n".join(items), stamp)


# Pages written by hand (or by benchmark.py), for the folder indexes: (path under docs/, title, about).
HAND_WRITTEN = [
    ("experiments/minerl_transfer.md", "MineRL models in CraftGround",
     "how well MineRL-trained models transfer, and what fine-tuning them here gives (written by hand)"),
    ("reference/speed.md", "Environment speed", "steps per second of each world (written by benchmark.py)"),
    ("reference/parallel_launch.md", "Starting several Minecraft instances at once",
     "why parallel launches hung, and what the code does about it (written by hand)"),
    ("results/archive/held_out.md", "Held-out world (archived)",
     "agents score the same in a world they never trained in (written by hand)"),
]

FOLDERS = {
    "results": ("Results", "Every agent compared, one page per environment, plus evaluations under other rules."),
    "experiments": ("Experiments", "Why each run was made: its question, what it found, its runs."),
    "reference": ("Reference", "How the environments are set up, how fast they run, and launching instances."),
    "logs": ("Logs", "Every evaluation and training run, newest first."),
}


if __name__ == "__main__":
    update()
