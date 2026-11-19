"""docs/natural/: the natural world's report, one page per table.

README.md is the index (and keeps your Notes); every other page is rebuilt from the CSVs. Like any
report, a page keeps what you write outside its AUTO-GENERATED block."""

import re
from datetime import datetime
from pathlib import Path

from chop.envs import common
from chop.envs import natural as natural_env
from chop.reports.agents import (label_with_folder, line_folders, line_meta, line_note,
                                 load_lines, model_label, pieces, run_config, run_dir_of, run_end, short_steps,
                                 start_of)
from chop.reports.csvio import (EVAL_CSV, NATURAL_EVAL_CSV, NATURAL_TRAIN_CSV, episodes_by_agent, is_fixed, is_held_out,
                                is_wooded, normalize_agent, read_rows)
from chop.reports.markdown import END, START, mean_ci, table, write_report
from chop.reports.sections import evals_section
from chop.paths import DOCS_DIR

NATURAL_DIR = DOCS_DIR / "natural"
NATURAL_REPORT = NATURAL_DIR / "README.md"  # the index

# (file, title, what it holds) in index order; update() fills each from its section function.
PAGES = [
    ("results.md", "Results", "the 100 fixed forest spots, one row per agent: the main comparison"),
    ("wooded.md", "Wooded spots", "100 fixed spots in any biome with trees (snowy taiga, taiga, jungle, swamp...), "
                                  "next to each agent's forest score"),
    ("held_out.md", "Held-out world", "fixed spots in a world training never sees: does the agent generalise?"),
    ("lines.md", "Training lines", "every training line: what it tested, what it found, its pieces"),
    ("evaluation_runs.md", "Evaluation runs", "one row per evaluate.py run, newest first"),
    ("random_spawns.md", "Earlier random-spawn evaluations", "historical, before the fixed spots (2026-11-16)"),
    ("setup.md", "Environment setup", "world, spawns, actions, reward, episode length"),
]

INDEX_TEMPLATE = f"""# CraftGround Experiments: Chop Trees in a Natural World

Goal: chop as many logs as possible in one episode, starting in a real generated
forest instead of the walled arena (chop/envs/natural.py). Arena results are in
[EXPERIMENTS_ARENA.md](../EXPERIMENTS_ARENA.md); the summary of findings is [FINDINGS.md](../FINDINGS.md).

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""


def page_template(title: str) -> str:
    return f"""# Natural world: {title}

[← Natural world index](README.md)

{START}
{END}
"""


def body(section: str) -> str:
    """A section without its own "## Title" line (the page's title replaces it)."""
    return re.sub(r"\A## [^\n]*\n+", "", section)


def natural_setup_section() -> str:
    ne = natural_env
    rows = [
        ["World", f"Normal generated world, fixed seed {ne.SEED} (held-out evaluation world: "
                  f"seed {ne.HELD_OUT_SEED}, never trained in)"],
        ["Evaluation", "100 fixed forest spots (`results/eval_spots_natural.json`, `make_eval_spots.py`); "
                       "`evaluate.py --random-spots` for random spawns. Held-out world: "
                       "`make_eval_spots.py --held-out` once, then `evaluate.py <model> --env natural "
                       "--held-out`. Wooded spots: see [Wooded spots](wooded.md)"],
        ["Spawn", f"A new spot every episode, on the ground in an oak/birch forest: first within "
                  f"{ne.HOP_RANGE} blocks of the last spot, else anywhere within ±{ne.FAR_RANGE:,}. "
                  f"`train.py --spawn wooded` (Line 5, since 2026-11-18): any biome with trees but savanna, "
                  f"and a world-wide search after {ne.MAX_HOP_EPISODES['wooded']} episodes near one spot"],
        ["Rebuilt each episode", "no: chopped trees stay chopped, so every episode moves to fresh forest"],
        ["Boundaries", "none; terrain loads as the agent walks"],
        ["Render distance", f"{ne.RENDER_DISTANCE} chunks, Minecraft heap {ne.JVM_MAX_MEMORY} "
                            "(since 2026-11-15, to fit 4 instances in memory; earlier runs: 6 chunks, 3G)"],
        ["Tool", "Unbreakable diamond axe, Efficiency X (1 hit per log)"],
        ["Observation", f"RGB image {common.IMAGE_WIDTH}×{common.IMAGE_HEIGHT} (same as the arena)"],
        ["Actions", f"{len(common.ACTION_NAMES_JUMP)} discrete: the arena's {len(common.ACTION_NAMES)} plus "
                    + ", ".join(common.JUMP_ACTIONS) + " (since 2026-11-15; earlier runs and the random "
                    "baseline use the arena's 11)"],
        ["Reward", ne.REWARD],
        ["Damage", "Lava, fire, drowning etc. hurt and dying ends the episode; no fall damage "
                   "(since 2026-11-16; earlier runs: player immune to all damage)"],
        ["Episode length", f"{common.EPISODE_STEPS} steps (no log cap), or until the agent dies"],
    ]
    return "## Environment setup\n\n" + table(["Setting", "Value"], rows)


def fixed_spots_section(evals: list[dict], runs: list[dict] | None = None) -> str:
    """One row per agent from fixed-spot evaluations (evaluate.py's default in the natural world
    since 2026-11-16): every agent played the same forests, so these means compare fairly."""
    runs = runs or []
    by_agent = episodes_by_agent(evals)
    if not by_agent:
        return ("## Results\n\n_None yet: `run.bat evaluate.py <model.zip> --env natural` "
                "plays the 100 fixed spots in `results/eval_spots_natural.json`._")
    random_eps = by_agent.get("random")
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", -sum(kv[1]) / len(kv[1]))):
        n = len(eps)
        mean, ci = mean_ci(eps)
        median = sorted(eps)[n // 2]
        vs = ("baseline" if agent == "random"
              else f"**{mean / random_mean:.1f}×**" if random_mean else "no baseline yet")
        rows.append([label_with_folder(agent, runs), line_note(agent), n, f"**{mean:.2f}** ± {ci:.2f}", median,
                     sum(x == 0 for x in eps), max(eps), vs])
    return (
        "## Results\n\n"
        "Every agent here played the same 100 forest spots (`results/eval_spots_natural.json`, one "
        "episode each, same starting direction, current setup), so differences come from the agents, "
        "not from where they spawned. Best first. **± is the 95% range** of the mean: two agents whose "
        "ranges overlap a lot may not really differ. Older random-spawn scores are in "
        "[Earlier random-spawn evaluations](random_spawns.md).\n\n"
        + table(["Model", "Line", "Episodes", "Mean logs", "Median", "0-log episodes", "Max",
                 "vs random"], rows)
    )


def held_out_section(held_out: list[dict], training_fixed: list[dict], runs: list[dict] | None = None) -> str:
    """Fixed-spot scores in the held-out world next to the same agent's fixed-spot score in the
    training world: the gap is how much of the score depends on having trained in that world."""
    runs = runs or []
    by_agent = episodes_by_agent([r for r in held_out if is_fixed(r)])
    if not by_agent:
        return ""  # shown once there are results; how to get some is in Environment setup
    train = episodes_by_agent(training_fixed)
    random_eps = by_agent.get("random")
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", -sum(kv[1]) / len(kv[1]))):
        mean, ci = mean_ci(eps)
        if agent in train:
            t_mean, t_ci = mean_ci(train[agent])
            t_cell, gap = f"{t_mean:.2f} ± {t_ci:.2f}", f"{mean - t_mean:+.2f}"
        else:
            t_cell, gap = "–", "–"
        vs = ("baseline" if agent == "random"
              else f"**{mean / random_mean:.1f}×**" if random_mean else "no baseline yet")
        rows.append([label_with_folder(agent, runs), line_note(agent), len(eps), f"**{mean:.2f}** ± {ci:.2f}",
                     t_cell, gap, vs])
    return (
        "## Held-out world\n\n"
        f"Fixed spots in a world training never uses (seed {natural_env.HELD_OUT_SEED}, "
        f"`results/{natural_env.eval_spots_path(natural_env.HELD_OUT_SEED).name}`). **Training world** is "
        "the same agent's fixed-spot score in the world it trained in. A **gap** near 0 means the agent "
        "learned to chop trees in general; a clearly negative gap (larger than the ± ranges) means part "
        "of its score comes from knowing the training world. vs random uses the random baseline in this "
        "world.\n\n"
        + table(["Model", "Line", "Episodes", "Held-out mean", "Training world", "Gap", "vs random"], rows)
    )


def wooded_section(wooded: list[dict], forest_fixed: list[dict], runs: list[dict] | None = None) -> str:
    """Fixed-spot scores on the wooded spot set (any biome with trees: snowy taiga, taiga, jungle,
    swamp...) next to the same agent's forest-spot score: the gap is how much it loses outside
    oak/birch forests."""
    runs = runs or []
    by_agent = episodes_by_agent(wooded)
    if not by_agent:
        return ""
    forest = episodes_by_agent(forest_fixed)
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", -sum(kv[1]) / len(kv[1]))):
        mean, ci = mean_ci(eps)
        if agent in forest:
            f_mean, f_ci = mean_ci(forest[agent])
            f_cell, gap = f"{f_mean:.2f} ± {f_ci:.2f}", f"{mean - f_mean:+.2f}"
            f_zeros = sum(x == 0 for x in forest[agent])
        else:
            f_cell, gap, f_zeros = "–", "–", "–"
        rows.append([label_with_folder(agent, runs), line_note(agent), len(eps), f"**{mean:.2f}** ± {ci:.2f}",
                     f_cell, gap, sum(x == 0 for x in eps), f_zeros])
    return (
        "## Wooded spots\n\n"
        "100 fixed spots in any biome with trees except savanna (`results/eval_spots_natural_wooded.json`, "
        "`make_eval_spots.py --spawn wooded`): snowy taiga, taiga, dark forest, jungle, swamp and more, "
        "same training world. **Forest mean** is the same agent's score on the [Results](results.md) page's oak/birch "
        "spots; the **gap** is how much it loses outside them. **0-log episodes** count the episodes "
        "that chopped nothing on each set (out of 100: the agent got stuck, lost or never found a tree). Run with "
        "`evaluate.py <model> --env natural --spots results/eval_spots_natural_wooded.json`.\n\n"
        + table(["Model", "Line", "Episodes", "Wooded mean", "Forest mean", "Gap", "0-log episodes (wooded)",
                 "0-log episodes (forest)"], rows)
    )


def lines_section(fixed: list[dict], random_spawns: list[dict], runs: list[dict]) -> str:
    """One block per training line: what it tested, then its pieces in order."""
    line_dirs = [d for d in line_folders() if pieces(d)]
    if not line_dirs:
        return ""
    notes = load_lines().get("piece_notes", {})
    fixed_by, random_by = episodes_by_agent(fixed), episodes_by_agent(random_spawns)
    blocks = []
    for line_dir in line_dirs:
        meta = line_meta(line_dir.name)
        rows = []
        for run_dir in pieces(line_dir):
            cfg = run_config(run_dir) or {}
            start, origin = start_of(run_dir, runs)
            end, exact = run_end(run_dir, runs)
            span = (f"{short_steps(start) if start else '0'} → "
                    f"{'' if exact else '~'}{short_steps(end) if end else '?'}")
            row = next((r for r in runs if normalize_agent(r["model"]) == (run_dir / "model.zip").as_posix()), None)
            scores = []
            for agent in sorted(set(fixed_by) | set(random_by)):
                if run_dir_of(agent).as_posix() != run_dir.as_posix():
                    continue
                at = model_label(agent, runs).split(" @ ")[-1]
                if agent in fixed_by:
                    scores.append(f"**{sum(fixed_by[agent]) / len(fixed_by[agent]):.1f}** @ {at}")
                else:
                    scores.append(f"{sum(random_by[agent]) / len(random_by[agent]):.1f}* @ {at}")
            rows.append([f"`{run_dir.name}`", origin, span, cfg.get("n_envs", "–"),
                         cfg.get("lr_schedule", "constant"),
                         row["minutes"] if row and row.get("minutes") else "–",
                         row["status"] if row else "no record (killed or hung)", ", ".join(scores) or "–",
                         notes.get(run_dir.name, ""),
                         f"[open]({row['wandb_url']})" if row and row.get("wandb_url") else "–"])
        blocks.append(f"<a id=\"line-{meta['id']}\"></a>\n\n### Line {meta['id']} · {meta['name']}\n\n`{line_dir.as_posix()}/`\n\n"
                      + (f"**Question:** {meta['tested']}\n\n" if meta.get("tested") else "")
                      + (f"**Found:** {meta['found']}\n\n" if meta.get("found") else
                         "**Found:** _not yet (add `found` in `results/lines_natural.json`)_\n\n")
                      + table(["Run", "Start", "Steps", "Instances", "LR", "Minutes", "Status", "Eval (logs)",
                               "Notes", "W&B"], rows))
    return ("## Training lines\n\n"
            "Each line is one training history in its own folder (`runs/natural/<line>/`); its pieces are "
            "the run folders inside, split by crashes, memory kills or deliberate stops, in order. "
            "`train.py --resume` adds a piece to the model's line; `--line <folder>` starts a new one (a "
            "branch when resuming). **Steps** with ~ come from the last checkpoint (the run left no "
            "record). **Eval** is the fixed-spot mean logs per episode at that point (* = random spawns "
            "only, older and noisier). Names, purposes and notes: `results/lines_natural.json`.\n\n"
            + "\n\n".join(blocks))


def natural_random_spawn_section(evals: list[dict], arena_evals: list[dict], runs: list[dict]) -> str:
    """Random-spawn evaluations: all natural-world evaluations before fixed spots (2026-11-16)."""
    by_agent, arena = episodes_by_agent(evals), episodes_by_agent(arena_evals)
    if not by_agent:
        return ""
    random_eps = by_agent.get("random", [])
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", kv[0])):  # baseline first
        is_random = agent == "random"
        name = model_label(agent, runs)
        arena_mean = f"{sum(arena[agent]) / len(arena[agent]):.2f}" if agent in arena else "–"
        mean = sum(eps) / len(eps)
        vs = ("baseline" if is_random
              else f"{mean / random_mean:.1f}×" if random_mean else "no baseline yet")
        rows.append([name, line_note(agent), arena_mean, len(eps), f"{mean:.2f}", min(eps), max(eps), vs])
    return (
        "## Earlier random-spawn evaluations\n\n"
        "Historical: every natural-world evaluation before the fixed spots (2026-11-16) spawned in random "
        "forests, so these means also move by ±3-4 logs from spawn luck, and most were made under an "
        "older setup (render distance 6, no damage). Use [Results](results.md) to compare agents. "
        "**Arena mean** is the same agent's score in the walled arena "
        "([EXPERIMENTS_ARENA.md](../EXPERIMENTS_ARENA.md)).\n\n"
        + table(["Model", "Line", "Arena mean", "Episodes", "Mean logs", "Min", "Max", "vs random"],
                rows)
    )


def update(stamp: str | None = None) -> None:
    """Rebuild docs/natural/ from the natural-world CSVs: the index, then one page per table."""
    stamp = stamp or f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"
    evals, runs = read_rows(NATURAL_EVAL_CSV), read_rows(NATURAL_TRAIN_CSV)
    held_out = [r for r in evals if is_held_out(r)]
    training_world = [r for r in evals if not is_held_out(r)]
    wooded = [r for r in training_world if is_wooded(r)]
    fixed = [r for r in training_world if is_fixed(r) and not is_wooded(r)]  # the forest spots
    random_spawns = [r for r in training_world if not is_fixed(r)]
    sections = {
        "results.md": fixed_spots_section(fixed, runs),
        "wooded.md": wooded_section(wooded, fixed, runs),
        "held_out.md": held_out_section(held_out, fixed, runs),
        "lines.md": lines_section(fixed, random_spawns, runs),
        "evaluation_runs.md": evals_section(evals),
        "random_spawns.md": natural_random_spawn_section(random_spawns, read_rows(EVAL_CSV), runs),
        "setup.md": natural_setup_section(),
    }
    NATURAL_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for name, title, about in PAGES:
        text = body(sections[name]) or "_No results yet._"
        write_report(NATURAL_DIR / name, page_template(title), f"{stamp}\n\n{text}")
        index.append(f"- [{title}]({name}): {about}")
    write_report(NATURAL_REPORT, INDEX_TEMPLATE, f"{stamp}\n\n" + "\n".join(index))
