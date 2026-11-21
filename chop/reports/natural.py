"""The natural world's report sections: fixed-spot results, training lines, random spawns, setup.

chop/reports/site.py decides which page each one goes on."""

from pathlib import Path

from chop.envs import common
from chop.envs import natural as natural_env
from chop.reports.agents import (label_with_folder, line_folders, line_meta, line_note,
                                 load_lines, model_label, pieces, run_config, run_dir_of, run_end, short_steps,
                                 start_of)
from chop.reports.compare import REFERENCE, REFERENCE_NAME, paired, per_spot
from chop.reports.csvio import episodes_by_agent, normalize_agent
from chop.reports.markdown import mean_ci, table


def rows_by_agent(evals: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in evals:
        out.setdefault(normalize_agent(r["agent"]), []).append(r)
    return out


def vs_reference(agent: str, by_rows: dict[str, list[dict]]) -> str:
    if agent == REFERENCE:
        return "reference"
    if REFERENCE not in by_rows:
        return "–"
    return paired(per_spot(by_rows[agent]), per_spot(by_rows[REFERENCE]), short=True)

def natural_setup_section() -> str:
    ne = natural_env
    rows = [
        ["World", f"Normal generated world, fixed seed {ne.SEED} (held-out evaluation world: "
                  f"seed {ne.HELD_OUT_SEED}, never trained in)"],
        ["Evaluation", "100 fixed spots in oak/birch forests (`results/eval_spots_natural.json`, `make_eval_spots.py`), "
                       "and 100 in every biome with trees (`results/eval_spots_natural_wooded.json`); "
                       "`evaluate.py --random-spots` for random spawns. Held-out world (archived: "
                       "[held-out world](../results/archive/held_out.md)): `evaluate.py <model> --env natural "
                       "--held-out`. Every biome with trees: [Natural world: every biome with trees](../results/natural_all_tree_biomes.md)"],
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
    return "## Natural world\n\n" + table(["Setting", "Value"], rows)


def fixed_spots_section(evals: list[dict], runs: list[dict] | None = None) -> str:
    """One row per agent from fixed-spot evaluations (evaluate.py's default in the natural world
    since 2026-11-16): every agent played the same forests, so these means compare fairly."""
    runs = runs or []
    by_agent = episodes_by_agent(evals)
    by_rows = rows_by_agent(evals)
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
                     sum(x == 0 for x in eps), max(eps), vs_reference(agent, by_rows), vs])
    return (
        "## Results\n\n"
        "Every agent here played the same 100 oak/birch forest spots (`results/eval_spots_natural.json`, one "
        "episode each, same starting direction, current setup), so differences come from the agents, "
        "not from where they spawned. Best first. **± is the 95% range** of the mean: two agents whose "
        f"ranges overlap a lot may not really differ. **vs {REFERENCE_NAME}** is the average difference per "
        "spot with its 95% range; a range that doesn't include 0 is a real difference. MineRL rows play in "
        "MineRL mode (their own view and actions) under the same rules. Random-spawn scores are in "
        "[Other evaluations](other_evaluations.md#random-spawns).\n\n"
        + table(["Model", "Line", "Episodes", "Mean logs", "Median", "0-log episodes", "Max",
                 f"vs {REFERENCE_NAME}", "vs random"], rows)
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
    by_rows = rows_by_agent(wooded)
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
                     vs_reference(agent, by_rows), f_cell, gap, sum(x == 0 for x in eps), f_zeros])
    return (
        "## Results\n\n"
        "100 fixed spots in every biome with trees except savanna (`results/eval_spots_natural_wooded.json`, "
        "made with `make_eval_spots.py --spawn wooded`): snowy taiga, taiga, dark forest, jungle, swamp and more, "
        "same training world. **Oak/birch mean** is the same agent's score on the "
        "[oak and birch forests](natural_oak_birch_forests.md) page; the **gap** is how much it loses outside them. **0-log episodes** count the episodes "
        "that chopped nothing on each set (out of 100: the agent got stuck, lost or never found a tree). Run with "
        "`evaluate.py <model> --env natural --spots results/eval_spots_natural_wooded.json`.\n\n"
        + table(["Model", "Line", "Episodes", "Mean logs", f"vs {REFERENCE_NAME}", "Oak/birch mean", "Gap",
                 "0-log episodes (here)", "0-log episodes (oak/birch)"], rows)
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
        # A short heading is its own anchor (#line-4b) in GitHub, Cursor and VS Code alike.
        blocks.append(f"### Line {meta['id']}\n\n**{meta['name']}** · `{line_dir.as_posix()}/`\n\n"
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


def natural_random_spawn_section(forest: list[dict], wooded: list[dict], arena_evals: list[dict],
                                 runs: list[dict]) -> str:
    """Random-spawn evaluations (evaluate.py --random-spots [--spawn wooded]): every episode a new spot
    found by the training spawn search, so no two agents play the same spots. One table per spawn rule."""
    if not forest and not wooded:
        return ""
    arena = episodes_by_agent(arena_evals)

    def rows_for(evals: list[dict], with_baseline: bool) -> list[list]:
        by_agent = episodes_by_agent(evals)
        random_eps = by_agent.get("random", []) if with_baseline else []
        random_mean = sum(random_eps) / len(random_eps) if random_eps else None
        rows = []
        for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", -sum(kv[1]) / len(kv[1]))):
            mean, ci = mean_ci(eps)
            row = [model_label(agent, runs), line_note(agent), len(eps), f"**{mean:.2f}** ± {ci:.2f}",
                   sum(x == 0 for x in eps), max(eps)]
            if with_baseline:
                row += [f"{sum(arena[agent]) / len(arena[agent]):.2f}" if agent in arena else "–",
                        "baseline" if agent == "random"
                        else f"{mean / random_mean:.1f}×" if random_mean else "–"]
            rows.append(row)
        return rows

    head = ["Model", "Line", "Episodes", "Mean logs", "0-log episodes", "Max"]
    parts = [
        "Every episode starts at a new spot found by the same search training uses, so agents play "
        "different spots and each mean also moves by ±3-4 logs from spawn luck: compare agents on the "
        "fixed spots first ([oak and birch forests](natural_oak_birch_forests.md), "
        "[every biome with trees](natural_all_tree_biomes.md)). This checks that a fixed-spawn score holds on "
        "spots no fixed set contains. Run with `evaluate.py <model> --env natural --random-spots "
        "[--spawn wooded]`."]
    if forest:
        parts.append("### Random spawns in oak/birch forests\n\nRows from before 2026-11-16 were "
                     "made under an older setup (render distance 6, no damage). **Arena mean** is the same "
                     "agent's score in the walled arena ([Arena](arena.md)).\n\n"
                     + table(head + ["Arena mean", "vs random"], rows_for(forest, True)))
    if wooded:
        parts.append("### Random spawns in every biome with trees\n\nAny biome with trees but savanna, a new region every "
                     f"{natural_env.MAX_HOP_EPISODES['wooded'] + 1} episodes (as in wooded training).\n\n"
                     + table(head, rows_for(wooded, False)))
    return "## Random spawns\n\n" + "\n\n".join(parts)
