"""Keeps EXPERIMENTS.md (arena) and EXPERIMENTS_NATURAL.md (natural world) up to date.

evaluate.py and train.py append rows to the CSV files in results/, then call
update_report(env), which rewrites the tables in that environment's report.
Only the part between the AUTO-GENERATED markers is replaced, so notes you
write anywhere else in the file are kept.

Run it by hand to rebuild both reports:  python experiments.py
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import natural_env
import tree_env

RESULTS_DIR = Path("results")
EVAL_CSV = RESULTS_DIR / "eval_results.csv"
TRAIN_CSV = RESULTS_DIR / "training_runs.csv"
REPORT = Path("EXPERIMENTS.md")
NATURAL_EVAL_CSV = RESULTS_DIR / "eval_results_natural.csv"
NATURAL_TRAIN_CSV = RESULTS_DIR / "training_runs_natural.csv"
NATURAL_REPORT = Path("EXPERIMENTS_NATURAL.md")

START = "<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->"
END = "<!-- AUTO-GENERATED:END -->"

TEMPLATE = f"""# CraftGround Experiments: Chop Trees

Goal: train an agent to chop as many oak logs as possible in one episode.

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""

NATURAL_TEMPLATE = f"""# CraftGround Experiments: Chop Trees in a Natural World

Goal: chop as many logs as possible in one episode, starting in a real generated
forest instead of the walled arena (natural_env.py). Arena results are in EXPERIMENTS.md.

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""


def append_row(path: Path, fields: list[str], row: dict) -> None:
    """Append a row. If `fields` has columns the file doesn't, the file is rewritten
    with the extra columns (left blank for older rows) so nothing gets misaligned."""
    path.parent.mkdir(exist_ok=True)
    old_rows = read_rows(path)
    old_fields = list(old_rows[0].keys()) if old_rows else []
    if path.exists() and old_rows and not set(fields) <= set(old_fields):
        merged = old_fields + [f for f in fields if f not in old_fields]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=merged, restval="")
            writer.writeheader()
            writer.writerows(old_rows + [row])
        return
    new_file = not path.exists() or not old_fields
    with path.open("a" if not new_file else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=old_fields or fields, restval="")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def setup_section() -> str:
    rows = [
        ["World", "Superflat arena, stone-brick walls at ±%d blocks" % tree_env.ARENA_RADIUS],
        ["Trees", f"{len(tree_env.TREE_POSITIONS)} oaks × {tree_env.TRUNK_HEIGHT} logs = {tree_env.MAX_LOGS} logs max"],
        ["Rebuilt each episode", "yes (trees regrow on reset)"],
        ["Tool", "Unbreakable diamond axe, Efficiency X (1 hit per log)"],
        ["Spawn", "Arena centre, random facing (15° steps)"],
        ["Observation", f"RGB image {tree_env.IMAGE_WIDTH}×{tree_env.IMAGE_HEIGHT}"],
        ["Actions", f"{len(tree_env.ACTION_NAMES)} discrete: " + ", ".join(tree_env.ACTION_NAMES)],
        ["Reward", tree_env.REWARD],
        ["Episode length", f"{tree_env.EPISODE_STEPS} steps (or all logs chopped)"],
    ]
    return "## Environment setup\n\n" + table(["Setting", "Value"], rows)


def normalize_agent(agent: str) -> str:
    """'runs\\x\\model.zip', './runs/x/model.zip' and 'runs/x/model.zip' are the same agent."""
    return agent if agent == "random" else Path(agent).as_posix().removeprefix("./")


def summary_section(evals: list[dict], runs: list[dict]) -> str:
    by_agent: dict[str, list[int]] = {}
    for r in evals:
        by_agent.setdefault(normalize_agent(r["agent"]), []).extend(int(x) for x in r["per_episode"].split())

    random_eps = by_agent.get("random", [])
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None

    def eval_cells(eps: list[int], is_random: bool) -> list:
        mean = sum(eps) / len(eps)
        vs = "baseline" if is_random else (f"**{mean / random_mean:.1f}×**" if random_mean else "no baseline yet")
        return [len(eps), f"**{mean:.2f}**", min(eps), max(eps), vs]

    rows = []
    if random_eps:
        rows.append(["Random actions (baseline)", "–", *eval_cells(random_eps, True)])
    else:
        rows.append(["Random actions (baseline)", "–", "_not evaluated: `run.bat evaluate.py --random --episodes 20`_",
                     "", "", "", ""])

    listed = {"random"}
    for run in reversed(runs):  # newest first
        model = normalize_agent(run["model"])
        listed.add(model)
        name = Path(model).parent.name
        eps = by_agent.get(model)
        if eps:
            rows.append([f"`{name}`", run["last20_avg_logs"], *eval_cells(eps, False)])
        else:
            rows.append([f"`{name}`", run["last20_avg_logs"],
                         f"_not evaluated: `run.bat evaluate.py {model} --episodes 20`_", "", "", "", ""])

    # Evaluated models that aren't from a recorded training run (e.g. a checkpoint).
    for agent, eps in by_agent.items():
        if agent not in listed:
            rows.append([f"`{agent}`", "–", *eval_cells(eps, False)])

    return (
        "## Summary\n\n"
        "One row per agent. **Mean logs** is the real score: the average over all `evaluate.py` "
        f"episodes for that agent (max {tree_env.MAX_LOGS}). **Training avg** is the last-20-episode "
        "average seen during training. It's a useful hint, but the agent was still exploring then, "
        "so evaluate before trusting it. **vs random** = how many times more logs than random actions.\n\n"
        + table(["Agent", "Training avg", "Eval episodes", "Mean logs", "Min", "Max", "vs random"], rows)
    )


def load_config(model: str) -> dict | None:
    """The config.json train.py saves next to each run's model.zip."""
    path = Path(model).parent / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def agents_section(evals: list[dict], runs: list[dict]) -> str:
    """One subsection per trained agent: how long it trained, its settings, and what was
    different about it (hyperparameters changed from the defaults, environment/reward changes
    since the previous run, and the run's notes)."""
    if not runs:
        return "## Agents\n\n_None yet._"
    by_agent: dict[str, list[int]] = {}
    for r in evals:
        by_agent.setdefault(normalize_agent(r["agent"]), []).extend(int(x) for x in r["per_episode"].split())

    blocks = []
    prev_env = None
    for run in runs:  # oldest first, so environment changes are compared with the run before
        model = normalize_agent(run["model"])
        cfg = load_config(model)
        lines = [f"### `{Path(model).parent.name}`", ""]
        target = f" of {int(cfg['timesteps_target']):,} planned" if cfg else ""
        n_envs = cfg.get("n_envs", 1) if cfg else 1
        envs = f", {n_envs} parallel envs" if n_envs > 1 else ""
        lines.append(f"- **Trained:** {int(run['timesteps_done']):,} steps{target}, {run['episodes']} episodes, "
                     f"{run['minutes']} min{envs} ({run['status']})")
        if run.get("resumed_from"):
            lines.append(f"- **Resumed from:** `{run['resumed_from']}`")
        eps = by_agent.get(model)
        score = (f"{sum(eps) / len(eps):.2f} logs/episode over {len(eps)} eval episodes" if eps
                 else "not evaluated yet")
        lines.append(f"- **Result:** {score} (training last-20 avg {run['last20_avg_logs']})")

        if cfg is None:
            lines.append("- _No config.json for this run, so its settings weren't recorded._")
        else:
            changed = cfg.get("changed_from_defaults", {})
            ppo = ", ".join(f"**{k}={v}**" if k in changed else f"{k}={v}" for k, v in cfg["ppo"].items())
            lines.append(f"- **PPO settings:** {ppo}")
            special = [f"{k}: {c['default']} → **{c['used']}**" for k, c in changed.items()]
            env = cfg.get("env", {})
            if prev_env is not None:
                special += [f"env {k}: {prev_env.get(k)} → **{v}** (changed since previous run)"
                            # A setting older runs didn't record yet is not a change.
                            for k, v in env.items() if k in prev_env and prev_env[k] != v]
            same = ("first recorded run, default settings (the baseline)" if prev_env is None
                    else "nothing, default settings and same environment as the previous run")
            prev_env = env
            lines.append("- **What's special:** " + ("; ".join(special) if special else same))
            if cfg.get("notes"):
                lines.append(f"- **Notes:** {cfg['notes']}")
        blocks.append("\n".join(lines))

    return ("## Agents\n\nOne section per trained agent, newest first. Hyperparameters changed from the "
            "defaults in `train.py` are in bold. Environment changes (reward, episode length, trees...) are "
            "flagged on the first run that used them. Add notes with `train.py --notes \"...\"`, or edit "
            "`notes` in the run's `config.json` and rerun `python experiments.py`.\n\n"
            + "\n\n".join(reversed(blocks)))


def evals_section(evals: list[dict]) -> str:
    if not evals:
        return "## Evaluation runs\n\n_None yet._"
    with_spots = any(r.get("spots") for r in evals)  # natural world: fixed or random spawns
    with_world = any(_is_held_out(r) for r in evals)  # natural world: training or held-out world
    rows = [[r["timestamp"], f"`{r['agent']}`",
             *([("fixed" if _is_fixed(r) else "random")] if with_spots else []),
             *([("held-out" if _is_held_out(r) else "training")] if with_world else []),
             r["episodes"], r["mean_logs"], r["min_logs"],
             r["max_logs"], r.get("n_envs") or "–", r.get("minutes") or "–", r.get("steps_per_s") or "–",
             r["per_episode"]] for r in reversed(evals)]
    return ("## Evaluation runs\n\nOne row per `evaluate.py` run, newest first. **Minutes** includes "
            "launching Minecraft; **Steps/s** is env steps per second over all instances while playing "
            "(runs from before timing was added show –)."
            + (" **Spots**: fixed = the saved evaluation spots, random = random spawns (all runs before "
               "2026-11-16)." if with_spots else "")
            + (" **World**: training = the world agents train in, held-out = one they never see "
               "(`--held-out`)." if with_world else "") + "\n\n"
            + table(["When", "Agent", *(["Spots"] if with_spots else []), *(["World"] if with_world else []),
                     "Episodes", "Mean logs", "Min", "Max",
                     "Instances", "Minutes", "Steps/s", "Per episode"], rows))


def training_section(runs: list[dict], start_cmd: str = "run.bat train.py") -> str:
    if not runs:
        return f"## Training runs\n\n_None yet. Start one with `{start_cmd}`._"
    def wandb_link(r: dict) -> str:
        url = r.get("wandb_url") or ""
        return f"[open]({url})" if url else "–"

    rows = [[r["started"], r["timesteps_done"], r["episodes"], r["minutes"], r["last20_avg_logs"],
             r["best_episode_logs"], r["status"], f"`{r['model']}`", wandb_link(r)] for r in reversed(runs)]
    return ("## Training runs\n\nOne row per `train.py` run, newest first.\n\n"
            + table(["Started", "Steps", "Episodes", "Minutes", "Last-20 avg logs", "Best episode",
                     "Status", "Saved model", "W&B"], rows))


def natural_setup_section() -> str:
    ne = natural_env
    rows = [
        ["World", f"Normal generated world, fixed seed {ne.SEED} (held-out evaluation world: "
                  f"seed {ne.HELD_OUT_SEED}, never trained in)"],
        ["Evaluation", "100 fixed forest spots (`results/eval_spots_natural.json`, `make_eval_spots.py`); "
                       "`evaluate.py --random-spots` for random spawns. Held-out world: "
                       "`make_eval_spots.py --held-out` once, then `evaluate.py <model> --env natural "
                       "--held-out` (its table appears once there are results)"],
        ["Spawn", f"A new spot every episode, on the ground in an oak/birch forest: first within "
                  f"{ne.HOP_RANGE} blocks of the last spot, else anywhere within ±{ne.FAR_RANGE:,}"],
        ["Rebuilt each episode", "no: chopped trees stay chopped, so every episode moves to fresh forest"],
        ["Boundaries", "none; terrain loads as the agent walks"],
        ["Render distance", f"{ne.RENDER_DISTANCE} chunks, Minecraft heap {ne.JVM_MAX_MEMORY} "
                            "(since 2026-11-15, to fit 4 instances in memory; earlier runs: 6 chunks, 3G)"],
        ["Tool", "Unbreakable diamond axe, Efficiency X (1 hit per log)"],
        ["Observation", f"RGB image {tree_env.IMAGE_WIDTH}×{tree_env.IMAGE_HEIGHT} (same as the arena)"],
        ["Actions", f"{len(tree_env.ACTION_NAMES_JUMP)} discrete: the arena's {len(tree_env.ACTION_NAMES)} plus "
                    + ", ".join(tree_env.JUMP_ACTIONS) + " (since 2026-11-15; earlier runs and the random "
                    "baseline use the arena's 11)"],
        ["Reward", ne.REWARD],
        ["Damage", "Lava, fire, drowning etc. hurt and dying ends the episode; no fall damage "
                   "(since 2026-11-16; earlier runs: player immune to all damage)"],
        ["Episode length", f"{tree_env.EPISODE_STEPS} steps (no log cap), or until the agent dies"],
    ]
    return "## Environment setup\n\n" + table(["Setting", "Value"], rows)


def _episodes_by_agent(rows: list[dict]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for r in rows:
        out.setdefault(normalize_agent(r["agent"]), []).extend(int(x) for x in r["per_episode"].split())
    return out


def _run_dir(agent: str) -> Path:
    """The run folder a model path belongs to (model.zip or checkpoints/ppo_chop_N_steps.zip)."""
    parent = Path(agent).parent
    return parent.parent if parent.name == "checkpoints" else parent


def _agent_label(agent: str) -> str:
    """Run folder name, plus the step count for a checkpoint."""
    path = Path(agent)
    if path.parent.name == "checkpoints":
        steps = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
        return f"`{_run_dir(agent).name}` @ {int(steps):,} steps" if steps.isdigit() else f"`{agent}`"
    return f"`{path.parent.name}`"


def _trained_in(agent: str) -> str:
    """'natural' or 'arena', from the run's config.json (folder name as a fallback)."""
    try:
        return json.loads((_run_dir(agent) / "config.json").read_text(encoding="utf-8"))["env"]["world"]
    except Exception:
        return "natural" if "_natural_" in _run_dir(agent).name else "arena"


def _steps_of(agent: str, runs: list[dict]) -> int | None:
    """Training steps behind a model: from a checkpoint's name, else the training CSV."""
    path = Path(agent)
    if path.parent.name == "checkpoints":
        steps = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
    else:
        steps = next((r["timesteps_done"] for r in runs if normalize_agent(r["model"]) == agent), "")
    return int(steps) if str(steps).isdigit() else None


def _short_steps(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 995_000 else f"{round(n / 1000)}k"


def _describe(agent: str, runs: list[dict]) -> str:
    """What an agent is, from the config.json of its run and the runs it was resumed from, e.g.
    "jump · constant LR to 601k, then linear LR · 1.0M steps". Runs resumed after a crash
    count as one history, so only changes of learning-rate schedule show up as stages."""
    if agent == "random":
        return "random actions"
    stages: list[tuple[str, int | None]] = []  # (LR schedule, steps at the end of that piece), newest first
    cfg0, cur = None, agent
    for _ in range(20):  # follow resumed_from back to the first run
        try:
            cfg = json.loads((_run_dir(cur) / "config.json").read_text(encoding="utf-8"))
        except Exception:
            break
        if cfg.get("env", {}).get("world") != "natural":
            stages.append(("arena", None))
            break
        cfg0 = cfg0 or cfg
        stages.append((f"{cfg.get('lr_schedule', 'constant')} LR", _steps_of(cur, runs)))
        cur = normalize_agent(cfg.get("resumed_from") or "")
        if not cur:
            break
    if cfg0 is None:
        return "trained in the arena" if stages else ""
    env = cfg0.get("env", {})
    stages.reverse()  # oldest first
    merged: list[list] = []
    for schedule, end in stages:
        if merged and merged[-1][0] == schedule:
            merged[-1][1] = end  # same schedule continued after a resume
        else:
            merged.append([schedule, end])
    history = ", then ".join(
        s if k == len(merged) - 1 or e is None else f"{s} to {_short_steps(e)}" for k, (s, e) in enumerate(merged))
    if merged and merged[0][0] == "arena":
        history = history.replace("arena, then ", "arena model, then ", 1)
    parts = ["jump" if len(env.get("actions", [])) == len(tree_env.ACTION_NAMES_JUMP) else "no jump", history]
    total = _steps_of(agent, runs)
    if total:
        parts.append(f"{_short_steps(total)} steps")
    rd = env.get("render_distance", 6)  # runs before render_distance was recorded used 6
    if rd != natural_env.RENDER_DISTANCE:
        parts.append(f"render distance {rd}")
    return " · ".join(parts)


# --- Training lines (natural world) -----------------------------------------------
# A line is one training history, and has one folder: runs/natural/<line folder>/<time>/ per
# piece (crashes, memory kills and deliberate stops split a history into several pieces; train.py
# --resume adds the next piece to the same folder). Display ids, names, purposes and notes on
# pieces come from results/lines_natural.json; a line folder it doesn't list still shows up.

LINES_FILE = RESULTS_DIR / "lines_natural.json"
RUNS_DIR = Path("runs")
NATURAL_RUNS = RUNS_DIR / "natural"


def _load_lines() -> dict:
    try:
        return json.loads(LINES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"lines": [], "piece_notes": {}}


def _run_config(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _line_meta(folder: str) -> dict:
    """id / name / tested for a line folder, from results/lines_natural.json or the folder name."""
    for line in _load_lines().get("lines", []):
        if line.get("folder") == folder:
            return line
    m = re.match(r"line([0-9]+[a-z]*)_?(.*)", folder)
    return {"folder": folder, "id": m.group(1) if m else folder,
            "name": (m.group(2).replace("-", " ") if m and m.group(2) else "unnamed line")
            + " (describe it in results/lines_natural.json)", "tested": ""}


def _line_dirs() -> list[Path]:
    """Line folders, in the order results/lines_natural.json lists them, then the rest by name."""
    dirs = [d for d in NATURAL_RUNS.glob("*") if d.is_dir()] if NATURAL_RUNS.exists() else []
    order = [l.get("folder") for l in _load_lines().get("lines", [])]
    return sorted(dirs, key=lambda d: (order.index(d.name) if d.name in order else len(order), d.name))


def _pieces(line_dir: Path) -> list[Path]:
    return sorted(p for p in line_dir.glob("*") if (p / "config.json").exists())


def _line_of_agent(agent: str) -> dict | None:
    run = _run_dir(agent)
    if run.parent.parent.as_posix() != NATURAL_RUNS.as_posix():
        return None
    return _line_meta(run.parent.name)


def _model_label(agent: str, runs: list[dict]) -> str:
    """"Line 2b @ 1.0M" for a natural-world model, else the folder label."""
    if agent == "random":
        return "Random actions (baseline)"
    line = _line_of_agent(agent)
    if line is None:
        return _agent_label(agent)
    steps = _steps_of(agent, runs)
    return f"Line {line['id']} @ {_short_steps(steps)}" if steps else f"Line {line['id']}"


def _line_note(agent: str) -> str:
    """The line's name, plus the render distance when it isn't today's."""
    if agent == "random":
        return "random actions"
    line = _line_of_agent(agent)
    if line is None:
        return _describe(agent, [])
    rd = (_run_config(_run_dir(agent)) or {}).get("env", {}).get("render_distance", 6)
    return line["name"] + (f" · render distance {rd}" if rd != natural_env.RENDER_DISTANCE else "")


def _run_end(run_dir: Path, runs: list[dict]) -> tuple[int | None, bool]:
    """Steps at the end of a piece: from the training CSV (exact), else its last checkpoint (approximate)."""
    model = (run_dir / "model.zip").as_posix()
    row = next((r for r in runs if normalize_agent(r["model"]) == model), None)
    if row and str(row["timesteps_done"]).isdigit():
        return int(row["timesteps_done"]), True
    ckpts = [int(s) for p in (run_dir / "checkpoints").glob("ppo_chop_*_steps.zip")
             if (s := p.stem.removeprefix("ppo_chop_").removesuffix("_steps")).isdigit()]
    return (max(ckpts), False) if ckpts else (None, False)


def _start_of(run_dir: Path, runs: list[dict]) -> tuple[int, str]:
    """(steps at the start, where it came from) for one piece."""
    src = normalize_agent((_run_config(run_dir) or {}).get("resumed_from") or "")
    if not src:
        return 0, "from scratch"
    path = Path(src)
    if path.parent.name == "checkpoints":
        s = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
        start = int(s) if s.isdigit() else 0
    else:
        start = _run_end(path.parent, runs)[0] or 0
    parent_line = _line_of_agent(src)
    if parent_line is None:
        return start, "from an arena model"
    if _run_dir(src).parent.name == run_dir.parent.name:
        return start, "continued"
    return start, f"branch from Line {parent_line['id']}"


def lines_section(fixed: list[dict], random_spawns: list[dict], runs: list[dict]) -> str:
    """One block per training line: what it tested, then its pieces in order."""
    line_dirs = [d for d in _line_dirs() if _pieces(d)]
    if not line_dirs:
        return ""
    notes = _load_lines().get("piece_notes", {})
    fixed_by, random_by = _episodes_by_agent(fixed), _episodes_by_agent(random_spawns)
    blocks = []
    for line_dir in line_dirs:
        meta = _line_meta(line_dir.name)
        rows = []
        for run_dir in _pieces(line_dir):
            cfg = _run_config(run_dir) or {}
            start, origin = _start_of(run_dir, runs)
            end, exact = _run_end(run_dir, runs)
            span = (f"{_short_steps(start) if start else '0'} → "
                    f"{'' if exact else '~'}{_short_steps(end) if end else '?'}")
            row = next((r for r in runs if normalize_agent(r["model"]) == (run_dir / "model.zip").as_posix()), None)
            scores = []
            for agent in sorted(set(fixed_by) | set(random_by)):
                if _run_dir(agent).as_posix() != run_dir.as_posix():
                    continue
                at = _model_label(agent, runs).split(" @ ")[-1]
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
        blocks.append(f"### Line {meta['id']} · {meta['name']}\n\n`{line_dir.as_posix()}/`\n\n"
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
    by_agent, arena = _episodes_by_agent(evals), _episodes_by_agent(arena_evals)
    if not by_agent:
        return ""
    random_eps = by_agent.get("random", [])
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", kv[0])):  # baseline first
        is_random = agent == "random"
        name = _model_label(agent, runs)
        arena_mean = f"{sum(arena[agent]) / len(arena[agent]):.2f}" if agent in arena else "–"
        mean = sum(eps) / len(eps)
        vs = ("baseline" if is_random
              else f"{mean / random_mean:.1f}×" if random_mean else "no baseline yet")
        rows.append([name, _line_note(agent), arena_mean, len(eps), f"{mean:.2f}", min(eps), max(eps), vs])
    return (
        "## Earlier random-spawn evaluations\n\n"
        "Historical: every natural-world evaluation before the fixed spots (2026-11-16) spawned in random "
        "forests, so these means also move by ±3-4 logs from spawn luck, and most were made under an "
        "older setup (render distance 6, no damage). Use the Results table above to compare agents. "
        "**Arena mean** is the same agent's score in the walled arena (EXPERIMENTS.md).\n\n"
        + table(["Model", "Line", "Arena mean", "Episodes", "Mean logs", "Min", "Max", "vs random"],
                rows)
    )


def _is_fixed(row: dict) -> bool:
    return (row.get("spots") or "").startswith("fixed")


def _is_held_out(row: dict) -> bool:
    """Played in a world other than the training one (blank = older rows, all in the training world)."""
    return (row.get("world_seed") or natural_env.SEED) != natural_env.SEED


def _mean_ci(eps: list[int]) -> tuple[float, float]:
    """Mean and the half-width of its 95% range."""
    n = len(eps)
    mean = sum(eps) / n
    sd = (sum((x - mean) ** 2 for x in eps) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, 1.96 * sd / n ** 0.5


def _label_with_folder(agent: str, runs: list[dict]) -> str:
    """The model's label and, underneath, its run folder (+ steps for a checkpoint), so it's easy to find."""
    return _model_label(agent, runs) + ("" if agent == "random" else f"<br>{_agent_label(agent)}")


def fixed_spots_section(evals: list[dict], runs: list[dict] | None = None) -> str:
    """One row per agent from fixed-spot evaluations (evaluate.py's default in the natural world
    since 2026-11-16): every agent played the same forests, so these means compare fairly."""
    runs = runs or []
    by_agent = _episodes_by_agent(evals)
    if not by_agent:
        return ("## Results\n\n_None yet: `run.bat evaluate.py <model.zip> --env natural` "
                "plays the 100 fixed spots in `results/eval_spots_natural.json`._")
    random_eps = by_agent.get("random")
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", -sum(kv[1]) / len(kv[1]))):
        n = len(eps)
        mean, ci = _mean_ci(eps)
        median = sorted(eps)[n // 2]
        vs = ("baseline" if agent == "random"
              else f"**{mean / random_mean:.1f}×**" if random_mean else "no baseline yet")
        rows.append([_label_with_folder(agent, runs), _line_note(agent), n, f"**{mean:.2f}** ± {ci:.2f}", median,
                     sum(x == 0 for x in eps), max(eps), vs])
    return (
        "## Results\n\n"
        "Every agent here played the same 100 forest spots (`results/eval_spots_natural.json`, one "
        "episode each, same starting direction, current setup), so differences come from the agents, "
        "not from where they spawned. Best first. **± is the 95% range** of the mean: two agents whose "
        "ranges overlap a lot may not really differ. Older random-spawn scores are near the bottom.\n\n"
        + table(["Model", "Line", "Episodes", "Mean logs", "Median", "0-log episodes", "Max",
                 "vs random"], rows)
    )


def held_out_section(held_out: list[dict], training_fixed: list[dict], runs: list[dict] | None = None) -> str:
    """Fixed-spot scores in the held-out world next to the same agent's fixed-spot score in the
    training world: the gap is how much of the score depends on having trained in that world."""
    runs = runs or []
    by_agent = _episodes_by_agent([r for r in held_out if _is_fixed(r)])
    if not by_agent:
        return ""  # shown once there are results; how to get some is in Environment setup
    train = _episodes_by_agent(training_fixed)
    random_eps = by_agent.get("random")
    random_mean = sum(random_eps) / len(random_eps) if random_eps else None
    rows = []
    for agent, eps in sorted(by_agent.items(), key=lambda kv: (kv[0] != "random", -sum(kv[1]) / len(kv[1]))):
        mean, ci = _mean_ci(eps)
        if agent in train:
            t_mean, t_ci = _mean_ci(train[agent])
            t_cell, gap = f"{t_mean:.2f} ± {t_ci:.2f}", f"{mean - t_mean:+.2f}"
        else:
            t_cell, gap = "–", "–"
        vs = ("baseline" if agent == "random"
              else f"**{mean / random_mean:.1f}×**" if random_mean else "no baseline yet")
        rows.append([_label_with_folder(agent, runs), _line_note(agent), len(eps), f"**{mean:.2f}** ± {ci:.2f}",
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


def _write_report(path: Path, template: str, generated: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else template
    if START not in text or END not in text:
        text = template + "\n" + text  # markers removed by hand: put a fresh block on top
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    path.write_text(f"{before}{START}\n{generated}\n{END}{after}", encoding="utf-8")
    print(f"Updated {path}")


def update_report(env: str = "arena") -> None:
    """Rebuild the tables of one environment's report: "arena" or "natural"."""
    stamp = f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"
    if env == "natural":
        evals, runs = read_rows(NATURAL_EVAL_CSV), read_rows(NATURAL_TRAIN_CSV)
        held_out = [r for r in evals if _is_held_out(r)]
        training_world = [r for r in evals if not _is_held_out(r)]
        fixed = [r for r in training_world if _is_fixed(r)]
        random_spawns = [r for r in training_world if not _is_fixed(r)]
        sections = [
            stamp,
            fixed_spots_section(fixed, runs),
            held_out_section(held_out, fixed, runs),
            lines_section(fixed, random_spawns, runs),  # replaces the per-run Agents and Training runs tables
            evals_section(evals),
            natural_random_spawn_section(random_spawns, read_rows(EVAL_CSV), runs),
            natural_setup_section(),
        ]
        generated = "\n\n".join(s for s in sections if s)  # empty sections (no data yet) are left out
        _write_report(NATURAL_REPORT, NATURAL_TEMPLATE, generated)
        return
    evals, runs = read_rows(EVAL_CSV), read_rows(TRAIN_CSV)
    generated = "\n\n".join([
        stamp,
        summary_section(evals, runs),
        agents_section(evals, runs),
        training_section(runs),
        evals_section(evals),
        setup_section(),
    ])
    _write_report(REPORT, TEMPLATE, generated)


if __name__ == "__main__":
    update_report("arena")
    update_report("natural")
