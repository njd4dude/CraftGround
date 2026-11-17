"""Which model is which: labels, training steps and training lines for the reports.

A natural-world line is one training history, and has one folder: runs/natural/<line folder>/<time>/
per piece (crashes, memory kills and deliberate stops split a history into several pieces;
train.py --resume adds the next piece to the same folder). Display ids, names, purposes and notes
on pieces come from results/lines_natural.json; a line folder it doesn't list still shows up."""

import json
import re
from pathlib import Path

from chop.envs import common
from chop.envs import natural as natural_env
from chop.paths import NATURAL_RUNS, RESULTS_DIR
from chop.reports.csvio import normalize_agent


LINES_FILE = RESULTS_DIR / "lines_natural.json"



def run_dir_of(agent: str) -> Path:
    """The run folder a model path belongs to (model.zip or checkpoints/ppo_chop_N_steps.zip)."""
    parent = Path(agent).parent
    return parent.parent if parent.name == "checkpoints" else parent

def agent_label(agent: str) -> str:
    """Run folder name, plus the step count for a checkpoint."""
    path = Path(agent)
    if path.parent.name == "checkpoints":
        steps = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
        return f"`{run_dir_of(agent).name}` @ {int(steps):,} steps" if steps.isdigit() else f"`{agent}`"
    return f"`{path.parent.name}`"

def steps_of(agent: str, runs: list[dict]) -> int | None:
    """Training steps behind a model: from a checkpoint's name, else the training CSV."""
    path = Path(agent)
    if path.parent.name == "checkpoints":
        steps = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
    else:
        steps = next((r["timesteps_done"] for r in runs if normalize_agent(r["model"]) == agent), "")
    return int(steps) if str(steps).isdigit() else None

def short_steps(n: int) -> str:
    return f"{n / 1e6:.1f}M" if n >= 995_000 else f"{round(n / 1000)}k"

def describe(agent: str, runs: list[dict]) -> str:
    """What an agent is, from the config.json of its run and the runs it was resumed from, e.g.
    "jump · constant LR to 601k, then linear LR · 1.0M steps". Runs resumed after a crash
    count as one history, so only changes of learning-rate schedule show up as stages."""
    if agent == "random":
        return "random actions"
    stages: list[tuple[str, int | None]] = []  # (LR schedule, steps at the end of that piece), newest first
    cfg0, cur = None, agent
    for _ in range(20):  # follow resumed_from back to the first run
        try:
            cfg = json.loads((run_dir_of(cur) / "config.json").read_text(encoding="utf-8"))
        except Exception:
            break
        if cfg.get("env", {}).get("world") != "natural":
            stages.append(("arena", None))
            break
        cfg0 = cfg0 or cfg
        stages.append((f"{cfg.get('lr_schedule', 'constant')} LR", steps_of(cur, runs)))
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
        s if k == len(merged) - 1 or e is None else f"{s} to {short_steps(e)}" for k, (s, e) in enumerate(merged))
    if merged and merged[0][0] == "arena":
        history = history.replace("arena, then ", "arena model, then ", 1)
    parts = ["jump" if len(env.get("actions", [])) == len(common.ACTION_NAMES_JUMP) else "no jump", history]
    total = steps_of(agent, runs)
    if total:
        parts.append(f"{short_steps(total)} steps")
    rd = env.get("render_distance", 6)  # runs before render_distance was recorded used 6
    if rd != natural_env.RENDER_DISTANCE:
        parts.append(f"render distance {rd}")
    return " · ".join(parts)

def load_lines() -> dict:
    try:
        return json.loads(LINES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"lines": [], "piece_notes": {}}

def run_config(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def line_meta(folder: str) -> dict:
    """id / name / tested for a line folder, from results/lines_natural.json or the folder name."""
    for line in load_lines().get("lines", []):
        if line.get("folder") == folder:
            return line
    m = re.match(r"line([0-9]+[a-z]*)_?(.*)", folder)
    return {"folder": folder, "id": m.group(1) if m else folder,
            "name": (m.group(2).replace("-", " ") if m and m.group(2) else "unnamed line")
            + " (describe it in results/lines_natural.json)", "tested": ""}

def line_folders() -> list[Path]:
    """Line folders, in the order results/lines_natural.json lists them, then the rest by name."""
    dirs = [d for d in NATURAL_RUNS.glob("*") if d.is_dir()] if NATURAL_RUNS.exists() else []
    order = [l.get("folder") for l in load_lines().get("lines", [])]
    return sorted(dirs, key=lambda d: (order.index(d.name) if d.name in order else len(order), d.name))

def pieces(line_dir: Path) -> list[Path]:
    return sorted(p for p in line_dir.glob("*") if (p / "config.json").exists())

def line_of_agent(agent: str) -> dict | None:
    run = run_dir_of(agent)
    if run.parent.parent.as_posix() != NATURAL_RUNS.as_posix():
        return None
    return line_meta(run.parent.name)

def model_label(agent: str, runs: list[dict]) -> str:
    """"Line 2b @ 1.0M" for a natural-world model, else the folder label."""
    if agent == "random":
        return "Random actions (baseline)"
    line = line_of_agent(agent)
    if line is None:
        return agent_label(agent)
    steps = steps_of(agent, runs)
    return f"Line {line['id']} @ {short_steps(steps)}" if steps else f"Line {line['id']}"

def line_note(agent: str) -> str:
    """The line's name, plus the render distance when it isn't today's."""
    if agent == "random":
        return "random actions"
    line = line_of_agent(agent)
    if line is None:
        return describe(agent, [])
    rd = (run_config(run_dir_of(agent)) or {}).get("env", {}).get("render_distance", 6)
    return line["name"] + (f" · render distance {rd}" if rd != natural_env.RENDER_DISTANCE else "")

def label_with_folder(agent: str, runs: list[dict]) -> str:
    """The model's label and, underneath, its run folder (+ steps for a checkpoint), so it's easy to find."""
    return model_label(agent, runs) + ("" if agent == "random" else f"<br>{agent_label(agent)}")

def run_end(run_dir: Path, runs: list[dict]) -> tuple[int | None, bool]:
    """Steps at the end of a piece: from the training CSV (exact), else its last checkpoint (approximate)."""
    model = (run_dir / "model.zip").as_posix()
    row = next((r for r in runs if normalize_agent(r["model"]) == model), None)
    if row and str(row["timesteps_done"]).isdigit():
        return int(row["timesteps_done"]), True
    ckpts = [int(s) for p in (run_dir / "checkpoints").glob("ppo_chop_*_steps.zip")
             if (s := p.stem.removeprefix("ppo_chop_").removesuffix("_steps")).isdigit()]
    return (max(ckpts), False) if ckpts else (None, False)

def start_of(run_dir: Path, runs: list[dict]) -> tuple[int, str]:
    """(steps at the start, where it came from) for one piece."""
    src = normalize_agent((run_config(run_dir) or {}).get("resumed_from") or "")
    if not src:
        return 0, "from scratch"
    path = Path(src)
    if path.parent.name == "checkpoints":
        s = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
        start = int(s) if s.isdigit() else 0
    else:
        start = run_end(path.parent, runs)[0] or 0
    parent_line = line_of_agent(src)
    if parent_line is None:
        return start, "from an arena model"
    if run_dir_of(src).parent.name == run_dir.parent.name:
        return start, "continued"
    return start, f"branch from Line {parent_line['id']}"
