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
MINERL_MODELS_FILE = RESULTS_DIR / "minerl_models.json"


def minerl_model(agent: str) -> dict | None:
    """{label, minerl_score, about} for a model copied from the MineRL project (runs/minerl/<run>/),
    else None. Unlisted MineRL models get their folder name."""
    if not agent.startswith("runs/minerl/"):
        return None
    folder = run_dir_of(agent).name
    try:
        models = json.loads(MINERL_MODELS_FILE.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError):
        models = {}
    return models.get(folder) or {"label": f"MineRL model `{folder}`", "about": "MineRL model"}


def minerl_finetune(agent: str) -> dict | None:
    """{label, about} for a MineRL-mode model trained here (runs/minerl_finetuned/...): follows resumed_from
    back to the MineRL model it started from, and counts the steps trained here. None otherwise."""
    if not agent.startswith("runs/minerl_finetuned/"):
        return None
    from chop.reports.csvio import MINERL_TRAIN_CSV, read_rows  # noqa: PLC0415 (csvio imports nothing from here)
    path = Path(agent)
    if path.parent.name == "checkpoints":
        s = path.stem.removeprefix("ppo_chop_").removesuffix("_steps")
        steps = int(s) if s.isdigit() else None
    else:
        row = next((r for r in read_rows(MINERL_TRAIN_CSV) if normalize_agent(r["model"]) == agent), None)
        steps = int(row["timesteps_done"]) if row and str(row["timesteps_done"]).isdigit() else None
    first_cfg, base, cur = None, None, agent
    for _ in range(20):
        cfg = run_config(run_dir_of(cur)) or {}
        first_cfg = cfg if cfg else first_cfg
        src = normalize_agent(cfg.get("resumed_from") or "")
        if not src:
            break
        if src.startswith("runs/minerl/"):
            base = minerl_model(src)
            break
        cur = src
    env = (first_cfg or {}).get("env", {})
    where = f"{env.get('world', '?')}, {env.get('spawn', 'forest')} spawns, "             f"{'one-hit' if 'diamond' in str(env.get('axe', '')) else 'iron'} axe"
    if base is None:
        # The folder goes under the label (label_with_folder), so the label names the network instead.
        net = {"impala": "Impala", "nature": "NatureCNN"}.get((first_cfg or {}).get("network", "nature"))
        ppo = (first_cfg or {}).get("ppo", {})
        settings = "Line 4 PPO settings" if ppo.get("ent_coef") == 0.01 else "gentle PPO settings"
        trained = steps
        label = (f"Scratch {net} (MineRL mode)" + (f" + CG {short_steps(trained * 4)}" if trained else "")
                 + f", {settings}")
    else:
        trained = steps - base.get("ppo_steps", 0) if steps is not None else None
        # "CG": PPO in CraftGround, counted in game ticks (4 per decision) like our lines' steps.
        label = f"{base['label']} + CG" + (f" {short_steps(trained * 4)}" if trained else "")
    about = (f"**MineRL mode**, trained here ({where})"
             + (f" for {trained:,} decisions = {trained * 4 / 1e6:.2f}M ticks" if trained else "")
             + (f", from {base['label']}" if base else ", from scratch"))
    return {"label": label, "about": about}


def run_dir_of(agent: str) -> Path:
    """The run folder a model path belongs to (model.zip or checkpoints/ppo_chop_N_steps.zip)."""
    parent = Path(agent).parent
    return parent.parent if parent.name == "checkpoints" else parent


def run_action_repeat(agent: str) -> int:
    """Game ticks per agent step the model was trained with (its run's config.json; 1 if not recorded)."""
    try:
        cfg = json.loads((run_dir_of(agent) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 1
    return int(cfg.get("env", {}).get("action_repeat", 1))


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
    mrl = minerl_model(agent)
    if mrl is not None:
        return mrl["label"] + ", zero-shot"
    ft = minerl_finetune(agent)
    if ft is not None:
        return ft["label"]
    line = line_of_agent(agent)
    if line is None:
        return agent_label(agent)
    steps = steps_of(agent, runs)
    return f"Line {line['id']} @ {short_steps(steps)}" if steps else f"Line {line['id']}"


def line_note(agent: str) -> str:
    """The line's name, plus the render distance when it isn't today's."""
    if agent == "random":
        return "random actions"
    ft = minerl_finetune(agent)
    if ft is not None:
        return ft["about"] + " (see [MineRL transfer](../experiments/minerl_transfer.md))"
    mrl = minerl_model(agent)
    if mrl is not None:
        return (f"**MineRL mode** (64×64 view, 4 ticks per decision; see "
                f"[MineRL transfer](../experiments/minerl_transfer.md)): {mrl['about']}")
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
