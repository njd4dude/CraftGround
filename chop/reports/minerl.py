"""docs/minerl/: MineRL-trained models run in CraftGround (MineRL mode), and MineRL-style scoring of ours.

Built from results/eval_results_minerl.csv. Its rows differ in more than the model: world, spots,
episode length, log cap and axe all vary, so results are grouped by all five ("evaluation type")
and only rows of the same type share a table. Line 5's standard evaluation (natural CSV: fixed
forest spots, 1000 ticks, no cap, one-hit axe) is added to its type as the reference.
Model names and MineRL scores come from results/minerl_models.json."""

import json
from datetime import datetime
from pathlib import Path

from chop.envs import minerl as minerl_env
from chop.envs import natural as natural_env
from chop.paths import DOCS_DIR, RESULTS_DIR
from chop.reports.agents import model_label
from chop.reports.csvio import (MINERL_EVAL_CSV, MINERL_TRAIN_CSV, NATURAL_EVAL_CSV, NATURAL_TRAIN_CSV,
                                is_fixed, is_held_out, is_wooded, normalize_agent, read_rows)
from chop.reports.markdown import END, START, mean_ci, table, write_pages

MINERL_DIR = DOCS_DIR / "minerl"
MODELS_FILE = RESULTS_DIR / "minerl_models.json"

# Our best agent, compared spot by spot with everything evaluated on the same spots and settings.
REFERENCE = "runs/natural/line5_wooded-spawns/20261119_005447/model.zip"
REFERENCE_NAME = "Line 5"
# Our agents whose standard natural evaluations (fixed spots, 1000 ticks, no cap, one-hit axe) are
# shown next to the MineRL models of the same type: the reference and our other best ones.
OUR_AGENTS = [REFERENCE,
              "runs/natural/line4c_resume-2M/20261118_183414/model.zip",
              "runs/natural/line4d_resume-3M/20261119_112830/model.zip"]

PAGES = [
    ("results.md", "Results", "one table per evaluation type (world, spots, episode length, log cap, axe)"),
    ("evaluation_runs.md", "Evaluation runs", "one row per evaluate.py run, newest first, with its settings"),
    ("training_runs.md", "Training runs", "MineRL-mode fine-tunes and from-scratch runs (train.py)"),
    ("setup.md", "MineRL mode", "what it copies from MineRL, how it was checked, what still differs"),
]

INDEX_TEMPLATE = f"""# CraftGround Experiments: MineRL Models in CraftGround

Models trained in the MineRL project (behavioural cloning on VPT videos, then PPO), run in
CraftGround's arena and natural world with MineRL's own view and actions ("MineRL mode",
`chop/envs/minerl.py`). Our own agents are in [natural/](../natural/README.md) and
[arena/](../arena/README.md); the summary of findings is [FINDINGS.md](../FINDINGS.md).

{START}
{END}

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
"""


def load_models() -> dict:
    try:
        return json.loads(MODELS_FILE.read_text(encoding="utf-8")).get("models", {})
    except (OSError, ValueError):
        return {}


def run_folder(agent: str) -> str:
    p = Path(agent)
    return (p.parent.parent if p.parent.name == "checkpoints" else p.parent).name


def is_minerl_mode(agent: str) -> bool:
    return agent == "random-minerl" or agent.startswith(("runs/minerl/", "runs/minerl_mode/"))


def label(agent: str, models: dict, natural_runs: list[dict]) -> str:
    if agent == "random-minerl":
        return "Random actions"
    if agent.startswith("runs/minerl/"):
        return models.get(run_folder(agent), {}).get("label", f"`{run_folder(agent)}`")
    return model_label(agent, natural_runs)  # also names MineRL-mode fine-tunes (agents.minerl_finetune)


# --- evaluation types ----------------------------------------------------------

def spots_of(row: dict) -> str:
    if row.get("env") == "arena":
        return "arena"
    world = " (held-out world)" if is_held_out(row) else ""
    if is_wooded(row):
        return "fixed wooded spots" + world
    if is_fixed(row):
        return "fixed forest spots" + world
    return ("random wooded spawns" if row.get("spots") == "random_wooded" else "random forest spawns") + world


def eval_type(row: dict) -> tuple:
    """(world, spots, episode ticks, log cap, axe): rows that share all five are comparable."""
    return (row.get("env", "natural"), spots_of(row), int(row.get("episode_ticks") or row.get("episode_steps")),
            int(row["log_cap"]) if row.get("log_cap") else 0, row.get("axe") or "instant")


def type_title(key: tuple) -> str:
    env, spots, ticks, cap, axe = key
    where = "Arena" if env == "arena" else f"Natural world, {spots}"
    cap_text = f", ends at {cap} logs" if cap else (", ends at 32 logs (all of them)" if env == "arena" else ", no log cap")
    axe_text = "iron axe (MineRL's)" if axe == "iron" else "one-hit axe (ours)"
    return f"{where} · {ticks:,}-tick episodes{cap_text} · {axe_text}"


def is_minerl_scoring(key: tuple) -> bool:
    """Scored the way the MineRL project scored its models (its 59.6 / 59.8 / 58.8)."""
    _, _, ticks, cap, axe = key
    return ticks == 4000 and cap == 64 and axe == "iron"


def reference_rows() -> list[dict]:
    """OUR_AGENTS' standard natural-world evaluations, tagged with the type they belong to."""
    rows = []
    for r in read_rows(NATURAL_EVAL_CSV):
        if normalize_agent(r["agent"]) in OUR_AGENTS and is_fixed(r) and not is_held_out(r):
            rows.append({**r, "env": "natural", "axe": "instant", "episode_ticks": r.get("episode_steps") or "1000",
                         "log_cap": "", "reference": "1"})
    return rows


def per_spot(rows: list[dict]) -> dict[int, float]:
    """Score per fixed spot, averaged if the spot was played more than once."""
    acc: dict[int, list[int]] = {}
    for r in rows:
        spots = (r.get("spot_per_episode") or "").split()
        for s, x in zip(spots, r["per_episode"].split()):
            if s.isdigit():
                acc.setdefault(int(s), []).append(int(x))
    return {s: sum(v) / len(v) for s, v in acc.items()}


def paired(a: dict[int, float], b: dict[int, float]) -> str:
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return "–"
    d = [a[s] - b[s] for s in shared]
    mean, ci = mean_ci(d)
    better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return f"{mean:+.1f} ± {ci:.1f} ({better} better / {worse} worse, {len(shared)} spots)"


def results_section(evals: list[dict], models: dict, natural_runs: list[dict]) -> str:
    if not evals:
        return "## Results\n\n_No MineRL-mode evaluations yet._"
    groups: dict[tuple, list[dict]] = {}
    for r in evals + reference_rows():
        groups.setdefault(eval_type(r), []).append(r)
    order = sorted(groups, key=lambda k: (k[0] != "natural", k[1], not is_minerl_scoring(k), k[2], k[3], k[4]))
    parts = []
    for key in order:
        if all(r.get("reference") for r in groups[key]):
            continue  # only our own agents: that's the natural report's table
        by_agent: dict[str, list[dict]] = {}
        for r in groups[key]:
            by_agent.setdefault(normalize_agent(r["agent"]), []).append(r)
        ref = per_spot(by_agent[REFERENCE]) if REFERENCE in by_agent else None
        minerl_col = is_minerl_scoring(key)
        rows = []
        for agent, rs in by_agent.items():
            eps = [int(x) for r in rs for x in r["per_episode"].split()]
            mean, ci = mean_ci(eps)
            row = [label(agent, models, natural_runs) + (" *(reference)*" if agent == REFERENCE else ""),
                   "MineRL mode" if is_minerl_mode(agent) else "ours (114×64, 1 tick)",
                   len(eps), f"**{mean:.1f}** ± {ci:.1f}", sorted(eps)[len(eps) // 2], sum(x == 0 for x in eps),
                   max(eps)]
            if minerl_col:
                score = models.get(run_folder(agent), {}).get("minerl_score")
                row.append(f"{score}" if score is not None and agent.startswith("runs/minerl/") else "–")
            if ref is not None:
                row.append("–" if agent == REFERENCE else paired(per_spot(rs), ref))
            rows.append((agent == "random-minerl", -mean, row))
        rows.sort(key=lambda t: (t[0], t[1]))
        headers = ["Model", "View & actions", "Spots played" if key[1].startswith("fixed") else "Episodes",
                   "Mean logs", "Median", "0-log episodes", "Max"]
        headers += ["In MineRL"] if minerl_col else []
        headers += [f"vs {REFERENCE_NAME}, spot by spot"] if ref is not None else []
        note = ("Scored the way the MineRL project scored these models (its 24 worlds, same rules), so "
                "**In MineRL** is directly comparable.\n\n" if minerl_col else "")
        parts.append(f"### {type_title(key)}\n\n{note}" + table(headers, [r for *_, r in rows]))
    return ("## Results\n\nOne table per **evaluation type**: the world and spots, episode length, log cap and "
            "axe. Only rows in the same table are comparable. Logs per episode; **±** is the 95% range of the "
            "mean. **vs " + REFERENCE_NAME + "** is the average difference per fixed spot over the spots both "
            "played, with its 95% range (a range that doesn't include 0 is a real difference). "
            "Our agents' 1,000-tick rows are their standard evaluations from the natural-world report "
            "(MineRL rows of that type are also shown there). **Spots played**: fixed spots are played in "
            "file order, so 50 = the first 50 of the 100; means over 50 and 100 spots can differ by which spots "
            "they include, so compare those rows with the spot-by-spot column, which only uses shared spots. "
            "MineRL mode runs 4 ticks per decision, so a 1,000-tick episode is 250 of its decisions.\n\n"
            + "\n\n".join(parts))


def evals_section(evals: list[dict], models: dict, natural_runs: list[dict]) -> str:
    if not evals:
        return "## Evaluation runs\n\n_None yet._"
    rows = [[r["timestamp"], label(normalize_agent(r["agent"]), models, natural_runs), f"`{r['agent']}`",
             r.get("env", ""), spots_of(r), r.get("episode_ticks", ""), r.get("log_cap") or "–", r.get("axe", ""),
             r["episodes"], r["mean_logs"], r.get("minutes") or "–", r.get("steps_per_s") or "–"]
            for r in reversed(evals)]
    return ("## Evaluation runs\n\nOne row per `evaluate.py` run that went to `results/eval_results_minerl.csv` "
            "(MineRL mode, or one of our models with a non-standard axe, episode length or log cap), newest first. "
            "**Steps/s** counts decisions (4 ticks each in MineRL mode).\n\n"
            + table(["When", "Model", "Path", "World", "Spots", "Ticks", "Log cap", "Axe", "Episodes",
                     "Mean logs", "Minutes", "Steps/s"], rows))


def training_section(runs: list[dict]) -> str:
    if not runs:
        return ("## Training runs\n\n_None yet. Fine-tune a MineRL model with `run.bat train.py --env natural "
                "--resume runs/minerl/<run>/final.zip`, or train from scratch in MineRL mode with "
                "`--actions minerl`._")
    rows = [[r["started"], f"`{r['resumed_from']}`" if r.get("resumed_from") else "from scratch",
             r["timesteps_done"], r["episodes"], r["minutes"], r["last20_avg_logs"], r["status"], f"`{r['model']}`",
             f"[open]({r['wandb_url']})" if r.get("wandb_url") else "–"] for r in reversed(runs)]
    return ("## Training runs\n\nOne row per MineRL-mode `train.py` run, newest first. **Steps** are decisions "
            "(4 ticks each) and include the MineRL model's own steps.\n\n"
            + table(["Started", "From", "Steps", "Episodes", "Minutes", "Last-20 avg logs", "Status", "Saved model",
                     "W&B"], rows))


def setup_section() -> str:
    m = minerl_env
    rows = [
        ["Turned on by", "`--actions minerl`, or automatically for a model with a 64×64 image and "
                         "`MultiDiscrete([3, 11, 9])` actions (`minerl.is_minerl_model`)"],
        ["Render", f"{m.RENDER_RES[0]}×{m.RENDER_RES[1]}, HUD shown (hand and held axe visible, as in MineRL)"],
        ["Observation", f"top {m.HUD_TOP_ROW} rows (hotbar, hearts, hunger cut off), `cv2.resize` to "
                        f"{m.OBS_W}×{m.OBS_H} with `INTER_AREA`: MineRL's `process_pov`"],
        ["Actions", f"`MultiDiscrete({m.NVEC})`: move {m.MOVES}, yaw {m.YAW_DEGREES}° (+ = right), "
                    f"pitch {m.PITCH_DEGREES}° (+ = down); attack always held"],
        ["Ticks per decision", f"{m.FRAME_SKIP}: keys held every tick, the turn split evenly over them "
                               "(as MineRL), logs added up"],
        ["Axe", "iron (MineRL's) by default; `--axe instant` for our one-hit Efficiency X axe. That one is a "
                "diamond axe, so the hand shows a light-blue axe with the enchantment shimmer instead of MineRL's "
                "plain iron one: a small visual difference in every one-hit-axe result and fine-tune (not tested "
                "on its own)"],
        ["Pop-ups", "every recipe unlocked on the first reset, so no \"New Recipes Unlocked!\" toast covers "
                    "the view (MineRL patched toasts out; `tools/check_minerl_toasts.py`)"],
        ["Field of view", "70° (Minecraft's default, as MineRL)"],
        ["Episode", "`--episode-ticks` (default 1,000; MineRL scored 4,000), `--log-cap` (MineRL: 64), "
                    "or death"],
        ["Models loaded from", "`runs/minerl/` (copies of `~/mineRL/runs/` in WSL); ImpalaCNN via "
                               "`chop/networks.py`, a copy of the MineRL project's `networks.py`"],
    ]
    return ("## MineRL mode\n\nThe env copies the MineRL project's `MineRLTreechopAxe320-v0` "
            "(`~/mineRL/train_treechop.py`), so its models run unchanged.\n\n"
            + table(["Setting", "Value"], rows)
            + "\n\n**Checked in the game** (`tools/check_minerl_compat.py`, 2026-11-19): every yaw and pitch "
              "choice turns the camera the right way by the right amount (within 0.6°: Minecraft turns in "
              "0.15° steps), forward walks on the ground, forward+jump jumps 1.25 blocks, and walking into a "
              "tree with attack held chops it with the iron axe.\n\n"
              "**Still different from MineRL:** CraftGround runs a newer Minecraft than MineRL's 1.16 "
              "(slightly different textures); worlds are peaceful (MineRL had mobs); and the worlds "
              f"themselves (MineRL: seeds 1-24; here: seed {natural_env.SEED}'s fixed spots, or the arena).")


def update(stamp: str | None = None) -> None:
    """Rebuild docs/minerl/ from results/eval_results_minerl.csv (and Line 5's standard evaluations)."""
    stamp = stamp or f"_Tables last updated {datetime.now():%Y-%m-%d %H:%M}._"
    evals, models = read_rows(MINERL_EVAL_CSV), load_models()
    natural_runs = read_rows(NATURAL_TRAIN_CSV)
    sections = {
        "results.md": results_section(evals, models, natural_runs),
        "evaluation_runs.md": evals_section(evals, models, natural_runs),
        "training_runs.md": training_section(read_rows(MINERL_TRAIN_CSV)),
        "setup.md": setup_section(),
    }
    write_pages(MINERL_DIR, "MineRL in CraftGround", INDEX_TEMPLATE, stamp, PAGES, sections)
