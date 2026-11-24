"""Pictures of every fixed evaluation spot, for reviewing the spot sets by eye.

    run.bat tools/review_spots.py --spots results/eval_spots_natural_anywhere.json
    run.bat tools/review_spots.py --spots results/eval_spots_natural.json --limit 3   # quick test
    run.bat tools/review_spots.py --build-only    # remake the pages from saved pictures + latest results

For each spot (same teleport and wait as an evaluation episode): the first view, at 4x the agent's
resolution (456x256; the agent itself sees 114x64, shown pixelated next to it), then the views to
the right, behind and to the left, a top-down view from 50 blocks up (spectator mode), and the
biome. Pictures go to check_frames/spot_review/<set>/ (not in git); open
check_frames/spot_review/index.html. Each spot card also shows what some agents chopped there in
their latest evaluation (logs, and stuck ticks if recorded), and has a notes box saved in the
browser, with a button to export all notes as JSON.

Spots already pictured are skipped, so an interrupted run continues where it stopped.
"""

import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs import natural as natural_env
from chop.envs.common import IMAGE_HEIGHT, IMAGE_WIDTH
from chop.reports.csvio import MINERL_EVAL_CSV, NATURAL_EVAL_CSV, normalize_agent, read_rows

OUT = Path("check_frames/spot_review")
SCALE = 4
SETS = {"forest": "results/eval_spots_natural.json",
        "wooded": "results/eval_spots_natural_wooded.json",
        "anywhere": "results/eval_spots_natural_anywhere.json"}
SET_TITLES = {"forest": "Oak/birch forests", "wooded": "All tree biomes", "anywhere": "Anywhere on land"}
VIEWS = ["front", "right", "back", "left", "top"]
TOP_HEIGHT = 50
# Agents whose per-spot scores the cards show (latest standard evaluation of each).
AGENTS = {
    "Line 5d": "runs/natural/line5d_stuck-penalty/20261122_223446/model.zip",
    "Line 5b": "runs/natural/line5b_resume-4M/20261119_161818/model.zip",
    "Line 5": "runs/natural/line5_wooded-spawns/20261119_005447/model.zip",
    "Impala BC→PPO + CG 2M": "runs/minerl_finetuned/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/"
                             "checkpoints/ppo_chop_615292_steps.zip",
}
BIOMES = [
    "plains", "sunflower_plains", "snowy_plains", "ice_spikes", "desert", "swamp", "mangrove_swamp",
    "forest", "flower_forest", "birch_forest", "dark_forest", "old_growth_birch_forest",
    "old_growth_pine_taiga", "old_growth_spruce_taiga", "taiga", "snowy_taiga", "savanna",
    "savanna_plateau", "windswept_hills", "windswept_gravelly_hills", "windswept_forest",
    "windswept_savanna", "jungle", "sparse_jungle", "bamboo_jungle", "badlands", "eroded_badlands",
    "wooded_badlands", "meadow", "cherry_grove", "grove", "snowy_slopes", "frozen_peaks", "jagged_peaks",
    "stony_peaks", "river", "frozen_river", "beach", "snowy_beach", "stony_shore", "mushroom_fields",
    "ocean", "deep_ocean", "warm_ocean", "lukewarm_ocean", "deep_lukewarm_ocean", "cold_ocean",
    "deep_cold_ocean", "frozen_ocean", "deep_frozen_ocean", "dripstone_caves", "lush_caves", "deep_dark",
]


def set_name(path: str) -> str:
    stem = Path(path).stem
    return next((k for k in ("wooded", "anywhere") if stem.endswith(f"_{k}")), "forest")


def capture(spots_file: str, port: int, limit: int | None):
    name = set_name(spots_file)
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    spots = natural_env.load_eval_spots(Path(spots_file))
    todo = [s for s in spots if not (folder / f"spot{s['index']:03d}_top.jpg").exists()][:limit]
    print(f"{name}: {len(spots)} spots, {len(todo)} to picture")
    if not todo:
        return
    # Minecraft draws at 4x the agent's resolution (the agent's view is the same picture, smaller).
    natural_env.IMAGE_WIDTH, natural_env.IMAGE_HEIGHT = IMAGE_WIDTH * SCALE, IMAGE_HEIGHT * SCALE
    env = natural_env.NaturalChopTreesEnv(port=port, spots=todo)
    seen_biome = {}

    def command(cmd, replies, timeout=natural_env.REPLY_TIMEOUT_S):
        """env._command, but also remembers any BIOME= line that passes by."""
        env.env.unwrapped.add_commands([cmd] if isinstance(cmd, str) else cmd)
        deadline = time.monotonic() + timeout
        while True:
            obs, _, _, _, _ = env.env.step(env._noop)
            hit = None
            for m in obs["full"].chat_messages:
                if "BIOME=" in m.message:
                    seen_biome["b"] = m.message.split("BIOME=", 1)[1].strip().strip('"')
                if any(r in m.message for r in replies):
                    hit = m.message
            if hit or time.monotonic() > deadline:
                return obs, hit

    def settle(obs):
        for _ in range(3):  # the image shows a change 1 tick late
            obs, _, _, _, _ = env.env.step(env._noop)
        return env._wait_for_render(obs)

    def save(obs, view, idx):
        Image.fromarray(np.asarray(obs["pov"])).save(folder / f"spot{idx:03d}_{view}.jpg", quality=88)

    try:
        for s in todo:
            t0 = time.monotonic()
            idx = s["index"]
            _, info = env.reset()
            obs, _ = command("execute as @p at @s run tp @s ~ ~ ~ ~ 0", ("Teleported",))
            obs = settle(obs)
            save(obs, "front", idx)
            Image.fromarray(np.asarray(obs["pov"])).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BOX).save(
                folder / f"spot{idx:03d}_agent.png")
            f = obs["full"]
            pos = {"x": round(f.x, 1), "y": round(f.y, 1), "z": round(f.z, 1)}
            seen_biome.clear()
            command([f'execute as @p at @s if biome ~ ~ ~ minecraft:{b} run tellraw @p "BIOME={b}"' for b in BIOMES]
                    + ['tellraw @p "BIOME_DONE"'], ("BIOME_DONE",), timeout=5)
            for _ in range(3):
                command('tellraw @p "BIOME_WAIT"', ("BIOME_WAIT",), timeout=2)
            for view in ("right", "back", "left"):
                obs, _ = command("execute as @p at @s run tp @s ~ ~ ~ ~90 0", ("Teleported",))
                save(settle(obs), view, idx)
            command("gamemode spectator @p", ("game mode", "Game mode"))
            obs, _ = command(f"execute as @p at @s run tp @s ~ ~{TOP_HEIGHT} ~ {s['yaw']} 90", ("Teleported",))
            save(settle(obs), "top", idx)
            command("gamemode survival @p", ("game mode", "Game mode"))
            meta[str(idx)] = {**{k: s[k] for k in ("x", "y", "z", "yaw")}, "landed": pos,
                              "biome": seen_biome.get("b", "?"), "teleported": bool(info.get("found"))}
            meta_path.write_text(json.dumps(meta, indent=1))
            print(f"  spot {idx}: {meta[str(idx)]['biome']} at ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f}) "
                  f"in {time.monotonic() - t0:.0f}s", flush=True)
    finally:
        env.close()


# --- the pages ---------------------------------------------------------------------------------

def agent_scores(spots_file: str) -> dict[str, dict[int, tuple[int, int | None]]]:
    """{agent label: {spot index: (logs, stuck ticks or None)}} from each agent's latest standard
    evaluation on this spot set."""
    rows = read_rows(NATURAL_EVAL_CSV) + [r for r in read_rows(MINERL_EVAL_CSV)
                                          if r.get("env") in ("", "natural") and not r.get("log_cap")
                                          and (r.get("episode_ticks") or "1000") == "1000"]
    key = f"fixed:{Path(spots_file).name}"
    out = {}
    for label, agent in AGENTS.items():
        mine = [r for r in rows if normalize_agent(r["agent"]) == agent and r.get("spots") == key
                and r.get("episode_steps") == "1000"]
        if not mine:
            continue
        r = max(mine, key=lambda r: (bool(r.get("stuck_per_episode")), r["timestamp"]))
        logs = list(map(int, r["per_episode"].split()))
        stuck = [int(x) for x in (r.get("stuck_per_episode") or "").split()] or [None] * len(logs)
        out[label] = {int(s): (l, st) for s, l, st in zip(r["spot_per_episode"].split(), logs, stuck)}
    return out


PAGE_CSS = """
:root { --bg:#f6f6f4; --card:#fff; --ink:#1b1b1b; --muted:#666; --line:#ddd; --zero:#c0392b; --good:#1e7d45; }
@media (prefers-color-scheme: dark) { :root { --bg:#151515; --card:#1f1f1f; --ink:#eee; --muted:#9a9a9a;
  --line:#333; --zero:#ff6b5b; --good:#5fd08a; } }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui, sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line);
  padding:10px 16px; display:flex; flex-wrap:wrap; gap:12px; align-items:center; }
header h1 { font-size:17px; margin:0 12px 0 0; }
header a { color:inherit; }
main { padding:16px; display:grid; gap:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; }
.card h2 { font-size:15px; margin:0 0 6px; display:flex; flex-wrap:wrap; gap:10px; align-items:baseline; }
.card h2 small { color:var(--muted); font-weight:400; }
.row { display:flex; flex-wrap:wrap; gap:8px; align-items:flex-start; }
figure { margin:0; } figcaption { font-size:12px; color:var(--muted); }
img { display:block; border-radius:4px; max-width:100%; cursor:zoom-in; }
img.big { width:456px; } img.agent { width:228px; image-rendering:pixelated; }
img.small { width:224px; } img.top { width:256px; height:256px; object-fit:cover; }
.scores { display:flex; flex-wrap:wrap; gap:6px; margin:6px 0; }
.chip { border:1px solid var(--line); border-radius:12px; padding:1px 8px; font-size:12px; }
.chip.zero { color:var(--zero); border-color:var(--zero); } .chip.hi { color:var(--good); border-color:var(--good); }
.notes { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin-top:8px; }
.notes label { font-size:12px; border:1px solid var(--line); border-radius:12px; padding:2px 8px; cursor:pointer; }
.notes textarea { flex:1 1 260px; min-height:34px; background:var(--bg); color:var(--ink);
  border:1px solid var(--line); border-radius:6px; padding:4px 6px; font:inherit; }
#zoom { position:fixed; inset:0; background:rgba(0,0,0,.85); display:none; align-items:center;
  justify-content:center; z-index:10; cursor:zoom-out; }
#zoom img { max-width:96vw; max-height:92vh; width:auto; image-rendering:auto; }
select, button { font:inherit; }
"""

TAGS = ["no tree in reach", "trees far", "wall / cliff", "water", "fine"]

PAGE_JS = """
const SET = document.body.dataset.set, KEY = 'spot-notes-' + SET;
let notes = {}; try { notes = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
function save() { try { localStorage.setItem(KEY, JSON.stringify(notes)); } catch (e) {} count(); }
function count() { document.getElementById('reviewed').textContent =
  Object.values(notes).filter(n => (n.tags && n.tags.length) || n.text).length; }
document.querySelectorAll('.card').forEach(card => {
  const id = card.dataset.spot, n = notes[id] || {tags: [], text: ''};
  card.querySelectorAll('input[type=checkbox]').forEach(cb => {
    cb.checked = (n.tags || []).includes(cb.value);
    cb.onchange = () => { const t = new Set((notes[id] || {}).tags || []);
      cb.checked ? t.add(cb.value) : t.delete(cb.value);
      notes[id] = {...(notes[id] || {}), tags: [...t]}; save(); };
  });
  const ta = card.querySelector('textarea'); ta.value = n.text || '';
  ta.oninput = () => { notes[id] = {...(notes[id] || {}), text: ta.value}; save(); };
});
document.getElementById('export').onclick = () => {
  const blob = new Blob([JSON.stringify({set: SET, notes}, null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'spot_notes_' + SET + '.json'; a.click();
};
const main = document.querySelector('main'), cards = [...document.querySelectorAll('.card')];
function apply() {
  const sort = document.getElementById('sort').value, filt = document.getElementById('filter').value;
  const v = c => parseFloat(c.dataset[sort] ?? 'NaN');
  cards.sort((a, b) => sort === 'spot' ? a.dataset.spot - b.dataset.spot : (v(a) - v(b)) || (a.dataset.spot - b.dataset.spot));
  cards.forEach(c => { main.appendChild(c);
    const n = notes[c.dataset.spot] || {};
    c.style.display = filt === 'all' || (filt === 'zero' && c.dataset.allzero === '1')
      || (filt === 'anyzero' && c.dataset.anyzero === '1')
      || (filt === 'unreviewed' && !((n.tags && n.tags.length) || n.text)) ? '' : 'none'; });
}
document.getElementById('sort').onchange = apply; document.getElementById('filter').onchange = apply;
const zoom = document.getElementById('zoom'), zimg = zoom.querySelector('img');
document.querySelectorAll('main img').forEach(i => i.onclick = () => { zimg.src = i.src;
  zimg.style.imageRendering = i.classList.contains('agent') ? 'pixelated' : 'auto'; zoom.style.display = 'flex'; });
zoom.onclick = () => zoom.style.display = 'none';
count(); apply();
"""


def build_page(name: str, spots_file: str) -> int:
    folder = OUT / name
    meta_path = folder / "meta.json"
    if not meta_path.exists():
        return 0
    meta = json.loads(meta_path.read_text())
    scores = agent_scores(spots_file)
    cards = []
    for idx in sorted(meta, key=int):
        m, i = meta[idx], int(idx)
        chips, data = [], {}
        for label, by_spot in scores.items():
            if i not in by_spot:
                continue
            logs, stuck = by_spot[i]
            cls = "zero" if logs == 0 else "hi" if logs >= 40 else ""
            st = f", stuck {stuck / 10:.0f}%" if stuck is not None else ""
            chips.append(f'<span class="chip {cls}">{html.escape(label)}: {logs} logs{st}</span>')
            data[label] = logs
        logs_all = list(data.values())
        attrs = (f'data-spot="{i}" data-allzero="{int(bool(logs_all) and max(logs_all) == 0)}" '
                 f'data-anyzero="{int(0 in logs_all)}" '
                 + (f'data-l5d="{data["Line 5d"]}" ' if "Line 5d" in data else "")
                 + (f'data-mean="{sum(logs_all) / len(logs_all):.2f}" ' if logs_all else ""))
        p = f"spot{i:03d}"
        tags = "".join(f'<label><input type="checkbox" value="{t}"> {t}</label>' for t in TAGS)
        cards.append(f"""<section class="card" {attrs}>
<h2>Spot {i} <small>{html.escape(m['biome'].replace('_', ' '))} · x {m['x']:.0f}, y {m['y']:.0f}, z {m['z']:.0f} · facing yaw {m['yaw']}</small></h2>
<div class="scores">{''.join(chips) or '<span class="chip">no agent results</span>'}</div>
<div class="row">
<figure><img class="big" src="{name}/{p}_front.jpg" loading="lazy" alt="first view"><figcaption>First view (4× the agent's resolution)</figcaption></figure>
<figure><img class="agent" src="{name}/{p}_agent.png" loading="lazy" alt="agent view"><figcaption>What the agent sees (114×64)</figcaption></figure>
<figure><img class="top" src="{name}/{p}_top.jpg" loading="lazy" alt="top-down"><figcaption>From {TOP_HEIGHT} blocks up, start at the centre, first view = up</figcaption></figure>
</div>
<div class="row" style="margin-top:8px">
<figure><img class="small" src="{name}/{p}_right.jpg" loading="lazy" alt="right"><figcaption>Right</figcaption></figure>
<figure><img class="small" src="{name}/{p}_back.jpg" loading="lazy" alt="behind"><figcaption>Behind</figcaption></figure>
<figure><img class="small" src="{name}/{p}_left.jpg" loading="lazy" alt="left"><figcaption>Left</figcaption></figure>
</div>
<div class="notes">{tags}<textarea placeholder="Notes on spot {i}"></textarea></div>
</section>""")
    nav = " · ".join(f'<a href="{k}.html">{SET_TITLES[k]}</a>' for k in SETS if (OUT / k / "meta.json").exists())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spot review: {SET_TITLES[name]}</title><style>{PAGE_CSS}</style></head>
<body data-set="{name}"><header><h1>{SET_TITLES[name]} · {len(cards)} spots</h1><span>{nav}</span>
<label>Sort <select id="sort"><option value="spot">spot number</option><option value="l5d">Line 5d logs, low first</option>
<option value="mean">mean logs, low first</option></select></label>
<label>Show <select id="filter"><option value="all">all</option><option value="zero">0 logs for every agent</option>
<option value="anyzero">0 logs for some agent</option><option value="unreviewed">not reviewed yet</option></select></label>
<span><b id="reviewed">0</b> reviewed</span><button id="export">Export notes (JSON)</button></header>
<main>{''.join(cards)}</main><div id="zoom"><img alt=""></div><script>{PAGE_JS}</script></body></html>"""
    (OUT / f"{name}.html").write_text(page, encoding="utf-8")
    return len(cards)


def build_all():
    links = []
    for name, f in SETS.items():
        n = build_page(name, f)
        if n:
            links.append(f'<li><a href="{name}.html">{SET_TITLES[name]}</a> ({n} spots, <code>{f}</code>)</li>')
    (OUT / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>Spot review</title><style>{PAGE_CSS} body{{padding:16px}}</style>
<h1>Fixed evaluation spots</h1><ul>{''.join(links)}</ul>""", encoding="utf-8")
    print(f"pages in {OUT.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spots", action="append", help="spot file(s) to picture (default: all three sets)")
    parser.add_argument("--port", type=int, default=8023,
                        help="use a port no other script is on; two scripts on one port reset each other's game")
    parser.add_argument("--limit", type=int, help="picture at most this many spots per set (a quick test)")
    parser.add_argument("--build-only", action="store_true", help="only remake the pages")
    args = parser.parse_args()
    if not args.build_only:
        for f in args.spots or SETS.values():
            capture(f, args.port, args.limit)
    build_all()


if __name__ == "__main__":
    main()
