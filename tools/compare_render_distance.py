"""Side-by-side videos of the same spots at two render distances.

    python tools/compare_render_distance.py                     # render distance 4 vs 6, spots 0 5 10 15
    python tools/compare_render_distance.py --distances 4 8 --spots 3 7

Two instances play the same fixed evaluation spots (results/eval_spots_natural.json) with the
same scripted actions, so the only difference between the two videos is the render distance:
  1. a slow full turn: 24 turns of 15 degrees, holding each direction for HOLD steps so the
     terrain there has time to draw;
  2. a straight walk forward (WALK steps, jumping over ledges).
Writes runs/tools/render_distance/spot<N>_rd<a>_vs_rd<b>.mp4 (left: first distance, right: second),
at the agent's own 114x64 view enlarged, plus one contact sheet of matching frames.
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs import natural as natural_env
from chop.infra.vec import close_vec_env, game_folder, instance_ports, launch_vec_env
from chop.envs.common import ACTION_NAMES_JUMP

OUT = Path("runs/tools/render_distance")
SCALE = 4
HOLD = 10   # steps facing each direction during the turn
WALK = 160  # steps walking forward afterwards
A = {name: i for i, name in enumerate(ACTION_NAMES_JUMP)}


def script() -> list[int]:
    turn = []
    for _ in range(24):
        turn += [A["turn_right"]] + [A["noop"]] * (HOLD - 1)
    return turn + [A["forward_jump"] if k % 8 == 0 else A["forward"] for k in range(WALK)]


def label(frame: np.ndarray, text: str) -> Image.Image:
    img = Image.fromarray(frame).resize((frame.shape[1] * SCALE, frame.shape[0] * SCALE), Image.NEAREST)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances", type=int, nargs=2, default=[4, 6])
    parser.add_argument("--spots", type=int, nargs="+", default=[0, 5, 10, 15])
    parser.add_argument("--port", type=int, default=8023)
    args = parser.parse_args()

    all_spots = natural_env.load_eval_spots()
    spots = [all_spots[i] for i in args.spots]
    ports = instance_ports(args.port, 2)
    env = launch_vec_env([lambda p=p, rd=rd: natural_env.make_env(port=p, env_path=game_folder(p),
                                                                  render_distance=rd, spots=spots)
                          for p, rd in zip(ports, args.distances)], ports)
    OUT.mkdir(parents=True, exist_ok=True)
    actions = script()
    sheet_rows = []
    a, b = args.distances
    try:
        for spot in args.spots:
            obs = env.reset()
            frames = [[obs[0]], [obs[1]]]
            for act in actions:
                obs, _, dones, infos = env.step(np.array([act, act]))
                for i in range(2):
                    frames[i].append(infos[i]["terminal_observation"] if dones[i] else obs[i])
            path = OUT / f"spot{spot}_rd{a}_vs_rd{b}.mp4"
            with imageio.get_writer(path, fps=20, codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=1) as w:
                for k, (fa, fb) in enumerate(zip(*frames)):
                    left = label(fa, f"render distance {a}   spot {spot}   step {k}")
                    right = label(fb, f"render distance {b}")
                    both = Image.new("RGB", (left.width * 2 + 8, left.height), (255, 255, 255))
                    both.paste(left, (0, 0)); both.paste(right, (left.width + 8, 0))
                    w.append_data(np.asarray(both))
            print(f"spot {spot}: {path}", flush=True)
            # Matching frames for the sheet: 4 directions during the turn, plus mid-walk.
            picks = [HOLD * k - 1 for k in (1, 7, 13, 19)] + [len(actions) - WALK // 2]
            sheet_rows.append([(label(frames[0][t], f"rd {a}  spot {spot}  step {t}"),
                                label(frames[1][t], f"rd {b}")) for t in picks])
    finally:
        close_vec_env(env, ports)

    tw, th = sheet_rows[0][0][0].size
    sheet = Image.new("RGB", ((tw * 2 + 8) * len(sheet_rows[0]) // 2, th * len(sheet_rows) * 2), "white")
    for r, row in enumerate(sheet_rows):
        for c, (fa, fb) in enumerate(row[:4]):
            x = (c % 2) * (tw * 2 + 8)
            y = (r * 2 + c // 2) * th
            sheet.paste(fa, (x, y)); sheet.paste(fb, (x + tw + 4, y))
    sheet_path = OUT / f"sheet_rd{a}_vs_rd{b}.png"
    sheet.save(sheet_path)
    print(f"contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
