"""Human ceiling check: play fixed evaluation spots yourself with normal Minecraft controls.

    run.bat play_human.py                       # the 5 ceiling-check spots (all-biome spot file)
    run.bat play_human.py --spots 15 21         # other spots of the same file

Same episode as the agents' evaluation (evaluate.py --spots results/eval_spots_natural_wooded.json
--axe instant): the same spot and facing, the one-hit axe, 1,000 game ticks (50 s), logs counted
the same way. Unlike play.py you are not limited to the agent's actions: keys and mouse work
together, as in Minecraft, and you see the full 320x180 frame (the agent sees it shrunk to 64x64).

    W/A/S/D   move              mouse         look (click the window first)
    Space     jump              left button   attack (or hold F)
    Ctrl      sprint            Shift         sneak
    Esc       quit

Each spot waits for a click before its 1,000 ticks start. Results are added to
results/human_ceiling.csv (spot, logs, ticks), and each spot is saved as a video (the full frame
at 20 fps, real speed) in runs/human/videos/, named like the evaluation videos' logs.
"""

import argparse
import csv
import os
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageTk
from craftground.environment.action_space import no_op_v2

from chop.envs import natural as natural_env
from chop.envs.common import AXE, EPISODE_STEPS

# Spots 15, 21, 34, 68, 93 of the all-biome file: the 10/30/50/70/90th percentile of Impala BC->PPO
# + CG 4.0M's per-spot scores (unfair spots left out); that model averages 48.0 on them, its overall mean.
CEILING_SPOTS = [15, 21, 34, 68, 93]
SPOT_FILE = natural_env.ROOT / "results" / "eval_spots_natural_wooded.json"
RESULTS = natural_env.ROOT / "results" / "human_ceiling.csv"
VIDEOS = natural_env.ROOT / "runs" / "human" / "videos"
SCALE = 4              # 320x180 -> 1280x720
TICK_SECONDS = 0.05    # 20 ticks a second
SENSITIVITY = 0.15     # degrees per mouse pixel (about Minecraft's default)


class HumanPlayer:
    def __init__(self, env, spots: list[int], sensitivity: float):
        self.env, self.spots, self.sens = env, spots, sensitivity
        self.root = tk.Tk()
        self.root.title("CraftGround - human ceiling check")
        w, h = 320 * SCALE, 180 * SCALE
        self.canvas = tk.Label(self.root, width=w, height=h, bg="black", cursor="none")
        self.canvas.pack()
        self.status = tk.Label(self.root, font=("Consolas", 13), anchor="w", justify="left")
        self.status.pack(fill="x")
        self.center = (w // 2, h // 2)
        self.held: set[str] = set()
        self.dx = self.dy = 0.0
        self.attack = False
        self.playing = False
        self.running = True
        self.photo = None
        self.done: list[tuple[int, int]] = []
        self.video = None  # cv2.VideoWriter of the spot being played
        self.video_path = None
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", lambda e: self.held.discard(e.keysym.lower()))
        self.canvas.bind("<Motion>", self._motion)
        self.canvas.bind("<ButtonPress-1>", self._click)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "attack", False))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    # --- input -----------------------------------------------------------------
    def _press(self, event):
        key = event.keysym.lower()
        if key == "escape":
            self.quit()
        else:
            self.held.add(key)

    def _motion(self, event):
        if not self.playing:
            return
        cx, cy = self.center
        if (event.x, event.y) == (cx, cy):
            return  # our own warp back to the centre
        self.dx += event.x - cx
        self.dy += event.y - cy
        self.canvas.event_generate("<Motion>", warp=True, x=cx, y=cy)

    def _click(self, _event):
        if self.playing:
            self.attack = True
        elif self.waiting:
            self.waiting = False
            self.dx = self.dy = 0.0
            self.playing = True
            self._start_video()
            cx, cy = self.center
            self.canvas.event_generate("<Motion>", warp=True, x=cx, y=cy)
            self.root.after(1, self.tick)

    def action(self) -> dict:
        h = self.held
        act = no_op_v2()
        act["forward"], act["back"] = "w" in h, "s" in h
        act["left"], act["right"] = "a" in h, "d" in h
        act["jump"] = "space" in h
        act["sprint"] = bool({"control_l", "control_r"} & h)
        act["sneak"] = bool({"shift_l", "shift_r"} & h)
        act["attack"] = self.attack or "f" in h
        act["camera_yaw"], act["camera_pitch"] = self.dx * self.sens, self.dy * self.sens
        self.dx = self.dy = 0.0
        return act

    # --- episodes ----------------------------------------------------------------
    def show(self):
        pov = self.env.last_pov
        img = Image.fromarray(pov).resize((pov.shape[1] * SCALE, pov.shape[0] * SCALE), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.configure(image=self.photo, width=img.width, height=img.height)

    def next_spot(self):
        if len(self.done) == len(self.spots):
            self.finish()
            return
        n = len(self.done) + 1
        self.status.configure(text=f"spot {n}/{len(self.spots)} (#{self.spots[n - 1]}): teleporting, "
                                   "waiting for the terrain to draw...")
        self.root.update()
        self.env.reset()
        self.ticks, self.logs = 0, 0
        self.held.clear()
        self.attack = False
        self.show()
        self.waiting = True
        self.status.configure(text=f"spot {n}/{len(self.spots)} (#{self.spots[n - 1]}): click the picture "
                                   f"to start ({EPISODE_STEPS} ticks = {EPISODE_STEPS // 20} s)")

    def _start_video(self):
        VIDEOS.mkdir(parents=True, exist_ok=True)
        spot = self.spots[len(self.done)]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = VIDEOS / f"human_{stamp}_spot{spot}.mp4"
        h, w = self.env.last_pov.shape[:2]
        self.video = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        self._write_frame()

    def _write_frame(self):
        if self.video is not None:
            self.video.write(cv2.cvtColor(self.env.last_pov, cv2.COLOR_RGB2BGR))

    def _end_video(self, logs: int):
        if self.video is None:
            return
        self.video.release()
        self.video = None
        final = self.video_path.with_name(f"{self.video_path.stem}_{logs}logs.mp4")
        os.replace(self.video_path, final)
        print(f"video: {final}")

    def tick(self):
        if not self.running or not self.playing:
            return
        start = time.perf_counter()
        _, _, terminated, truncated, info = self.env.step(self.action())
        self.ticks += 1
        self.logs = info["logs_chopped"]
        self._write_frame()
        self.show()
        left = (EPISODE_STEPS - self.ticks) / 20
        self.status.configure(text=f"spot {len(self.done) + 1}/{len(self.spots)}  logs {self.logs}  "
                                   f"time left {left:4.1f} s   WASD move, mouse look, click attack, "
                                   "Space jump, Ctrl sprint, Esc quit")
        if terminated or truncated:
            self.playing = False
            spot = self.spots[len(self.done)]
            self.done.append((spot, self.logs))
            self._save(spot, self.logs, self.ticks)
            self._end_video(self.logs)
            print(f"spot {spot}: {self.logs} logs in {self.ticks} ticks")
            self.root.after(1500, self.next_spot)
            return
        wait = TICK_SECONDS - (time.perf_counter() - start)
        self.root.after(max(1, int(wait * 1000)), self.tick)

    def _save(self, spot: int, logs: int, ticks: int):
        new = not RESULTS.exists()
        with RESULTS.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["time", "spots_file", "spot", "logs", "ticks", "axe", "controls"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), SPOT_FILE.name, spot, logs, ticks,
                        "instant", "human: keyboard + mouse, full 320x180 view"])

    def finish(self):
        total = sum(l for _, l in self.done)
        text = "  ".join(f"#{s}: {l}" for s, l in self.done)
        self.status.configure(text=f"done. {text}   mean {total / len(self.done):.1f}   (Esc to quit)")
        print(f"mean over {len(self.done)} spots: {total / len(self.done):.1f}")

    def quit(self):
        self.running = False
        if self.video is not None:  # quit mid-spot: keep what was played
            self.video.release()
            self.video = None
            print(f"video (unfinished spot): {self.video_path}")
        self.root.destroy()

    def run(self):
        self.waiting = False
        self.root.after(1, self.next_spot)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spots", type=int, nargs="+", default=CEILING_SPOTS,
                        help=f"spot numbers in {SPOT_FILE.name}, played in this order. Default: {CEILING_SPOTS}")
    parser.add_argument("--sensitivity", type=float, default=SENSITIVITY, help="degrees per mouse pixel")
    parser.add_argument("--port", type=int, default=8031,
                        help="use a port no training/eval script is on, or they'll reset each other")
    args = parser.parse_args()

    from chop.infra.instances import clear_stale_loom_locks, env_path_for, kill_instance_processes
    kill_instance_processes([args.port])
    clear_stale_loom_locks()
    all_spots = natural_env.load_eval_spots(SPOT_FILE, seed=natural_env.SEED)
    env = natural_env.make_env(port=args.port, env_path=env_path_for(args.port, verbose=True),
                               spots=[all_spots[i] for i in args.spots], seed=natural_env.SEED,
                               actions="minerl", spawn="wooded", axe=AXE, episode_ticks=EPISODE_STEPS,
                               minerl_ticks=1)
    env._game_action = lambda act: act  # raw CraftGround V2 actions from the keyboard and mouse
    try:
        HumanPlayer(env, args.spots, args.sensitivity).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
