"""Play the tree-chopping env yourself, with the keyboard, seeing exactly what the agent sees.

    run.bat play.py                 # natural world (training seed)
    run.bat play.py --held-out      # natural world the agent never trains on
    run.bat play.py --arena         # the flat 8-oak arena (chop/envs/arena.py)

Each step you get one of the agent's actions, no more: the env takes one discrete action per
game tick (1/20 s), so e.g. you can't walk and turn at the same time, just like the agent.
When several keys are held, the first match in this list wins:

    F + W      forward_attack        F (or Enter)   attack
    Space + W  forward_jump          Space          jump
    W / S      forward / back        A / D          strafe left / right
    Left/Right turn                  Up/Down        look up / down
    R          start a new episode   Esc            quit

The window shows the agent's 114x64 image scaled up, plus step, reward and logs so far.
"""

import argparse
import time
import tkinter as tk

from PIL import Image, ImageTk

from chop.envs.common import EPISODE_STEPS

SCALE = 7          # 114x64 -> 798x448
TICK_SECONDS = 0.05  # 20 steps a second, Minecraft's tick rate

# (keys that must all be held, action name), checked in order.
BINDINGS = [
    ({"f", "w"}, "forward_attack"),
    ({"return", "w"}, "forward_attack"),
    ({"f"}, "attack"),
    ({"return"}, "attack"),
    ({"space", "w"}, "forward_jump"),
    ({"space"}, "jump"),
    ({"w"}, "forward"),
    ({"s"}, "back"),
    ({"a"}, "strafe_left"),
    ({"d"}, "strafe_right"),
    ({"left"}, "turn_left"),
    ({"right"}, "turn_right"),
    ({"up"}, "look_up"),
    ({"down"}, "look_down"),
]


def make_env(args):
    # Own game folder per port, like train.py / evaluate.py, so this can run alongside training.
    from chop.infra.instances import clear_stale_loom_locks, env_path_for, kill_instance_processes
    kill_instance_processes([args.port])  # leftovers of an earlier play.py on this port only
    clear_stale_loom_locks()
    env_path = env_path_for(args.port, verbose=True)
    if args.arena:
        from chop.envs.arena import ChopTreesEnv
        return ChopTreesEnv(port=args.port, env_path=env_path)
    from chop.envs import natural as natural_env
    seed = natural_env.HELD_OUT_SEED if args.held_out else natural_env.SEED
    return natural_env.NaturalChopTreesEnv(port=args.port, env_path=env_path, seed=seed)


class Player:
    def __init__(self, env):
        self.env = env
        self.names = list(getattr(env, "action_names", None) or _arena_names())
        self.held: set[str] = set()
        self.root = tk.Tk()
        self.root.title("CraftGround - play as the agent")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status = tk.Label(self.root, font=("Consolas", 12), anchor="w", justify="left")
        self.status.pack(fill="x")
        self.root.bind("<KeyPress>", self._press)
        self.root.bind("<KeyRelease>", self._release)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.running = True
        self.photo = None

    def _press(self, event):
        key = event.keysym.lower()
        if key == "escape":
            self.quit()
        elif key == "r":
            self.new_episode()
        else:
            self.held.add(key)

    def _release(self, event):
        self.held.discard(event.keysym.lower())

    def action(self) -> str:
        for keys, name in BINDINGS:
            if keys <= self.held and name in self.names:
                return name
        return "noop"

    def show(self, image):
        img = Image.fromarray(image).resize((image.shape[1] * SCALE, image.shape[0] * SCALE),
                                            Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.label.configure(image=self.photo)

    def new_episode(self):
        self.status.configure(text="resetting (finding a spot, waiting for terrain)...")
        self.root.update()
        image, _ = self.env.reset()
        self.steps, self.total, self.logs = 0, 0.0, 0
        self.show(image)
        spot = getattr(self.env, "spot", None)
        self.where = f"  spot ({spot[0]:.0f}, {spot[1]:.0f}, {spot[2]:.0f})" if spot else ""
        self.held.clear()

    def tick(self):
        if not self.running:
            return
        start = time.perf_counter()
        name = self.action()
        image, reward, terminated, truncated, info = self.env.step(self.names.index(name))
        self.steps += 1
        self.total += reward
        self.logs = info.get("logs_chopped", self.logs)
        self.show(image)
        self.status.configure(
            text=f"step {self.steps}/{EPISODE_STEPS}  action {name:<15} reward {self.total:.0f}"
                 f"  logs {self.logs}{self.where}\n"
                 "W/S/A/D move  arrows look  F attack  Space jump  R new episode  Esc quit")
        if terminated or truncated:
            print(f"episode over: {self.logs} logs in {self.steps} steps")
            self.new_episode()
        wait = TICK_SECONDS - (time.perf_counter() - start)
        self.root.after(max(1, int(wait * 1000)), self.tick)

    def quit(self):
        self.running = False
        self.root.destroy()

    def run(self):
        self.new_episode()
        self.root.after(1, self.tick)
        self.root.mainloop()


def _arena_names():
    from chop.envs.common import ACTION_NAMES
    return ACTION_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arena", action="store_true", help="the flat 8-oak arena instead of the natural world")
    parser.add_argument("--held-out", action="store_true", help="natural world with the held-out seed")
    parser.add_argument("--port", type=int, default=8030,
                        help="use a port no training/eval script is on, or they'll reset each other")
    args = parser.parse_args()
    env = make_env(args)
    try:
        Player(env).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
