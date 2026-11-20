"""Which pop-ups ("toasts") show up in MineRL mode, and whether reset clears them.

MineRL patched toasts out of Minecraft, so a model trained there never saw one; here a log pickup
unlocks recipes ("New Recipes Unlocked!"), which draws a box in the top-right of the view.
Plays two arena episodes with a MineRL model and saves the full frame every 25 decisions to
check_frames/toasts/, so you can see when toasts appear.

    run.bat tools/check_minerl_toasts.py runs/minerl/<run>/final.zip
"""

import sys
from pathlib import Path

from PIL import Image
from stable_baselines3 import PPO

import _bootstrap  # noqa: F401  (makes the chop package importable from tools/)
from chop.envs import minerl
from chop.envs.arena import make_env

OUT = Path("check_frames/toasts")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    minerl.register_networks()
    model = PPO.load(sys.argv[1], device="cpu")
    env = make_env(port=8123, actions="minerl")
    try:
        for ep in range(2):
            obs, _ = env.reset()
            for t in range(env.max_steps):
                action, _ = model.predict(obs, deterministic=False)
                obs, _, term, trunc, info = env.step(action)
                if t % 25 == 0:
                    Image.fromarray(env.last_pov).save(OUT / f"ep{ep}_t{t:03d}_logs{info['logs_chopped']}.png")
                if term or trunc:
                    break
            print(f"episode {ep}: {info['logs_chopped']} logs in {t + 1} decisions")
    finally:
        env.close()


if __name__ == "__main__":
    main()
