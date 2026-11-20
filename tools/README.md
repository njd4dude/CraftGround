# tools/

One-off diagnostics, kept for re-running when something changes. Run them from the
repository root, e.g. `run.bat tools\memtest.py --minutes 20`.

| Script | What it checks |
| --- | --- |
| `smoke_test.py` | CraftGround starts, resets and steps at all (no tree chopping) |
| `check_env.py` | The arena: a scripted agent walks to a tree, chops it, and the tree regrows on reset |
| `check_arena_jump.py` | The arena with `--jump`: same action space as the natural world, jump lifts the player, walls still hold |
| `check_minerl_compat.py` | MineRL mode (`--actions minerl`): the 64x64 view, every turn size and direction, chopping with the iron axe |
| `check_minerl_toasts.py` | MineRL mode: plays a MineRL model and saves frames, to check no pop-ups (recipe toasts) cover the view |
| `check_natural.py` | The natural world: a few resets, saving what the agent sees to `check_frames/` |
| `compare_render_distance.py` | Side-by-side videos of the same spots at two render distances |
| `launch_test.py` | Whether launching many instances at once crashes, and why (`results/launch_test.csv`) |
| `memtest.py` | How much memory natural-world instances use over time (`results/memtest.csv`) |
| `memprobe.py` | Where running instances' memory goes, next to training (`results/memprobe.csv`) |
| `wandb_lineage.py` | Sets the W&B group and `lineage:` tags on runs logged before train.py did (`--apply` to write) |
