# MineRL in CraftGround: MineRL mode

[← MineRL in CraftGround index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 22:06._

The env copies the MineRL project's `MineRLTreechopAxe320-v0` (`~/mineRL/train_treechop.py`), so its models run unchanged.

| Setting | Value |
| --- | --- |
| Turned on by | `--actions minerl`, or automatically for a model with a 64×64 image and `MultiDiscrete([3, 11, 9])` actions (`minerl.is_minerl_model`) |
| Render | 320×180, HUD shown (hand and held axe visible, as in MineRL) |
| Observation | top 138 rows (hotbar, hearts, hunger cut off), `cv2.resize` to 64×64 with `INTER_AREA`: MineRL's `process_pov` |
| Actions | `MultiDiscrete([3, 11, 9])`: move ['stand', 'forward', 'forward_jump'], yaw [-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40]° (+ = right), pitch [-20, -10, -5, -2, 0, 2, 5, 10, 20]° (+ = down); attack always held |
| Ticks per decision | 4: keys held every tick, the turn split evenly over them (as MineRL), logs added up |
| Axe | iron (MineRL's) by default; `--axe instant` for our one-hit Efficiency X axe |
| Pop-ups | every recipe unlocked on the first reset, so no "New Recipes Unlocked!" toast covers the view (MineRL patched toasts out; `tools/check_minerl_toasts.py`) |
| Field of view | 70° (Minecraft's default, as MineRL) |
| Episode | `--episode-ticks` (default 1,000; MineRL scored 4,000), `--log-cap` (MineRL: 64), or death |
| Models loaded from | `runs/minerl/` (copies of `~/mineRL/runs/` in WSL); ImpalaCNN via `chop/networks.py`, a copy of the MineRL project's `networks.py` |

**Checked in the game** (`tools/check_minerl_compat.py`, 2026-11-19): every yaw and pitch choice turns the camera the right way by the right amount (within 0.6°: Minecraft turns in 0.15° steps), forward walks on the ground, forward+jump jumps 1.25 blocks, and walking into a tree with attack held chops it with the iron axe.

**Still different from MineRL:** CraftGround runs a newer Minecraft than MineRL's 1.16 (slightly different textures); worlds are peaceful (MineRL had mobs); and the worlds themselves (MineRL: seeds 1-24; here: seed 20261114's fixed spots, or the arena).
<!-- AUTO-GENERATED:END -->
