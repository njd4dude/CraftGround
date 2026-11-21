# Reference: environments

How the arena, the natural world and MineRL mode are set up.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 15:32._

## Arena

| Setting | Value |
| --- | --- |
| World | Superflat arena, stone-brick walls at ±13 blocks |
| Trees | 8 oaks × 4 logs = 32 logs max, rebuilt every episode |
| Ground | Refilled every episode (since 2026-11-17; before, dug holes stayed) |
| Tool | Unbreakable diamond axe, Efficiency X (1 hit per log) |
| Spawn | Arena centre, random facing (15° steps) |
| Observation | RGB image 114×64 |
| Single actions (default) | 11, one per step: noop, forward, back, strafe_left, strafe_right, turn_left, turn_right, look_up, look_down, attack, forward_attack; turns are 15° |
| Multi actions (`--actions multi`) | every step: move (none, forward, back) × turn (-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45°) × look (-20, -10, -5, -2, 0, 2, 5, 10, 20°), attack always held |
| Ticks per step | 1 (default); `--action-repeat N` holds each action for N ticks |
| Reward | +1 per log chopped |
| Episode length | 1000 ticks (or all logs chopped) |
| Random actions | 3.00 logs over 3 episodes (too few to trust) |

## Natural world

| Setting | Value |
| --- | --- |
| World | Normal generated world, fixed seed 20261114 (held-out evaluation world: seed 20261117, never trained in) |
| Evaluation | 100 fixed spots in oak/birch forests (`results/eval_spots_natural.json`, `make_eval_spots.py`), and 100 in every biome with trees (`results/eval_spots_natural_wooded.json`); `evaluate.py --random-spots` for random spawns. Held-out world (archived: [held-out world](../results/archive/held_out.md)): `evaluate.py <model> --env natural --held-out`. Every biome with trees: [Natural world: every biome with trees](../results/natural_all_tree_biomes.md) |
| Spawn | A new spot every episode, on the ground in an oak/birch forest: first within 150 blocks of the last spot, else anywhere within ±25,000. `train.py --spawn wooded` (Line 5, since 2026-11-18): any biome with trees but savanna, and a world-wide search after 4 episodes near one spot |
| Rebuilt each episode | no: chopped trees stay chopped, so every episode moves to fresh forest |
| Boundaries | none; terrain loads as the agent walks |
| Render distance | 4 chunks, Minecraft heap 2G (since 2026-11-15, to fit 4 instances in memory; earlier runs: 6 chunks, 3G) |
| Tool | Unbreakable diamond axe, Efficiency X (1 hit per log) |
| Observation | RGB image 114×64 (same as the arena) |
| Actions | 13 discrete: the arena's 11 plus jump, forward_jump (since 2026-11-15; earlier runs and the random baseline use the arena's 11) |
| Reward | +1 per log chopped (any wood type) |
| Damage | Lava, fire, drowning etc. hurt and dying ends the episode; no fall damage (since 2026-11-16; earlier runs: player immune to all damage) |
| Episode length | 1000 steps (no log cap), or until the agent dies |

## MineRL mode

Models trained in the MineRL project run here with this. The env copies the MineRL project's `MineRLTreechopAxe320-v0` (`~/mineRL/train_treechop.py`), so its models run unchanged.

| Setting | Value |
| --- | --- |
| Turned on by | `--actions minerl`, or automatically for a model with a 64×64 image and `MultiDiscrete([3, 11, 9])` actions (`minerl.is_minerl_model`) |
| Render | 320×180, HUD shown (hand and held axe visible, as in MineRL) |
| Observation | top 138 rows (hotbar, hearts, hunger cut off), `cv2.resize` to 64×64 with `INTER_AREA`: MineRL's `process_pov` |
| Actions | `MultiDiscrete([3, 11, 9])`: move ['stand', 'forward', 'forward_jump'], yaw [-40, -20, -10, -5, -2, 0, 2, 5, 10, 20, 40]° (+ = right), pitch [-20, -10, -5, -2, 0, 2, 5, 10, 20]° (+ = down); attack always held |
| Ticks per decision | 4: keys held every tick, the turn split evenly over them (as MineRL), logs added up |
| Axe | iron (MineRL's) by default; `--axe instant` for our one-hit Efficiency X axe. That one is a diamond axe, so the hand shows a light-blue axe with the enchantment shimmer instead of MineRL's plain iron one: a small visual difference in every one-hit-axe result and fine-tune (not tested on its own) |
| Pop-ups | every recipe unlocked on the first reset, so no "New Recipes Unlocked!" toast covers the view (MineRL patched toasts out; `tools/check_minerl_toasts.py`) |
| Field of view | 70° (Minecraft's default, as MineRL) |
| Episode | `--episode-ticks` (default 1,000; MineRL scored 4,000), `--log-cap` (MineRL: 64), or death |
| Models loaded from | `runs/minerl/` (copies of `~/mineRL/runs/` in WSL); ImpalaCNN via `chop/networks.py`, a copy of the MineRL project's `networks.py` |

**Checked in the game** (`tools/check_minerl_compat.py`, 2026-11-19): every yaw and pitch choice turns the camera the right way by the right amount (within 0.6°: Minecraft turns in 0.15° steps), forward walks on the ground, forward+jump jumps 1.25 blocks, and walking into a tree with attack held chops it with the iron axe.

**Still different from MineRL:** CraftGround runs a newer Minecraft than MineRL's 1.16 (slightly different textures); worlds are peaceful (MineRL had mobs); and the worlds themselves (MineRL: seeds 1-24; here: seed 20261114's fixed spots, or the arena).
<!-- AUTO-GENERATED:END -->
