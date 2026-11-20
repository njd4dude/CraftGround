# Arena: Setup

[← Arena index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 17:04._

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
<!-- AUTO-GENERATED:END -->
