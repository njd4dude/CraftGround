# CraftGround Experiments: Chop Trees

Goal: train an agent to chop as many oak logs as possible in one episode.

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-13 13:28._

## Summary

All evaluation episodes pooled per agent, best first.

| Agent | Episodes | Mean logs | Min | Max | vs random |
| --- | --- | --- | --- | --- | --- |
| `random` | 3 | 3.00 | 0 | 5 | baseline |

## Training runs

One row per `train.py` run, newest first.

| Started | Steps | Episodes | Minutes | Last-20 avg logs | Best episode | Status | Saved model | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-13T13:17:02 | 200704 | 206 | 11.1 | 31.40 | 32 | completed | `runs/chop_ppo_20261113_131702/model.zip` | – |

## Evaluation runs

One row per `evaluate.py` run, newest first.

| When | Agent | Episodes | Mean logs | Min | Max | Per episode |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-13T13:12:10 | `random` | 3 | 3.00 | 0 | 5 | 5 0 4 |

## Environment setup

| Setting | Value |
| --- | --- |
| World | Superflat arena, stone-brick walls at ±13 blocks |
| Trees | 8 oaks × 4 logs = 32 logs max |
| Rebuilt each episode | yes (trees regrow on reset) |
| Tool | Unbreakable diamond axe, Efficiency X (1 hit per log) |
| Spawn | Arena centre, random facing (15° steps) |
| Observation | RGB image 114×64 |
| Actions | 11 discrete: noop, forward, back, strafe_left, strafe_right, turn_left, turn_right, look_up, look_down, attack, forward_attack |
| Reward | +1 per log chopped |
| Episode length | 1000 steps (or all logs chopped) |
<!-- AUTO-GENERATED:END -->

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.

- **2026-11-13:** `check_env.py` passed. A scripted agent chopped 3 logs per episode and the trees regrew after reset. Rewards sometimes show up 1–2 steps after the attack that earned them.
- **2026-11-13:** The random baseline is only 3 episodes (5, 0, 4), which is too few to trust. Rerun with `run.bat evaluate.py --random --episodes 20` before comparing.
- **2026-11-13:** Speed: training runs at about 300 steps/s, and 200k steps took 11 minutes. (The earlier "60 steps/s" estimate was wrong because Minecraft's startup time dominated that 3-episode run.)
- **2026-11-13, run 1 (`chop_ppo_20261113_131702`):** PPO learned fast. Logs per episode went 3.5 → 18 → 25 → 28 over the first 70 episodes and finished at a last-20 average of 31.4/32. The task may now be too easy, so next ideas are harder layouts, more trees or random tree positions. Still needs a proper 20-episode evaluation against a 20-episode random baseline.
