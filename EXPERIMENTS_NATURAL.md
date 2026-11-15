# CraftGround Experiments: Chop Trees in a Natural World

Goal: chop as many logs as possible in one episode, starting in a real generated
forest instead of the walled arena (natural_env.py). Arena results are in EXPERIMENTS.md.

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-14 17:52._

## Summary

One row per agent. **Mean logs** is the average over all `evaluate.py --env natural` episodes. **Arena mean** is the same agent's score in the walled arena (EXPERIMENTS.md), to show how well it carries over. **vs random** = how many times more logs than random actions in this world.

| Agent | Arena mean | Eval episodes | Mean logs | Min | Max | vs random |
| --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | 3.00 | 100 | **1.81** | 0 | 24 | baseline |
| `chop_ppo_20261113_131702` | 31.67 | 100 | **5.01** | 0 | 30 | **2.8×** |
| `chop_ppo_20261113_135620` | 26.20 | 100 | **5.10** | 0 | 18 | **2.8×** |

## Evaluation runs

One row per `evaluate.py` run, newest first.

| When | Agent | Episodes | Mean logs | Min | Max | Per episode |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-11-14T17:52:51 | `runs/chop_ppo_20261113_131702/model.zip` | 100 | 5.01 | 0 | 30 | 11 30 7 0 2 12 2 21 0 11 3 17 0 5 0 3 0 17 3 16 0 3 2 9 0 6 0 8 0 0 6 0 0 5 4 0 1 5 25 4 7 2 0 4 2 16 13 0 1 2 13 3 3 4 10 0 5 15 4 6 2 9 13 12 8 8 9 4 1 5 0 2 8 3 10 0 4 6 2 8 0 0 8 4 0 0 1 0 0 0 25 0 0 0 0 0 1 0 0 0 |
| 2026-11-14T17:26:55 | `random` | 100 | 1.81 | 0 | 24 | 0 0 2 2 0 1 0 2 7 0 2 3 0 1 0 3 0 3 1 2 1 0 0 0 0 1 2 1 0 5 0 2 2 4 1 0 5 8 1 0 2 0 0 3 4 1 2 0 0 6 0 0 0 0 5 0 3 0 7 0 1 4 0 1 3 2 0 2 0 0 0 3 0 24 0 2 2 0 2 8 1 3 1 0 1 0 0 1 3 2 0 5 3 2 7 0 0 0 0 3 |
| 2026-11-14T17:12:38 | `runs/chop_ppo_20261113_135620/model.zip` | 100 | 5.10 | 0 | 18 | 10 0 4 4 6 8 15 7 3 2 5 7 0 8 4 2 12 2 3 0 9 5 4 7 13 3 9 4 9 4 13 0 5 6 6 6 2 0 5 2 3 7 1 1 7 3 2 0 3 4 0 4 12 2 7 9 4 0 2 7 8 14 11 0 3 12 1 6 12 7 11 15 8 3 1 6 3 6 0 0 0 0 6 18 3 0 0 5 2 2 4 11 9 4 10 6 4 3 2 2 |

## Environment setup

| Setting | Value |
| --- | --- |
| World | Normal generated world, fixed seed 20261114 |
| Spawn | A new spot every episode, on the ground in an oak/birch forest: first within 150 blocks of the last spot, else anywhere within ±25,000 |
| Rebuilt each episode | no: chopped trees stay chopped, so every episode moves to fresh forest |
| Boundaries | none; terrain loads as the agent walks |
| Tool | Unbreakable diamond axe, Efficiency X (1 hit per log) |
| Observation | RGB image 114×64 (same as the arena) |
| Actions | 11 discrete (same as the arena) |
| Reward | +1 per log chopped (any wood type) |
| Episode length | 1000 steps (no log cap) |
<!-- AUTO-GENERATED:END -->

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
