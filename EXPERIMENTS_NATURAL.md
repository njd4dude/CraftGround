# CraftGround Experiments: Chop Trees in a Natural World

Goal: chop as many logs as possible in one episode, starting in a real generated
forest instead of the walled arena (natural_env.py). Arena results are in EXPERIMENTS.md.

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-15 23:50._

## Summary

One row per agent. **Trained in** is the world it was trained in (see Agents below for models resumed from an arena run). **Mean logs** is the average over all `evaluate.py --env natural` episodes. **Arena mean** is the same agent's score in the walled arena (EXPERIMENTS.md), to show how well it carries over. **vs random** = how many times more logs than random actions in this world.

| Agent | Trained in | Arena mean | Eval episodes | Mean logs | Min | Max | vs random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | – | 3.00 | 100 | **1.81** | 0 | 24 | baseline |
| `chop_ppo_20261113_131702` | arena | 31.67 | 100 | **5.01** | 0 | 30 | **2.8×** |
| `chop_ppo_20261113_135620` | arena | 26.20 | 100 | **5.10** | 0 | 18 | **2.8×** |
| `chop_ppo_natural_20261115_155106` | natural | – | _not evaluated: `run.bat evaluate.py runs/chop_ppo_natural_20261115_155106/model.zip --env natural`_ |  |  |  |  |
| `chop_ppo_natural_20261115_161832` | natural | – | 100 | **9.68** | 0 | 43 | **5.3×** |
| `chop_ppo_natural_20261115_173930` | natural | – | 100 | **17.96** | 0 | 63 | **9.9×** |
| `chop_ppo_natural_20261115_185527` @ 601,536 steps | natural | – | 100 | **21.67** | 0 | 66 | **12.0×** |
| `chop_ppo_natural_20261115_192015` | natural | – | 100 | **14.99** | 0 | 41 | **8.3×** |
| `chop_ppo_natural_20261115_231007` | natural | – | 100 | **25.44** | 0 | 66 | **14.1×** |

## Agents

One section per trained agent, newest first. Hyperparameters changed from the defaults in `train.py` are in bold. Environment changes (reward, episode length, trees...) are flagged on the first run that used them. Add notes with `train.py --notes "..."`, or edit `notes` in the run's `config.json` and rerun `python experiments.py`.

### `chop_ppo_natural_20261115_231007`

- **Trained:** 1,002,552 steps of 400,000 planned, 399 episodes, 29.7 min, 3 parallel envs (completed)
- **Resumed from:** `runs/chop_ppo_natural_20261115_185527/checkpoints/ppo_chop_601536_steps.zip`
- **Result:** 25.44 logs/episode over 100 eval episodes (training last-20 avg 28.9)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** nothing, default settings and same environment as the previous run
- **Notes:** 600k -> 1M with linear LR decay (2.5e-4 -> 0), 3 instances, render distance 4, 2G heap; compare with the constant-LR 1M model (15.0)

### `chop_ppo_natural_20261115_192015`

- **Trained:** 1,003,072 steps of 320,000 planned, 320 episodes, 21.8 min, 4 parallel envs (completed)
- **Resumed from:** `runs/chop_ppo_natural_20261115_185527/checkpoints/ppo_chop_681536_steps.zip`
- **Result:** 14.99 logs/episode over 100 eval episodes (training last-20 avg 14.15)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** nothing, default settings and same environment as the previous run
- **Notes:** resume jump model from 681k checkpoint (killed for low memory) to 1M, with memory logging

### `chop_ppo_natural_20261115_173930`

- **Trained:** 501,536 steps of 320,000 planned, 320 episodes, 21.6 min, 4 parallel envs (completed)
- **Resumed from:** `runs/chop_ppo_natural_20261115_172419/checkpoints/ppo_chop_180000_steps.zip`
- **Result:** 17.96 logs/episode over 100 eval episodes (training last-20 avg 22.4)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env actions: ['noop', 'forward', 'back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'look_up', 'look_down', 'attack', 'forward_attack'] → **['noop', 'forward', 'back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'look_up', 'look_down', 'attack', 'forward_attack', 'jump', 'forward_jump']** (changed since previous run)
- **Notes:** resume jump run chop_ppo_natural_20261115_172419 from 180k checkpoint (killed for low memory) to 500k

### `chop_ppo_natural_20261115_161832`

- **Trained:** 502,814 steps of 300,000 planned, 300 episodes, 19.0 min, 4 parallel envs (completed)
- **Resumed from:** `runs/chop_ppo_natural_20261115_160512/model.zip`
- **Result:** 9.68 logs/episode over 100 eval episodes (training last-20 avg 9.15)
- **PPO settings:** n_steps=2046, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** nothing, default settings and same environment as the previous run
- **Notes:** continue chop_ppo_natural_20261115_160512 (stopped at ~200k, 6 instances) to 500k on 4 instances

### `chop_ppo_natural_20261115_155106`

- **Trained:** 100,352 steps of 100,000 planned, 100 episodes, 9.8 min, 4 parallel envs (completed)
- **Result:** not evaluated yet (training last-20 avg 4.05)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** first recorded run, default settings (the baseline)
- **Notes:** first natural-world run from scratch, 100k steps

## Training runs

One row per `train.py` run, newest first.

| Started | Steps | Episodes | Minutes | Last-20 avg logs | Best episode | Status | Saved model | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-15T23:10:07 | 1002552 | 399 | 29.7 | 28.9 | 73 | completed | `runs/chop_ppo_natural_20261115_231007/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/f61norjd) |
| 2026-11-15T19:20:15 | 1003072 | 320 | 21.8 | 14.15 | 57 | completed | `runs/chop_ppo_natural_20261115_192015/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/12zz1rvp) |
| 2026-11-15T17:39:30 | 501536 | 320 | 21.6 | 22.4 | 53 | completed | `runs/chop_ppo_natural_20261115_173930/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/wbto0vn4) |
| 2026-11-15T16:18:32 | 502814 | 300 | 19.0 | 9.15 | 44 | completed | `runs/chop_ppo_natural_20261115_161832/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/6hob06tk) |
| 2026-11-15T15:51:06 | 100352 | 100 | 9.8 | 4.05 | 30 | completed | `runs/chop_ppo_natural_20261115_155106/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/jnm6lysy) |

## Evaluation runs

One row per `evaluate.py` run, newest first. **Minutes** includes launching Minecraft; **Steps/s** is env steps per second over all instances while playing (runs from before timing was added show –).

| When | Agent | Episodes | Mean logs | Min | Max | Instances | Minutes | Steps/s | Per episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-15T23:50:28 | `runs/chop_ppo_natural_20261115_231007/model.zip` | 100 | 25.44 | 0 | 66 | 3 | 9.8 | 251 | 37 12 37 37 26 14 15 28 46 13 11 0 3 18 42 23 26 46 19 2 0 19 3 29 56 26 35 2 37 7 47 42 8 11 33 12 31 32 4 6 5 60 46 45 50 30 52 62 44 29 0 35 0 33 35 47 0 52 11 35 0 30 45 28 8 42 27 43 49 24 3 50 23 10 36 8 15 47 13 2 43 13 21 66 0 20 64 7 5 65 11 17 63 13 4 34 15 0 4 10 |
| 2026-11-15T20:02:25 | `runs/chop_ppo_natural_20261115_192015/model.zip` | 100 | 14.99 | 0 | 41 | 4 | 8.5 | 286 | 24 10 6 4 0 0 11 22 0 18 11 10 20 15 15 17 35 21 10 19 6 0 17 0 2 3 24 19 31 11 0 28 8 20 19 4 18 14 9 14 18 0 0 0 32 14 13 18 34 19 12 20 17 5 0 7 19 17 14 8 1 29 15 14 32 23 27 7 29 21 12 0 37 10 0 30 8 18 21 23 0 24 9 33 18 28 11 24 22 41 4 28 34 24 7 0 22 11 15 5 |
| 2026-11-15T19:53:35 | `runs/chop_ppo_natural_20261115_185527/checkpoints/ppo_chop_601536_steps.zip` | 100 | 21.67 | 0 | 66 | 4 | 8.7 | 268 | 51 0 6 26 52 21 16 37 4 12 51 3 27 33 33 3 37 6 28 27 40 4 19 11 38 3 34 10 13 0 66 26 50 7 24 38 19 8 21 5 65 20 14 0 50 7 56 1 33 36 23 11 19 16 30 13 29 18 17 20 33 2 22 0 26 0 5 13 27 24 36 0 38 0 18 0 28 38 54 0 10 35 28 3 16 21 62 0 10 49 41 17 32 49 10 12 8 0 13 0 |
| 2026-11-15T18:54:25 | `runs/chop_ppo_natural_20261115_173930/model.zip` | 100 | 17.96 | 0 | 63 | 4 | 8.2 | 298 | 0 2 1 19 32 41 33 0 25 30 15 0 4 9 26 11 43 9 14 20 19 0 23 22 25 31 28 10 23 0 0 0 0 27 16 1 5 31 27 25 20 20 11 10 14 26 16 1 13 35 8 0 38 0 0 4 13 3 52 23 5 9 24 27 25 40 31 0 0 29 0 22 26 25 17 26 10 38 11 3 0 29 10 2 12 48 26 0 11 30 30 17 36 13 63 21 17 41 52 16 |
| 2026-11-15T16:51:49 | `runs/chop_ppo_natural_20261115_161832/model.zip` | 100 | 9.68 | 0 | 43 | 4 | 8.1 | 300 | 4 5 0 11 0 4 13 0 2 4 3 0 10 31 0 2 10 22 5 9 9 11 4 3 11 4 10 26 7 26 12 0 15 0 27 11 2 3 18 28 6 33 10 3 6 9 9 0 0 33 1 18 5 7 12 16 12 14 7 8 3 4 0 0 0 2 0 4 16 29 8 2 34 20 3 3 0 22 3 43 2 7 15 0 6 15 42 3 2 26 6 10 2 8 8 10 21 7 5 6 |
| 2026-11-14T17:52:51 | `runs/chop_ppo_20261113_131702/model.zip` | 100 | 5.01 | 0 | 30 | – | – | – | 11 30 7 0 2 12 2 21 0 11 3 17 0 5 0 3 0 17 3 16 0 3 2 9 0 6 0 8 0 0 6 0 0 5 4 0 1 5 25 4 7 2 0 4 2 16 13 0 1 2 13 3 3 4 10 0 5 15 4 6 2 9 13 12 8 8 9 4 1 5 0 2 8 3 10 0 4 6 2 8 0 0 8 4 0 0 1 0 0 0 25 0 0 0 0 0 1 0 0 0 |
| 2026-11-14T17:26:55 | `random` | 100 | 1.81 | 0 | 24 | – | – | – | 0 0 2 2 0 1 0 2 7 0 2 3 0 1 0 3 0 3 1 2 1 0 0 0 0 1 2 1 0 5 0 2 2 4 1 0 5 8 1 0 2 0 0 3 4 1 2 0 0 6 0 0 0 0 5 0 3 0 7 0 1 4 0 1 3 2 0 2 0 0 0 3 0 24 0 2 2 0 2 8 1 3 1 0 1 0 0 1 3 2 0 5 3 2 7 0 0 0 0 3 |
| 2026-11-14T17:12:38 | `runs/chop_ppo_20261113_135620/model.zip` | 100 | 5.10 | 0 | 18 | – | – | – | 10 0 4 4 6 8 15 7 3 2 5 7 0 8 4 2 12 2 3 0 9 5 4 7 13 3 9 4 9 4 13 0 5 6 6 6 2 0 5 2 3 7 1 1 7 3 2 0 3 4 0 4 12 2 7 9 4 0 2 7 8 14 11 0 3 12 1 6 12 7 11 15 8 3 1 6 3 6 0 0 0 0 6 18 3 0 0 5 2 2 4 11 9 4 10 6 4 3 2 2 |

## Environment setup

| Setting | Value |
| --- | --- |
| World | Normal generated world, fixed seed 20261114 |
| Spawn | A new spot every episode, on the ground in an oak/birch forest: first within 150 blocks of the last spot, else anywhere within ±25,000 |
| Rebuilt each episode | no: chopped trees stay chopped, so every episode moves to fresh forest |
| Boundaries | none; terrain loads as the agent walks |
| Render distance | 4 chunks, Minecraft heap 2G (since 2026-11-15, to fit 4 instances in memory; earlier runs: 6 chunks, 3G) |
| Tool | Unbreakable diamond axe, Efficiency X (1 hit per log) |
| Observation | RGB image 114×64 (same as the arena) |
| Actions | 13 discrete: the arena's 11 plus jump, forward_jump (since 2026-11-15; earlier runs and the random baseline use the arena's 11) |
| Reward | +1 per log chopped (any wood type) |
| Episode length | 1000 steps (no log cap) |
<!-- AUTO-GENERATED:END -->

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
