# CraftGround Experiments: Chop Trees

Goal: train an agent to chop as many oak logs as possible in one episode.

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-17 21:38._

## Summary

One row per agent. **Mean logs** is the real score: the average over all `evaluate.py` episodes for that agent (max 32). **Training avg** is the last-20-episode average seen during training. It's a useful hint, but the agent was still exploring then, so evaluate before trusting it. **vs random** = how many times more logs than random actions.

| Agent | Training avg | Eval episodes | Mean logs | Min | Max | vs random |
| --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | – | 3 | **3.00** | 0 | 5 | baseline |
| `fixed_single_1tick_r1_20261117_212824` | 30.25 | 50 | **30.60** | 28 | 32 | **10.2×** |
| `multi_1tick_seed2_20261117_191858` | 29.6 | _not evaluated: `run.bat evaluate.py runs/arena/multi_1tick_seed2_20261117_191858/model.zip --episodes 20`_ |  |  |  |  |
| `single_1tick_seed2_20261117_190806` | 30.65 | 50 | **24.20** | 2 | 32 | **8.1×** |
| `multi_4tick_20261117_184506` | 21.65 | 50 | **27.68** | 24 | 31 | **9.2×** |
| `multi_1tick_20261117_183147` | 24.85 | 50 | **30.80** | 28 | 32 | **10.3×** |
| `single_4tick_seed2_20261117_181309` | 23.5 | 50 | **23.60** | 20 | 25 | **7.9×** |
| `single_4tick_20261117_173652` | 22.9 | 50 | **23.04** | 20 | 24 | **7.7×** |
| `single_1tick_20261117_172920` | 26.8 | 50 | **28.56** | 24 | 32 | **9.5×** |
| `chop_ppo_20261113_135620` | 24.85 | 429 | **26.20** | 18 | 31 | **8.7×** |
| `chop_ppo_20261113_131702` | 31.40 | 24 | **31.67** | 31 | 32 | **10.6×** |
| `runs/arena/single_4tick_20261117_173652/checkpoints/ppo_chop_39996_steps.zip` | – | 50 | **21.66** | 15 | 25 | **7.2×** |
| `runs/arena/single_4tick_20261117_173652/checkpoints/ppo_chop_59994_steps.zip` | – | 50 | **22.94** | 18 | 24 | **7.6×** |

## Agents

One section per trained agent, newest first. Hyperparameters changed from the defaults in `train.py` are in bold. Environment changes (reward, episode length, trees...) are flagged on the first run that used them. Add notes with `train.py --notes "..."`, or edit `notes` in the run's `config.json` and rerun `python experiments.py`.

### `fixed_single_1tick_r1_20261117_212824`

- **Trained:** 200,508 steps of 200,000 planned, 201 episodes, 6.7 min, 3 parallel envs (completed)
- **Result:** 30.60 logs/episode over 50 eval episodes (training last-20 avg 30.25)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env actions: {'move': ['none', 'forward', 'back'], 'turn_degrees': [-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45], 'look_degrees': [-20, -10, -5, -2, 0, 2, 5, 10, 20], 'attack': 'always held'} → **['noop', 'forward', 'back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'look_up', 'look_down', 'attack', 'forward_attack']** (changed since previous run)
- **Notes:** Fixed arena (ground repaired): setup A, single actions, 1 tick, run 1

### `multi_1tick_seed2_20261117_191858`

- **Trained:** 200,508 steps of 200,000 planned, 200 episodes, 9.2 min, 3 parallel envs (completed)
- **Result:** not evaluated yet (training last-20 avg 29.6)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env actions: ['noop', 'forward', 'back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'look_up', 'look_down', 'attack', 'forward_attack'] → **{'move': ['none', 'forward', 'back'], 'turn_degrees': [-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45], 'look_degrees': [-20, -10, -5, -2, 0, 2, 5, 10, 20], 'attack': 'always held'}** (changed since previous run)
- **Notes:** Seed comparison: run C setup (multi actions, 1 tick), second run

### `single_1tick_seed2_20261117_190806`

- **Trained:** 200,508 steps of 200,000 planned, 203 episodes, 6.9 min, 3 parallel envs (completed)
- **Result:** 24.20 logs/episode over 50 eval episodes (training last-20 avg 30.65)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env actions: {'move': ['none', 'forward', 'back'], 'turn_degrees': [-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45], 'look_degrees': [-20, -10, -5, -2, 0, 2, 5, 10, 20], 'attack': 'always held'} → **['noop', 'forward', 'back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'look_up', 'look_down', 'attack', 'forward_attack']** (changed since previous run); env action_repeat: 4 → **1** (changed since previous run)
- **Notes:** Seed comparison: run A setup (single actions, 1 tick), second run

### `multi_4tick_20261117_184506`

- **Trained:** 200,508 steps of 200,000 planned, 801 episodes, 18.6 min, 3 parallel envs (completed)
- **Result:** 27.68 logs/episode over 50 eval episodes (training last-20 avg 21.65)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env action_repeat: 1 → **4** (changed since previous run)
- **Notes:** Repeat comparison: multi actions, 4 ticks per step

### `multi_1tick_20261117_183147`

- **Trained:** 200,508 steps of 200,000 planned, 200 episodes, 9.1 min, 3 parallel envs (completed)
- **Result:** 30.80 logs/episode over 50 eval episodes (training last-20 avg 24.85)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env actions: ['noop', 'forward', 'back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'look_up', 'look_down', 'attack', 'forward_attack'] → **{'move': ['none', 'forward', 'back'], 'turn_degrees': [-45, -20, -10, -5, -2, 0, 2, 5, 10, 20, 45], 'look_degrees': [-20, -10, -5, -2, 0, 2, 5, 10, 20], 'attack': 'always held'}** (changed since previous run); env action_repeat: 4 → **1** (changed since previous run)
- **Notes:** Repeat comparison: multi actions, 1 tick per step

### `single_4tick_seed2_20261117_181309`

- **Trained:** 200,508 steps of 200,000 planned, 801 episodes, 14.9 min, 3 parallel envs (completed)
- **Result:** 23.60 logs/episode over 50 eval episodes (training last-20 avg 23.5)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** nothing, default settings and same environment as the previous run
- **Notes:** Repeat comparison: run B again with a new random seed

### `single_4tick_20261117_173652`

- **Trained:** 200,508 steps of 200,000 planned, 801 episodes, 15.2 min, 3 parallel envs (completed)
- **Result:** 23.04 logs/episode over 50 eval episodes (training last-20 avg 22.9)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** env action_repeat: 1 → **4** (changed since previous run)
- **Notes:** Repeat comparison: 4 ticks per step (same 200k decisions = 800k ticks)

### `single_1tick_20261117_172920`

- **Trained:** 200,508 steps of 200,000 planned, 198 episodes, 6.9 min, 3 parallel envs (completed)
- **Result:** 28.56 logs/episode over 50 eval episodes (training last-20 avg 26.8)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** nothing, default settings and same environment as the previous run
- **Notes:** Repeat comparison: baseline, 1 tick per step

### `chop_ppo_20261113_135620`

- **Trained:** 51,200 steps of 50,000 planned, 51 episodes, 4.3 min (completed)
- **Result:** 26.20 logs/episode over 429 eval episodes (training last-20 avg 24.85)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** nothing, default settings and same environment as the previous run
- **Notes:** Short 50k-step run, first one logged to W&B (recorded after the fact).

### `chop_ppo_20261113_131702`

- **Trained:** 200,704 steps of 200,000 planned, 206 episodes, 11.1 min (completed)
- **Result:** 31.67 logs/episode over 24 eval episodes (training last-20 avg 31.40)
- **PPO settings:** n_steps=2048, batch_size=256, n_epochs=4, learning_rate=0.00025, gamma=0.99, ent_coef=0.01, clip_range=0.2
- **What's special:** first recorded run, default settings (the baseline)
- **Notes:** First run (recorded after the fact).

## Training runs

One row per `train.py` run, newest first.

| Started | Steps | Episodes | Minutes | Last-20 avg logs | Best episode | Status | Saved model | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-17T21:28:24 | 200508 | 201 | 6.7 | 30.25 | 32 | completed | `runs/arena/fixed_single_1tick_r1_20261117_212824/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/x7qpqe0q) |
| 2026-11-17T19:18:58 | 200508 | 200 | 9.2 | 29.6 | 32 | completed | `runs/arena/multi_1tick_seed2_20261117_191858/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/mujfej23) |
| 2026-11-17T19:08:06 | 200508 | 203 | 6.9 | 30.65 | 32 | completed | `runs/arena/single_1tick_seed2_20261117_190806/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/g2h16i1m) |
| 2026-11-17T18:45:06 | 200508 | 801 | 18.6 | 21.65 | 32 | completed | `runs/arena/multi_4tick_20261117_184506/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/aamm66rk) |
| 2026-11-17T18:31:47 | 200508 | 200 | 9.1 | 24.85 | 32 | completed | `runs/arena/multi_1tick_20261117_183147/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/vb3oj5uf) |
| 2026-11-17T18:13:09 | 200508 | 801 | 14.9 | 23.5 | 26 | completed | `runs/arena/single_4tick_seed2_20261117_181309/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/b2am6n1k) |
| 2026-11-17T17:36:52 | 200508 | 801 | 15.2 | 22.9 | 25 | completed | `runs/arena/single_4tick_20261117_173652/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/p84qqsaz) |
| 2026-11-17T17:29:20 | 200508 | 198 | 6.9 | 26.8 | 31 | completed | `runs/arena/single_1tick_20261117_172920/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/x8bvlnzz) |
| 2026-11-13T13:56:20 | 51200 | 51 | 4.3 | 24.85 | 28 | completed | `runs/arena/chop_ppo_20261113_135620/model.zip` | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/9te3935j) |
| 2026-11-13T13:17:02 | 200704 | 206 | 11.1 | 31.40 | 32 | completed | `runs/arena/chop_ppo_20261113_131702/model.zip` | – |

## Evaluation runs

One row per `evaluate.py` run, newest first. **Minutes** includes launching Minecraft; **Steps/s** is env steps per second over all instances while playing (runs from before timing was added show –).

| When | Agent | Episodes | Mean logs | Min | Max | Instances | Minutes | Steps/s | Per episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-17T21:38:48 | `runs/arena/fixed_single_1tick_r1_20261117_212824/model.zip` | 50 | 30.60 | 28 | 32 | 3 | 3.1 | 585 | 30 31 29 31 32 31 30 31 30 31 32 32 32 30 31 29 31 32 28 29 32 31 31 31 32 31 31 31 31 29 29 30 30 29 29 31 29 32 32 29 30 31 32 32 32 30 32 30 29 30 |
| 2026-11-17T19:18:48 | `runs/arena/single_1tick_seed2_20261117_190806/model.zip` | 50 | 24.20 | 2 | 32 | 3 | 3.2 | 610 | 26 30 29 5 31 31 11 31 28 17 29 31 11 31 30 6 31 31 24 31 30 6 32 30 32 6 29 32 2 32 29 29 6 30 32 13 32 32 12 32 32 31 14 30 30 4 30 32 12 23 |
| 2026-11-17T19:07:24 | `runs/arena/multi_4tick_20261117_184506/model.zip` | 50 | 27.68 | 24 | 31 | 3 | 3.0 | 175 | 28 26 29 25 28 25 26 30 27 31 26 28 29 29 29 28 30 30 27 30 27 26 27 25 28 27 28 29 28 29 29 29 28 29 30 25 27 26 26 27 29 29 26 28 24 27 25 29 28 28 |
| 2026-11-17T18:44:54 | `runs/arena/multi_1tick_20261117_183147/model.zip` | 50 | 30.80 | 28 | 32 | 3 | 3.4 | 543 | 31 31 31 31 30 29 30 30 32 32 32 29 32 32 32 30 31 29 31 32 31 30 30 31 32 31 32 32 28 32 31 31 30 32 32 31 28 32 30 30 32 31 30 30 32 31 29 32 32 28 |
| 2026-11-17T18:31:40 | `runs/arena/single_4tick_seed2_20261117_181309/model.zip` | 50 | 23.60 | 20 | 25 | 3 | 3.0 | 196 | 24 24 23 22 23 23 24 24 24 24 24 24 24 24 22 24 23 23 24 25 24 23 24 24 24 24 20 24 24 24 23 22 24 24 24 24 24 23 24 22 24 24 24 24 24 23 24 24 24 24 |
| 2026-11-17T18:05:56 | `runs/arena/single_4tick_20261117_173652/checkpoints/ppo_chop_59994_steps.zip` | 50 | 22.94 | 18 | 24 | 3 | 2.8 | 198 | 24 22 23 24 22 24 18 23 23 20 24 23 24 20 24 24 23 24 24 24 24 24 24 23 23 23 23 22 21 24 23 24 20 24 22 20 24 23 23 22 23 24 23 24 22 23 24 23 24 24 |
| 2026-11-17T18:02:29 | `runs/arena/single_4tick_20261117_173652/checkpoints/ppo_chop_39996_steps.zip` | 50 | 21.66 | 15 | 25 | 3 | 2.9 | 203 | 24 21 18 21 19 23 22 22 20 25 23 18 23 19 20 22 23 22 23 20 23 23 22 23 24 22 23 24 21 22 22 23 21 21 23 23 23 25 23 18 22 21 21 21 21 21 20 22 15 22 |
| 2026-11-17T17:59:26 | `runs/arena/single_4tick_20261117_173652/model.zip` | 50 | 23.04 | 20 | 24 | 3 | 2.8 | 195 | 20 24 22 24 24 23 23 22 23 23 21 24 23 24 23 24 24 24 22 22 24 22 23 22 23 23 24 22 21 24 24 23 23 24 24 23 23 22 24 24 23 24 24 23 24 22 21 24 23 23 |
| 2026-11-17T17:56:24 | `runs/arena/single_1tick_20261117_172920/model.zip` | 50 | 28.56 | 24 | 32 | 3 | 3.3 | 610 | 29 27 27 29 29 30 27 27 31 29 32 30 26 28 29 32 29 24 29 30 29 28 28 30 30 30 31 27 24 31 29 28 28 27 27 29 28 29 28 29 29 28 30 29 28 27 31 28 27 27 |
| 2026-11-14T14:56:13 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 26.53 | 23 | 30 | – | – | – | 28 28 24 25 26 24 26 26 28 24 26 28 28 26 26 27 29 26 24 27 27 27 27 28 26 26 27 26 25 27 25 27 25 23 29 29 26 27 26 28 27 23 25 25 28 25 28 29 26 28 27 27 28 28 27 23 26 28 27 28 25 28 27 28 26 28 26 25 27 28 30 24 23 25 29 25 29 27 25 28 24 28 27 26 24 27 27 30 27 24 28 29 29 27 27 26 25 26 26 23 |
| 2026-11-14T14:49:56 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 26.67 | 23 | 31 | – | – | – | 26 29 24 28 24 27 27 23 26 28 27 25 28 26 29 24 24 25 26 27 26 27 25 25 28 26 24 25 27 28 24 27 27 25 27 26 26 29 30 23 26 27 26 26 30 28 27 29 26 26 27 27 25 26 28 27 25 26 27 27 28 27 29 27 28 26 27 23 29 24 30 25 24 27 31 28 27 26 28 26 28 29 31 27 29 24 25 29 25 26 27 28 25 26 29 30 27 25 28 25 |
| 2026-11-14T14:44:44 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 25.60 | 18 | 30 | – | – | – | 26 28 27 26 26 23 26 26 26 22 24 22 27 26 26 28 28 25 27 25 26 26 24 26 27 23 24 26 25 23 24 18 22 26 26 24 25 25 27 27 25 25 28 25 26 24 23 26 24 28 25 27 26 23 29 23 26 25 28 24 30 24 25 29 23 25 28 26 26 26 28 26 28 25 26 27 25 25 25 27 27 25 28 27 26 27 24 26 26 24 26 26 24 30 22 24 28 25 26 27 |
| 2026-11-14T14:21:57 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 26.00 | 21 | 30 | – | – | – | 25 27 28 27 26 30 27 26 25 25 26 25 28 26 27 26 25 27 25 26 28 26 29 24 25 30 28 27 28 24 26 27 27 27 25 27 28 26 22 25 27 24 25 25 25 24 30 29 26 24 25 27 24 29 27 26 23 26 26 28 25 24 24 26 30 25 21 25 23 24 25 28 26 25 26 26 27 27 30 28 26 25 26 26 28 27 24 24 25 23 24 25 25 25 25 27 25 26 24 29 |
| 2026-11-14T13:52:20 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 5 | 25.00 | 23 | 26 | – | – | – | 25 23 26 26 25 |
| 2026-11-14T13:36:22 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 2 | 23.00 | 22 | 24 | – | – | – | 22 24 |
| 2026-11-14T13:34:49 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 2 | 26.00 | 26 | 26 | – | – | – | 26 26 |
| 2026-11-13T14:13:19 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 20 | 26.85 | 23 | 30 | – | – | – | 27 29 28 24 27 26 27 26 26 23 30 28 25 25 25 27 28 29 30 27 |
| 2026-11-13T13:52:31 | `runs/arena/chop_ppo_20261113_131702/model.zip` | 2 | 32.00 | 32 | 32 | – | – | – | 32 32 |
| 2026-11-13T13:50:12 | `runs/arena/chop_ppo_20261113_131702/model.zip` | 2 | 32.00 | 32 | 32 | – | – | – | 32 32 |
| 2026-11-13T13:46:35 | `runs/arena/chop_ppo_20261113_131702/model.zip` | 20 | 31.60 | 31 | 32 | – | – | – | 31 31 31 32 31 31 32 32 32 32 31 31 32 32 32 32 32 32 32 31 |
| 2026-11-13T13:12:10 | `random` | 3 | 3.00 | 0 | 5 | – | – | – | 5 0 4 |

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
- **2026-11-13:** Parallel envs: `train.py` now runs 6 Minecraft instances by default (`--n-envs`). Steady training speed went from ~300 to ~740 steps/s (2.5×). Startup takes ~2.5 min instead of ~75 s. Raw env benchmark (random actions): 1 env 451, 2 → 714, 4 → 858, 8 → 1076 steps/s, with 8 using 92% CPU and 17 GB RAM. The CPU is the limit; the GPU stays under 40%. `--n-steps` is now the total across all envs, so each update is still 2048 steps (2046 with 6 envs).
