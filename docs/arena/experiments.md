# Arena: Experiments

[← Arena index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 17:04._

**Eval** is the mean over `evaluate.py` episodes (max 32), ± its 95% range. **Ground fixed** = no: holes the agent dug stayed for the rest of the run (fixed 2026-11-17), so those scores are rough. Edit the questions and findings in `results/experiments_arena.json`.

<a id="arena-exp-1"></a>

### 1 · First runs

**Question:** Can PPO with the default CNN learn to chop the arena's 32 logs at all?

**Found:** Yes, fast: 31.7 of 32 at 200k steps, 26.2 after a 50k run. The arena is close to solved, so later experiments compare setups by small differences near the cap.

| Run | Actions | Ticks/step | Ground fixed | Steps | Train min | Eval mean ± 95% (episodes) | Min | Max | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `chop_ppo_20261113_131702` | single | 1 | **no** | 200k | 11.1 | **31.67** ± 0.19 (24) | 31 | 32 | – |
| `chop_ppo_20261113_135620` | single | 1 | **no** | 51k | 4.3 | **26.20** ± 0.17 (429) | 18 | 31 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/9te3935j) |

<a id="arena-exp-2"></a>

### 2 · Action repeat: 1 vs 4 ticks per step (before the ground fix: preliminary)

**Question:** CraftGround shows an action's effects 1-3 ticks late (tools/check_timing.py). Does holding each action for 4 ticks, so each step shows its own result, learn better? Tried with single and multi actions.

**Found:** Not valid as a comparison: these ran before the ground fix, so holes the agents dug stayed for the rest of the run, and that hurt agents unevenly (single_1tick_seed2's 24.2 is mostly agents stuck in their own pits). What looks real: 4-tick single agents leave the bottom log of every tree (24 = 8 trees x 3), and their scores are tight, so pits don't explain them. Rerun on the fixed arena: experiment 4.

| Run | Actions | Ticks/step | Ground fixed | Steps | Train min | Eval mean ± 95% (episodes) | Min | Max | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single_1tick_20261117_172920` | single | 1 | **no** | 200k | 6.9 | **28.56** ± 0.47 (50) | 24 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/x8bvlnzz) |
| `single_1tick_seed2_20261117_190806` | single | 1 | **no** | 200k | 6.9 | **24.20** ± 2.79 (50) | 2 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/g2h16i1m) |
| `single_4tick_20261117_173652` | single | 4 | **no** | 200k | 15.2 | **23.04** ± 0.28 (50) | 20 | 24 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/p84qqsaz) |
| `single_4tick_20261117_173652` @ 40k (checkpoint) | single | 4 | **no** | 40k | – | **21.66** ± 0.53 (50) | 15 | 25 | – |
| `single_4tick_20261117_173652` @ 60k (checkpoint) | single | 4 | **no** | 60k | – | **22.94** ± 0.38 (50) | 18 | 24 | – |
| `single_4tick_seed2_20261117_181309` | single | 4 | **no** | 200k | 14.9 | **23.60** ± 0.23 (50) | 20 | 25 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/b2am6n1k) |
| `multi_1tick_20261117_183147` | multi | 1 | **no** | 200k | 9.1 | **30.80** ± 0.33 (50) | 28 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/vb3oj5uf) |
| `multi_1tick_seed2_20261117_191858` | multi | 1 | **no** | 200k | 9.2 | _not evaluated_ |  |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/mujfej23) |
| `multi_4tick_20261117_184506` | multi | 4 | **no** | 200k | 18.6 | **27.68** ± 0.46 (50) | 24 | 31 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/aamm66rk) |

| Setup | Runs | Average of runs | Each run |
| --- | --- | --- | --- |
| single, 1 tick, ground not fixed, 200k | 2 | **26.38** | 28.6 24.2 |
| single, 4 ticks, ground not fixed, 200k | 2 | **23.32** | 23.0 23.6 |
| multi, 1 tick, ground not fixed, 200k | 1 | **30.80** | 30.8 |
| multi, 4 ticks, ground not fixed, 200k | 1 | **27.68** | 27.7 |

<a id="arena-exp-3"></a>

### 3 · Single vs multi actions (ground fixed)

**Question:** Multi actions choose move, turn and look every step with attack always held, and turn by exact degrees. Do they beat the 11 single actions? 3 runs each, 1 tick, after the ground fix.

**Found:** Multi slightly ahead, not proven: 31.3 vs 30.5 average, and multi's worst episodes are better, but the groups overlap (permutation test p = 0.2). Both are near the 32 cap, so the arena can't separate them; in the natural world multi won clearly: 26.6 vs 22.4 for the same recipe (docs/natural/lines.md, Line 4).

| Run | Actions | Ticks/step | Ground fixed | Steps | Train min | Eval mean ± 95% (episodes) | Min | Max | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_single_1tick_r1_20261117_212824` | single | 1 | yes | 200k | 6.7 | **30.60** ± 0.32 (50) | 28 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/x7qpqe0q) |
| `fixed_single_1tick_r2_20261117_215228` | single | 1 | yes | 200k | 7.0 | **31.08** ± 0.25 (50) | 29 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/0nr3tr5u) |
| `fixed_single_1tick_r3_20261117_221638` | single | 1 | yes | 200k | 7.0 | **29.70** ± 0.41 (50) | 27 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/5lsm4fyz) |
| `fixed_multi_1tick_r1_20261117_213859` | multi | 1 | yes | 200k | 9.2 | **30.90** ± 0.25 (50) | 29 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/nzb09mc9) |
| `fixed_multi_1tick_r2_20261117_220328` | multi | 1 | yes | 200k | 9.3 | **31.74** ± 0.13 (50) | 30 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/j21v4v2p) |
| `fixed_multi_1tick_r3_resumed_20261118_114725` | multi | 1 | yes | 200k | 3.1 (resumed at 180k) | **31.38** ± 0.21 (50) | 29 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/17qszxj3) |

| Setup | Runs | Average of runs | Each run |
| --- | --- | --- | --- |
| single, 1 tick, 200k | 3 | **30.46** | 30.6 31.1 29.7 |
| multi, 1 tick, 200k | 3 | **31.34** | 30.9 31.7 31.4 |

<a id="arena-exp-4"></a>

### 4 · Action repeat: 1 vs 4 ticks per step (ground fixed)

**Question:** Experiment 2 again on the fixed arena: 1 run each of single and multi at 4 ticks, compared with experiment 3's 1-tick runs (listed again here).

**Found:** Yes, 4 ticks is worse, for both action sets: single 24.0 vs 30.5 at 1 tick (every 1-tick run scored 29.7 or more), multi 27.8 vs 31.3. One 4-tick run each, but the gaps are many times the spread between seeds. The scores match experiment 2's (single 23.0 and 23.6, multi 27.7), so the ground bug barely touched the 4-tick runs; single agents still leave most bottom logs. Runs were matched on decisions (200k), so the 4-tick agents saw 4x the game time (800k ticks) and still lost; matched on game time instead it's worse still (experiment 2's 4-tick checkpoint at 40k decisions scored 21.7). Wall time doesn't save it either: 4-tick runs took about twice as long. Keep 1 tick.

| Run | Actions | Ticks/step | Ground fixed | Steps | Train min | Eval mean ± 95% (episodes) | Min | Max | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fixed_single_4tick_20261118_142327` | single | 4 | yes | 200k | 14.6 | **23.98** ± 0.25 (50) | 21 | 26 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/zisdrew4) |
| `fixed_multi_4tick_20261118_144105` | multi | 4 | yes | 200k | 17.2 | **27.82** ± 0.47 (50) | 24 | 31 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/wny7ar7h) |
| `fixed_single_1tick_r1_20261117_212824` | single | 1 | yes | 200k | 6.7 | **30.60** ± 0.32 (50) | 28 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/x7qpqe0q) |
| `fixed_single_1tick_r2_20261117_215228` | single | 1 | yes | 200k | 7.0 | **31.08** ± 0.25 (50) | 29 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/0nr3tr5u) |
| `fixed_single_1tick_r3_20261117_221638` | single | 1 | yes | 200k | 7.0 | **29.70** ± 0.41 (50) | 27 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/5lsm4fyz) |
| `fixed_multi_1tick_r1_20261117_213859` | multi | 1 | yes | 200k | 9.2 | **30.90** ± 0.25 (50) | 29 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/nzb09mc9) |
| `fixed_multi_1tick_r2_20261117_220328` | multi | 1 | yes | 200k | 9.3 | **31.74** ± 0.13 (50) | 30 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/j21v4v2p) |
| `fixed_multi_1tick_r3_resumed_20261118_114725` | multi | 1 | yes | 200k | 3.1 (resumed at 180k) | **31.38** ± 0.21 (50) | 29 | 32 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/17qszxj3) |

| Setup | Runs | Average of runs | Each run |
| --- | --- | --- | --- |
| single, 4 ticks, 200k | 1 | **23.98** | 24.0 |
| multi, 4 ticks, 200k | 1 | **27.82** | 27.8 |
| single, 1 tick, 200k | 3 | **30.46** | 30.6 31.1 29.7 |
| multi, 1 tick, 200k | 3 | **31.34** | 30.9 31.7 31.4 |
<!-- AUTO-GENERATED:END -->
