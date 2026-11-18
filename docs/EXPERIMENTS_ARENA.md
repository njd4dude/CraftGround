# CraftGround Experiments: Chop Trees (arena)

Goal: train an agent to chop as many oak logs as possible in one episode. Natural-world results
are in [EXPERIMENTS_NATURAL.md](EXPERIMENTS_NATURAL.md); the summary of findings is [FINDINGS.md](FINDINGS.md).

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-18 15:31._

## Setup

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

## Experiments

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

**Found:** Multi slightly ahead, not proven: 31.3 vs 30.5 average, and multi's worst episodes are better, but the groups overlap (permutation test p = 0.2). Both are near the 32 cap, so the arena can't separate them; in the natural world multi won clearly: 26.6 vs 22.4 for the same recipe (docs/EXPERIMENTS_NATURAL.md, Line 4).

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

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.

- **2026-11-13:** `check_env.py` passed. A scripted agent chopped 3 logs per episode and the trees regrew after reset. Rewards sometimes show up 1–2 steps after the attack that earned them.
- **2026-11-13, run 1 (`chop_ppo_20261113_131702`):** PPO learned fast. Logs per episode went 3.5 → 18 → 25 → 28 over the first 70 episodes and finished at a last-20 average of 31.4/32. The task may now be too easy, so next ideas are harder layouts, more trees or random tree positions.
- **2026-11-13:** Parallel envs: `train.py` runs 6 Minecraft instances by default (`--n-envs`); `--n-steps` is the total across all envs, so each update is still ~2048 steps. Raw env benchmark (random actions): 1 env 451, 2 → 714, 4 → 858, 8 → 1076 steps/s, with 8 using 92% CPU and 17 GB RAM. The CPU is the limit; the GPU stays under 40%.
- **2026-11-17:** Speed with 3 instances: single actions ~490 steps/s, multi ~370 (turning the camera looks like the slow part, not verified), 4 ticks per step ~220 steps/s (~880 ticks/s). Each arena instance uses ~2.4 GB, so 3 is the safe number on this machine when anything else is open.
