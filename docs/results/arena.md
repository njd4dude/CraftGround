# Arena: walled flat arena with 8 oaks

Every agent in the one arena: superflat, stone-brick walls, 8 oaks of 4 logs (32 logs at most), rebuilt every episode. Only the agents' actions differ between runs.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-22 13:40._

## Results

Every evaluated arena agent, best first: 1,000-tick episodes, the one-hit axe, at most 32 logs. **± is the 95% range** of the mean. **Ground not fixed**: dug holes stayed for the rest of the run (before 2026-11-17), so those scores are rough. Why each run was made, grouped by experiment: [Arena experiments](../experiments/arena_experiments.md).

| Model | Setup | Episodes | Mean logs | Min | Max |
| --- | --- | --- | --- | --- | --- |
| `fixed_multi_1tick_r2_20261117_220328` | multi actions, 1 tick/step | 50 | **31.74** ± 0.13 | 30 | 32 |
| `chop_ppo_20261113_131702` | single actions, 1 tick/step, **ground not fixed** | 24 | **31.67** ± 0.19 | 31 | 32 |
| `fixed_multi_1tick_r3_resumed_20261118_114725` | multi actions, 1 tick/step | 50 | **31.38** ± 0.21 | 29 | 32 |
| `fixed_single_1tick_r2_20261117_215228` | single actions, 1 tick/step | 50 | **31.08** ± 0.25 | 29 | 32 |
| `fixed_multi_1tick_r1_20261117_213859` | multi actions, 1 tick/step | 50 | **30.90** ± 0.25 | 29 | 32 |
| `multi_1tick_20261117_183147` | multi actions, 1 tick/step, **ground not fixed** | 50 | **30.80** ± 0.33 | 28 | 32 |
| `fixed_single_1tick_r1_20261117_212824` | single actions, 1 tick/step | 50 | **30.60** ± 0.32 | 28 | 32 |
| `fixed_single_1tick_r3_20261117_221638` | single actions, 1 tick/step | 50 | **29.70** ± 0.41 | 27 | 32 |
| `single_1tick_20261117_172920` | single actions, 1 tick/step, **ground not fixed** | 50 | **28.56** ± 0.47 | 24 | 32 |
| `fixed_multi_4tick_20261118_144105` | multi actions, 4 ticks/step | 50 | **27.82** ± 0.47 | 24 | 31 |
| MineRL Impala BC→PPO | MineRL mode (64×64, 4 ticks/decision) | 50 | **27.76** ± 1.62 | 3 | 32 |
| `multi_4tick_20261117_184506` | multi actions, 4 ticks/step, **ground not fixed** | 50 | **27.68** ± 0.46 | 24 | 31 |
| MineRL NatureCNN BC→PPO | MineRL mode (64×64, 4 ticks/decision) | 50 | **26.40** ± 0.95 | 16 | 30 |
| `chop_ppo_20261113_135620` | single actions, 1 tick/step, **ground not fixed** | 429 | **26.20** ± 0.17 | 18 | 31 |
| `single_1tick_seed2_20261117_190806` | single actions, 1 tick/step, **ground not fixed** | 50 | **24.20** ± 2.79 | 2 | 32 |
| `fixed_single_4tick_20261118_142327` | single actions, 4 ticks/step | 50 | **23.98** ± 0.25 | 21 | 26 |
| `single_4tick_seed2_20261117_181309` | single actions, 4 ticks/step, **ground not fixed** | 50 | **23.60** ± 0.23 | 20 | 25 |
| `single_4tick_20261117_173652` | single actions, 4 ticks/step, **ground not fixed** | 50 | **23.04** ± 0.28 | 20 | 24 |
| `single_4tick_20261117_173652` @ 60k | single actions, 4 ticks/step, **ground not fixed** | 50 | **22.94** ± 0.38 | 18 | 24 |
| `single_4tick_20261117_173652` @ 40k | single actions, 4 ticks/step, **ground not fixed** | 50 | **21.66** ± 0.53 | 15 | 25 |
| Random actions | single actions | 3 | **3.00** ± 2.99 | 0 | 5 |
<!-- AUTO-GENERATED:END -->
