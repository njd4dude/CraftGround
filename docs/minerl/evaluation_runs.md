# MineRL in CraftGround: Evaluation runs

[← MineRL in CraftGround index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 22:33._

One row per `evaluate.py` run that went to `results/eval_results_minerl.csv` (MineRL mode, or one of our models with a non-standard axe, episode length or log cap), newest first. **Steps/s** counts decisions (4 ticks each in MineRL mode).

| When | Model | Path | World | Spots | Ticks | Log cap | Axe | Episodes | Mean logs | Minutes | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-19T22:32:21 | Line 5 @ 4.0M | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | natural | fixed forest spots | 4000 | 64 | instant | 50 | 54.64 | 10.5 | 259 |
| 2026-11-19T22:21:27 | Line 5 @ 4.0M | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | natural | fixed forest spots | 4000 | – | instant | 50 | 118.94 | 11.5 | 345 |
| 2026-11-19T22:09:47 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | natural | fixed forest spots | 4000 | 64 | iron | 50 | 44.14 | 11.4 | 78 |
| 2026-11-19T21:58:11 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | natural | fixed forest spots | 4000 | 64 | iron | 50 | 48.44 | 10.8 | 83 |
| 2026-11-19T21:47:19 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed forest spots | 4000 | 64 | iron | 50 | 54.04 | 11.2 | 75 |
| 2026-11-19T19:53:08 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed forest spots | 1000 | – | instant | 100 | 29.41 | 10.3 | 50 |
| 2026-11-19T19:42:44 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | natural | fixed forest spots | 1000 | – | instant | 100 | 24.31 | 11.4 | 44 |
| 2026-11-19T19:31:16 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | arena | arena | 1000 | – | instant | 50 | 27.76 | 3.5 | 174 |
| 2026-11-19T19:27:40 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | arena | arena | 1000 | – | instant | 50 | 26.40 | 3.3 | 186 |
| 2026-11-19T19:20:59 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | natural | fixed forest spots | 1000 | – | iron | 100 | 16.26 | 10.0 | 51 |
| 2026-11-19T19:10:50 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed forest spots | 1000 | – | iron | 100 | 19.58 | 9.8 | 53 |
| 2026-11-19T19:00:56 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | natural | fixed forest spots | 1000 | – | iron | 100 | 17.50 | 9.2 | 58 |
| 2026-11-19T18:51:39 | Random actions | `random-minerl` | natural | fixed forest spots | 1000 | – | iron | 100 | 0.59 | 8.4 | 63 |
| 2026-11-19T18:43:08 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | arena | arena | 1000 | – | iron | 50 | 16.66 | 3.4 | 180 |
| 2026-11-19T18:39:39 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | arena | arena | 1000 | – | iron | 50 | 21.40 | 3.4 | 176 |
| 2026-11-19T18:36:08 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | arena | arena | 1000 | – | iron | 50 | 19.96 | 3.3 | 193 |
| 2026-11-19T18:32:44 | Random actions | `random-minerl` | arena | arena | 1000 | – | iron | 50 | 0.18 | 3.2 | 226 |
<!-- AUTO-GENERATED:END -->
