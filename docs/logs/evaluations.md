# Log: evaluation runs

Every `evaluate.py` run, newest first. Each run's every episode is in the CSVs in `results/`.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 15:10._

## Natural world

One row per `evaluate.py` run, newest first. **Minutes** includes launching Minecraft; **Steps/s** is env steps per second over all instances while playing (runs from before timing was added show –). **Spots**: fixed = one of the two saved sets of 100 spots (oak/birch forests, or every biome with trees), random = random spawns (all runs before 2026-11-16). **World**: training = the world agents train in, held-out = one they never see (`--held-out`).

| When | Agent | Spots | World | Episodes | Mean logs | Min | Max | Instances | Minutes | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-19T18:02:55 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` | random, all tree biomes | training | 100 | 37.72 | 0 | 186 | 3 | 10.2 | 202 |
| 2026-11-19T17:52:17 | `runs/natural/line5b_resume-4M/20261119_161818/model.zip` | random, all tree biomes | training | 100 | 42.77 | 0 | 139 | 3 | 10.4 | 192 |
| 2026-11-19T17:41:29 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | random, all tree biomes | training | 100 | 35.01 | 0 | 182 | 3 | 10.0 | 225 |
| 2026-11-19T17:30:36 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` | random, oak/birch | training | 100 | 44.78 | 0 | 107 | 3 | 9.3 | 227 |
| 2026-11-19T17:20:51 | `runs/natural/line5b_resume-4M/20261119_161818/model.zip` | random, oak/birch | training | 100 | 45.32 | 0 | 102 | 3 | 8.0 | 288 |
| 2026-11-19T17:12:20 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | random, oak/birch | training | 100 | 38.76 | 0 | 89 | 3 | 7.6 | 301 |
| 2026-11-19T16:58:38 | `runs/natural/line5b_resume-4M/20261119_161818/model.zip` | fixed, all tree biomes | training | 100 | 31.33 | 0 | 92 | 3 | 10.8 | 184 |
| 2026-11-19T16:47:21 | `runs/natural/line5b_resume-4M/20261119_161818/model.zip` | fixed, oak/birch | training | 100 | 38.64 | 0 | 94 | 3 | 10.3 | 197 |
| 2026-11-19T13:22:10 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` | fixed, all tree biomes | training | 100 | 25.66 | 0 | 102 | 3 | 11.3 | 179 |
| 2026-11-19T13:10:22 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` | fixed, oak/birch | training | 100 | 41.27 | 0 | 109 | 3 | 10.2 | 203 |
| 2026-11-19T01:52:24 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | fixed, oak/birch | training | 100 | 39.99 | 0 | 89 | 3 | 10.1 | 203 |
| 2026-11-19T01:41:52 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | fixed, all tree biomes | training | 100 | 31.64 | 0 | 101 | 3 | 11.6 | 171 |
| 2026-11-19T01:29:47 | `runs/natural/line4c_resume-2M/20261118_183414/model.zip` | fixed, all tree biomes | training | 100 | 30.50 | 0 | 88 | 3 | 11.0 | 183 |
| 2026-11-18T20:22:22 | `runs/natural/line4c_resume-2M/20261118_183414/model.zip` | fixed, oak/birch | training | 100 | 41.18 | 0 | 94 | 3 | 9.3 | 222 |
| 2026-11-18T18:14:57 | `runs/natural/line4b_resume-1M/20261118_162138/model.zip` | fixed, oak/birch | held-out | 100 | 39.58 | 0 | 95 | 3 | 9.7 | 205 |
| 2026-11-18T17:53:52 | `runs/natural/line4b_resume-1M/20261118_162138/model.zip` | fixed, oak/birch | training | 100 | 35.73 | 0 | 97 | 3 | 9.1 | 227 |
| 2026-11-18T14:22:20 | `runs/natural/line4_multi-actions/20261118_124345/model.zip` | fixed, oak/birch | training | 100 | 26.60 | 0 | 67 | 3 | 9.3 | 226 |
| 2026-11-17T13:43:24 | `runs/natural/line3c_linear-lr-3M/20261116_173633/checkpoints/ppo_chop_996932_steps.zip` | fixed, oak/birch | training | 100 | 16.15 | 0 | 47 | 3 | 10.0 | 216 |
| 2026-11-17T13:32:59 | `runs/natural/line3c_linear-lr-3M/20261117_114712/model.zip` | fixed, oak/birch | training | 100 | 23.93 | 0 | 69 | 3 | 11.5 | 188 |
| 2026-11-16T16:47:06 | `random` | fixed, oak/birch | held-out | 100 | 1.14 | 0 | 11 | 3 | 9.7 | 213 |
| 2026-11-16T16:37:23 | `runs/natural/line3_jump-linear-lr-scratch/20261116_123155/model.zip` | fixed, oak/birch | held-out | 100 | 21.93 | 0 | 68 | 3 | 11.8 | 169 |
| 2026-11-16T16:25:32 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` | fixed, oak/birch | held-out | 100 | 25.42 | 0 | 70 | 3 | 14.2 | 136 |
| 2026-11-16T15:38:05 | `runs/natural/line3b_finetune-500k/20261116_145213/model.zip` | fixed, oak/birch | training | 100 | 21.86 | 0 | 53 | 3 | 10.8 | 188 |
| 2026-11-16T14:46:08 | `runs/natural/line2_jump-constant-lr/20261115_192015/model.zip` | fixed, oak/birch | training | 100 | 14.98 | 0 | 43 | 3 | 9.5 | 224 |
| 2026-11-16T13:51:42 | `runs/natural/line3_jump-linear-lr-scratch/20261116_123155/model.zip` | fixed, oak/birch | training | 100 | 22.39 | 0 | 53 | 3 | 11.3 | 181 |
| 2026-11-16T11:57:57 | `runs/natural/line3_jump-linear-lr-scratch/20261115_235545/model.zip` | fixed, oak/birch | training | 100 | 12.35 | 0 | 37 | 3 | 10.2 | 210 |
| 2026-11-16T11:47:39 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` | fixed, oak/birch | training | 100 | 24.74 | 0 | 69 | 3 | 12.5 | 162 |
| 2026-11-16T11:35:02 | `runs/natural/line2_jump-constant-lr/20261115_185527/checkpoints/ppo_chop_601536_steps.zip` | fixed, oak/birch | training | 100 | 18.07 | 0 | 59 | 3 | 10.6 | 202 |
| 2026-11-16T11:24:18 | `random` | fixed, oak/birch | training | 100 | 1.14 | 0 | 11 | 3 | 11.1 | 190 |
| 2026-11-15T23:50:28 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` | random, oak/birch | training | 100 | 25.44 | 0 | 66 | 3 | 9.8 | 251 |
| 2026-11-15T20:02:25 | `runs/natural/line2_jump-constant-lr/20261115_192015/model.zip` | random, oak/birch | training | 100 | 14.99 | 0 | 41 | 4 | 8.5 | 286 |
| 2026-11-15T19:53:35 | `runs/natural/line2_jump-constant-lr/20261115_185527/checkpoints/ppo_chop_601536_steps.zip` | random, oak/birch | training | 100 | 21.67 | 0 | 66 | 4 | 8.7 | 268 |
| 2026-11-15T18:54:25 | `runs/natural/line2_jump-constant-lr/20261115_173930/model.zip` | random, oak/birch | training | 100 | 17.96 | 0 | 63 | 4 | 8.2 | 298 |
| 2026-11-15T16:51:49 | `runs/natural/line1_no-jump/20261115_161832/model.zip` | random, oak/birch | training | 100 | 9.68 | 0 | 43 | 4 | 8.1 | 300 |
| 2026-11-14T17:52:51 | `runs/arena/chop_ppo_20261113_131702/model.zip` | random, oak/birch | training | 100 | 5.01 | 0 | 30 | – | – | – |
| 2026-11-14T17:26:55 | `random` | random, oak/birch | training | 100 | 1.81 | 0 | 24 | – | – | – |
| 2026-11-14T17:12:38 | `runs/arena/chop_ppo_20261113_135620/model.zip` | random, oak/birch | training | 100 | 5.10 | 0 | 18 | – | – | – |

Every episode's score is in the CSV (`per_episode`, with `spot_per_episode` saying which spot each one played).

## Arena

One row per `evaluate.py` run, newest first. **Minutes** includes launching Minecraft; **Steps/s** is env steps per second over all instances while playing (runs from before timing was added show –).

| When | Agent | Episodes | Mean logs | Min | Max | Instances | Minutes | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-18T15:01:16 | `runs/arena/fixed_multi_4tick_20261118_144105/model.zip` | 50 | 27.82 | 24 | 31 | 3 | 2.6 | 186 |
| 2026-11-18T14:40:58 | `runs/arena/fixed_single_4tick_20261118_142327/model.zip` | 50 | 23.98 | 21 | 26 | 3 | 2.4 | 200 |
| 2026-11-18T11:54:37 | `runs/arena/fixed_multi_1tick_r3_resumed_20261118_114725/model.zip` | 50 | 31.38 | 29 | 32 | 3 | 3.3 | 532 |
| 2026-11-17T22:27:32 | `runs/arena/fixed_single_1tick_r3_20261117_221638/model.zip` | 50 | 29.70 | 27 | 32 | 3 | 3.3 | 594 |
| 2026-11-17T22:16:28 | `runs/arena/fixed_multi_1tick_r2_20261117_220328/model.zip` | 50 | 31.74 | 30 | 32 | 3 | 3.0 | 520 |
| 2026-11-17T22:03:17 | `runs/arena/fixed_single_1tick_r2_20261117_215228/model.zip` | 50 | 31.08 | 29 | 32 | 3 | 3.2 | 579 |
| 2026-11-17T21:52:16 | `runs/arena/fixed_multi_1tick_r1_20261117_213859/model.zip` | 50 | 30.90 | 29 | 32 | 3 | 3.4 | 536 |
| 2026-11-17T21:38:48 | `runs/arena/fixed_single_1tick_r1_20261117_212824/model.zip` | 50 | 30.60 | 28 | 32 | 3 | 3.1 | 585 |
| 2026-11-17T19:18:48 | `runs/arena/single_1tick_seed2_20261117_190806/model.zip` | 50 | 24.20 | 2 | 32 | 3 | 3.2 | 610 |
| 2026-11-17T19:07:24 | `runs/arena/multi_4tick_20261117_184506/model.zip` | 50 | 27.68 | 24 | 31 | 3 | 3.0 | 175 |
| 2026-11-17T18:44:54 | `runs/arena/multi_1tick_20261117_183147/model.zip` | 50 | 30.80 | 28 | 32 | 3 | 3.4 | 543 |
| 2026-11-17T18:31:40 | `runs/arena/single_4tick_seed2_20261117_181309/model.zip` | 50 | 23.60 | 20 | 25 | 3 | 3.0 | 196 |
| 2026-11-17T18:05:56 | `runs/arena/single_4tick_20261117_173652/checkpoints/ppo_chop_59994_steps.zip` | 50 | 22.94 | 18 | 24 | 3 | 2.8 | 198 |
| 2026-11-17T18:02:29 | `runs/arena/single_4tick_20261117_173652/checkpoints/ppo_chop_39996_steps.zip` | 50 | 21.66 | 15 | 25 | 3 | 2.9 | 203 |
| 2026-11-17T17:59:26 | `runs/arena/single_4tick_20261117_173652/model.zip` | 50 | 23.04 | 20 | 24 | 3 | 2.8 | 195 |
| 2026-11-17T17:56:24 | `runs/arena/single_1tick_20261117_172920/model.zip` | 50 | 28.56 | 24 | 32 | 3 | 3.3 | 610 |
| 2026-11-14T14:56:13 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 26.53 | 23 | 30 | – | – | – |
| 2026-11-14T14:49:56 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 26.67 | 23 | 31 | – | – | – |
| 2026-11-14T14:44:44 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 25.60 | 18 | 30 | – | – | – |
| 2026-11-14T14:21:57 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 100 | 26.00 | 21 | 30 | – | – | – |
| 2026-11-14T13:52:20 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 5 | 25.00 | 23 | 26 | – | – | – |
| 2026-11-14T13:36:22 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 2 | 23.00 | 22 | 24 | – | – | – |
| 2026-11-14T13:34:49 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 2 | 26.00 | 26 | 26 | – | – | – |
| 2026-11-13T14:13:19 | `runs/arena/chop_ppo_20261113_135620/model.zip` | 20 | 26.85 | 23 | 30 | – | – | – |
| 2026-11-13T13:52:31 | `runs/arena/chop_ppo_20261113_131702/model.zip` | 2 | 32.00 | 32 | 32 | – | – | – |
| 2026-11-13T13:50:12 | `runs/arena/chop_ppo_20261113_131702/model.zip` | 2 | 32.00 | 32 | 32 | – | – | – |
| 2026-11-13T13:46:35 | `runs/arena/chop_ppo_20261113_131702/model.zip` | 20 | 31.60 | 31 | 32 | – | – | – |
| 2026-11-13T13:12:10 | `random` | 3 | 3.00 | 0 | 5 | – | – | – |

Every episode's score is in the CSV (`per_episode`, with `spot_per_episode` saying which spot each one played).

## MineRL mode and other evaluations

One row per `evaluate.py` run that went to `results/eval_results_minerl.csv` (MineRL mode, or one of our models with a non-standard axe, episode length or log cap), newest first. **Steps/s** counts decisions (4 ticks each in MineRL mode).

| When | Model | Path | World | Spots | Ticks | Log cap | Axe | Episodes | Mean logs | Minutes | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-20T23:48:34 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed spots in every biome with trees | 1000 | – | instant | 100 | 25.72 | 10.1 | 50 |
| 2026-11-20T23:38:20 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | natural | fixed spots in every biome with trees | 1000 | – | instant | 100 | 21.15 | 10.3 | 49 |
| 2026-11-20T23:27:54 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 22.18 | 10.0 | 51 |
| 2026-11-20T23:17:46 | MineRL Impala BC + fine-tune @ 480k ticks | `runs/minerl_mode/natural/ft_impala-bc_wooded_one-hit-axe_20261120_193516/checkpoints/ppo_chop_120000_steps.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 37.17 | 10.2 | 50 |
| 2026-11-20T23:07:26 | MineRL Impala BC + fine-tune @ 1.0M ticks | `runs/minerl_mode/natural/ft_impala-bc_wooded_one-hit-axe_20261120_220917/model.zip` | natural | fixed spots in every biome with trees | 1000 | – | instant | 100 | 37.48 | 11.6 | 43 |
| 2026-11-20T22:55:41 | MineRL Impala BC + fine-tune @ 1.0M ticks | `runs/minerl_mode/natural/ft_impala-bc_wooded_one-hit-axe_20261120_220917/model.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 40.43 | 9.8 | 53 |
| 2026-11-20T19:31:36 | MineRL Impala PPO + fine-tune @ 1.5M ticks | `runs/minerl_mode/natural/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_495292_steps.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 44.54 | 11.2 | 45 |
| 2026-11-20T19:20:16 | MineRL Impala PPO + fine-tune @ 2.0M ticks | `runs/minerl_mode/natural/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_615292_steps.zip` | natural | fixed spots in every biome with trees | 1000 | – | instant | 100 | 38.54 | 10.7 | 47 |
| 2026-11-20T19:09:25 | MineRL Impala PPO + fine-tune @ 2.0M ticks | `runs/minerl_mode/natural/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_615292_steps.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 44.66 | 10.7 | 47 |
| 2026-11-20T14:54:33 | MineRL Impala PPO + fine-tune @ 480k ticks | `runs/minerl_mode/natural/ft_impala-ppo_wooded_one-hit-axe_20261120_124557/checkpoints/ppo_chop_242874_steps.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 38.48 | 9.7 | 53 |
| 2026-11-20T14:44:44 | MineRL Impala PPO + fine-tune @ 1.0M ticks | `runs/minerl_mode/natural/ft_impala-ppo_wooded_one-hit-axe_20261120_124557/model.zip` | natural | fixed spots in every biome with trees | 1000 | – | instant | 100 | 37.22 | 12.8 | 38 |
| 2026-11-20T14:31:46 | MineRL Impala PPO + fine-tune @ 1.0M ticks | `runs/minerl_mode/natural/ft_impala-ppo_wooded_one-hit-axe_20261120_124557/model.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 43.35 | 9.8 | 53 |
| 2026-11-19T22:32:21 | Line 5 @ 4.0M | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | natural | fixed spots in oak/birch forests | 4000 | 64 | instant | 50 | 54.64 | 10.5 | 259 |
| 2026-11-19T22:21:27 | Line 5 @ 4.0M | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | natural | fixed spots in oak/birch forests | 4000 | – | instant | 50 | 118.94 | 11.5 | 345 |
| 2026-11-19T22:09:47 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | natural | fixed spots in oak/birch forests | 4000 | 64 | iron | 50 | 44.14 | 11.4 | 78 |
| 2026-11-19T21:58:11 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | natural | fixed spots in oak/birch forests | 4000 | 64 | iron | 50 | 48.44 | 10.8 | 83 |
| 2026-11-19T21:47:19 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed spots in oak/birch forests | 4000 | 64 | iron | 50 | 54.04 | 11.2 | 75 |
| 2026-11-19T19:53:08 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 29.41 | 10.3 | 50 |
| 2026-11-19T19:42:44 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | natural | fixed spots in oak/birch forests | 1000 | – | instant | 100 | 24.31 | 11.4 | 44 |
| 2026-11-19T19:31:16 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | arena | arena | 1000 | – | instant | 50 | 27.76 | 3.5 | 174 |
| 2026-11-19T19:27:40 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | arena | arena | 1000 | – | instant | 50 | 26.40 | 3.3 | 186 |
| 2026-11-19T19:20:59 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | natural | fixed spots in oak/birch forests | 1000 | – | iron | 100 | 16.26 | 10.0 | 51 |
| 2026-11-19T19:10:50 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | natural | fixed spots in oak/birch forests | 1000 | – | iron | 100 | 19.58 | 9.8 | 53 |
| 2026-11-19T19:00:56 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | natural | fixed spots in oak/birch forests | 1000 | – | iron | 100 | 17.50 | 9.2 | 58 |
| 2026-11-19T18:51:39 | Random actions | `random-minerl` | natural | fixed spots in oak/birch forests | 1000 | – | iron | 100 | 0.59 | 8.4 | 63 |
| 2026-11-19T18:43:08 | MineRL Impala BC | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` | arena | arena | 1000 | – | iron | 50 | 16.66 | 3.4 | 180 |
| 2026-11-19T18:39:39 | MineRL Impala PPO | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` | arena | arena | 1000 | – | iron | 50 | 21.40 | 3.4 | 176 |
| 2026-11-19T18:36:08 | MineRL NatureCNN PPO | `runs/minerl/model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05/final.zip` | arena | arena | 1000 | – | iron | 50 | 19.96 | 3.3 | 193 |
| 2026-11-19T18:32:44 | Random actions | `random-minerl` | arena | arena | 1000 | – | iron | 50 | 0.18 | 3.2 | 226 |
<!-- AUTO-GENERATED:END -->
