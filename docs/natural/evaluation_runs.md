# Natural world: Evaluation runs

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 13:52._

One row per `evaluate.py` run, newest first. **Minutes** includes launching Minecraft; **Steps/s** is env steps per second over all instances while playing (runs from before timing was added show –). **Spots**: fixed = the saved forest spots, wooded = the saved wooded spots (any biome with trees), random = random spawns (all runs before 2026-11-16). **World**: training = the world agents train in, held-out = one they never see (`--held-out`).

| When | Agent | Spots | World | Episodes | Mean logs | Min | Max | Instances | Minutes | Steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-19T13:22:10 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` | wooded | training | 100 | 25.66 | 0 | 102 | 3 | 11.3 | 179 |
| 2026-11-19T13:10:22 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` | fixed | training | 100 | 41.27 | 0 | 109 | 3 | 10.2 | 203 |
| 2026-11-19T01:52:24 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | fixed | training | 100 | 39.99 | 0 | 89 | 3 | 10.1 | 203 |
| 2026-11-19T01:41:52 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` | wooded | training | 100 | 31.64 | 0 | 101 | 3 | 11.6 | 171 |
| 2026-11-19T01:29:47 | `runs/natural/line4c_resume-2M/20261118_183414/model.zip` | wooded | training | 100 | 30.50 | 0 | 88 | 3 | 11.0 | 183 |
| 2026-11-18T20:22:22 | `runs/natural/line4c_resume-2M/20261118_183414/model.zip` | fixed | training | 100 | 41.18 | 0 | 94 | 3 | 9.3 | 222 |
| 2026-11-18T18:14:57 | `runs/natural/line4b_resume-1M/20261118_162138/model.zip` | fixed | held-out | 100 | 39.58 | 0 | 95 | 3 | 9.7 | 205 |
| 2026-11-18T17:53:52 | `runs/natural/line4b_resume-1M/20261118_162138/model.zip` | fixed | training | 100 | 35.73 | 0 | 97 | 3 | 9.1 | 227 |
| 2026-11-18T14:22:20 | `runs/natural/line4_multi-actions/20261118_124345/model.zip` | fixed | training | 100 | 26.60 | 0 | 67 | 3 | 9.3 | 226 |
| 2026-11-17T13:43:24 | `runs/natural/line3c_linear-lr-3M/20261116_173633/checkpoints/ppo_chop_996932_steps.zip` | fixed | training | 100 | 16.15 | 0 | 47 | 3 | 10.0 | 216 |
| 2026-11-17T13:32:59 | `runs/natural/line3c_linear-lr-3M/20261117_114712/model.zip` | fixed | training | 100 | 23.93 | 0 | 69 | 3 | 11.5 | 188 |
| 2026-11-16T16:47:06 | `random` | fixed | held-out | 100 | 1.14 | 0 | 11 | 3 | 9.7 | 213 |
| 2026-11-16T16:37:23 | `runs/natural/line3_jump-linear-lr-scratch/20261116_123155/model.zip` | fixed | held-out | 100 | 21.93 | 0 | 68 | 3 | 11.8 | 169 |
| 2026-11-16T16:25:32 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` | fixed | held-out | 100 | 25.42 | 0 | 70 | 3 | 14.2 | 136 |
| 2026-11-16T15:38:05 | `runs/natural/line3b_finetune-500k/20261116_145213/model.zip` | fixed | training | 100 | 21.86 | 0 | 53 | 3 | 10.8 | 188 |
| 2026-11-16T14:46:08 | `runs/natural/line2_jump-constant-lr/20261115_192015/model.zip` | fixed | training | 100 | 14.98 | 0 | 43 | 3 | 9.5 | 224 |
| 2026-11-16T13:51:42 | `runs/natural/line3_jump-linear-lr-scratch/20261116_123155/model.zip` | fixed | training | 100 | 22.39 | 0 | 53 | 3 | 11.3 | 181 |
| 2026-11-16T11:57:57 | `runs/natural/line3_jump-linear-lr-scratch/20261115_235545/model.zip` | fixed | training | 100 | 12.35 | 0 | 37 | 3 | 10.2 | 210 |
| 2026-11-16T11:47:39 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` | fixed | training | 100 | 24.74 | 0 | 69 | 3 | 12.5 | 162 |
| 2026-11-16T11:35:02 | `runs/natural/line2_jump-constant-lr/20261115_185527/checkpoints/ppo_chop_601536_steps.zip` | fixed | training | 100 | 18.07 | 0 | 59 | 3 | 10.6 | 202 |
| 2026-11-16T11:24:18 | `random` | fixed | training | 100 | 1.14 | 0 | 11 | 3 | 11.1 | 190 |
| 2026-11-15T23:50:28 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` | random | training | 100 | 25.44 | 0 | 66 | 3 | 9.8 | 251 |
| 2026-11-15T20:02:25 | `runs/natural/line2_jump-constant-lr/20261115_192015/model.zip` | random | training | 100 | 14.99 | 0 | 41 | 4 | 8.5 | 286 |
| 2026-11-15T19:53:35 | `runs/natural/line2_jump-constant-lr/20261115_185527/checkpoints/ppo_chop_601536_steps.zip` | random | training | 100 | 21.67 | 0 | 66 | 4 | 8.7 | 268 |
| 2026-11-15T18:54:25 | `runs/natural/line2_jump-constant-lr/20261115_173930/model.zip` | random | training | 100 | 17.96 | 0 | 63 | 4 | 8.2 | 298 |
| 2026-11-15T16:51:49 | `runs/natural/line1_no-jump/20261115_161832/model.zip` | random | training | 100 | 9.68 | 0 | 43 | 4 | 8.1 | 300 |
| 2026-11-14T17:52:51 | `runs/arena/chop_ppo_20261113_131702/model.zip` | random | training | 100 | 5.01 | 0 | 30 | – | – | – |
| 2026-11-14T17:26:55 | `random` | random | training | 100 | 1.81 | 0 | 24 | – | – | – |
| 2026-11-14T17:12:38 | `runs/arena/chop_ppo_20261113_135620/model.zip` | random | training | 100 | 5.10 | 0 | 18 | – | – | – |

Every episode's score is in the CSV (`per_episode`, with `spot_per_episode` saying which spot each one played).
<!-- AUTO-GENERATED:END -->
