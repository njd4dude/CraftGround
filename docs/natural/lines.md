# Natural world: Training lines

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 14:34._

Each line is one training history in its own folder (`runs/natural/<line>/`); its pieces are the run folders inside, split by crashes, memory kills or deliberate stops, in order. `train.py --resume` adds a piece to the model's line; `--line <folder>` starts a new one (a branch when resuming). **Steps** with ~ come from the last checkpoint (the run left no record). **Eval** is the fixed-spot mean logs per episode at that point (* = random spawns only, older and noisier). Names, purposes and notes: `results/lines_natural.json`.

<a id="line-1"></a>

### Line 1 · no jump, constant LR

`runs/natural/line1_no-jump/`

**Question:** Can PPO learn to chop trees in a generated world at all? First natural-world run, the arena's 11 actions.

**Found:** Yes, but modestly: 9.7 logs per episode at 503k (random spawns), 5.3x random actions.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261115_155106` | from scratch | 0 → 100k | 4 | constant | 9.8 | completed | – |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/jnm6lysy) |
| `20261115_160512` | continued | 100k → ~200k | 6 | constant | – | no record (killed or hung) | – | 6 instances; hung while closing them (stopped from W&B), so no CSV row | – |
| `20261115_161832` | continued | 200k → 503k | 4 | constant | 19.0 | completed | 9.7* @ 503k |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/6hob06tk) |

<a id="line-2"></a>

### Line 2 · jump, constant LR

`runs/natural/line2_jump-constant-lr/`

**Question:** Does adding jump and forward_jump help on hilly terrain? Compare with Line 1.

**Found:** Yes: jump nearly doubled the score at ~500k (18.0 vs Line 1's 9.7, random spawns). With a constant LR it peaked near 600k (18.1 on fixed spots) and then got worse: 15.0 at 1M, -3.1 +/- 2.6 logs vs its 601k checkpoint spot by spot.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261115_172419` | from scratch | 0 → ~180k | 4 | constant | – | no record (killed or hung) | – | killed for low memory at ~188k; resumed from the 180k checkpoint | – |
| `20261115_173930` | continued | 180k → 502k | 4 | constant | 21.6 | completed | 18.0* @ 502k |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/wbto0vn4) |
| `20261115_185527` | continued | 502k → ~682k | 4 | constant | – | no record (killed or hung) | **18.1** @ 602k | killed for low memory at ~688k; its 601k checkpoint is Line 2's best and starts Line 2b | – |
| `20261115_192015` | continued | 682k → 1.0M | 4 | constant | 21.8 | completed | **15.0** @ 1.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/12zz1rvp) |

<a id="line-2b"></a>

### Line 2b · linear LR from Line 2's 601k checkpoint

`runs/natural/line2b_linear-lr-from-601k/`

**Question:** Line 2 slid from ~22 to ~15 logs between 600k and 1M at a constant learning rate. Does decaying the rate to 0 keep the gains?

**Found:** Yes: from the same 601k model it rose to 24.7 on fixed spots, while Line 2 fell to 15.0. Spot by spot: +6.7 +/- 3.1 logs over the split point and +9.8 +/- 3.5 over Line 2 @ 1M. Best model so far.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261115_231007` | branch from Line 2 | 602k → 1.0M | 3 | linear | 29.7 | completed | **24.7** @ 1.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/f61norjd) |

<a id="line-3"></a>

### Line 3 · jump, linear LR, from scratch

`runs/natural/line3_jump-linear-lr-scratch/`

**Question:** Line 2b's recipe in one consistent setup from step 0 (3 instances, render distance 4, linear LR). Does the patchwork history of 2b matter?

**Found:** No: 22.4 on fixed spots at 1M, tied with Line 2b (-2.3 +/- 3.0 spot by spot), with fewer 0-log episodes (6 vs 13). So 2b's result doesn't depend on its patchwork history.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261115_235545` | from scratch | 0 → 257k | 3 | linear | 20.5 | interrupted | **12.3** @ 257k | stopped on purpose (Ctrl+C) to free the machine for fixed spots | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/idjjc1jj) |
| `20261116_123155` | continued | 257k → 1.0M | 3 | linear | 67.5 | completed | **22.4** @ 1.0M | resumed with --learning-rate 1.858e-4 so the schedule matches one uninterrupted 1M run; hazards hurt from here on | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/2x96rt9o) |

<a id="line-3b"></a>

### Line 3b · fine-tune Line 3 @ 1.0M for 500k more at a small LR

`runs/natural/line3b_finetune-500k/`

**Question:** Line 3 ended with its learning rate at 0. Does 500k more steps at a small rate (5e-5 -> 0) still improve it, or is it at its limit with this setup?

**Found:** No: 21.9 on fixed spots at 1.5M, -0.5 +/- 3.1 logs vs Line 3 @ 1.0M spot by spot (better at 41 spots, worse at 51). The extra 500k steps changed nothing measurable; with this setup the model looks to be at its limit.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261116_145213` | branch from Line 3 | 1.0M → 1.5M | 3 | linear | 34.3 | completed | **21.9** @ 1.5M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/nsc8z41e) |

<a id="line-3c"></a>

### Line 3c · 3M linear LR schedule, from Line 3 @ 257k

`runs/natural/line3c_linear-lr-3M/`

**Question:** Line 3's 1M schedule stalled near 22 and fine-tuning (3b) didn't help. Does one longer schedule (2.29e-4 -> 0 over 257k -> 3M, as if a single 3M run) keep improving past 22? Also the first run with 4 instances (same 2048-step updates).

**Found:** No: 23.9 on fixed spots at 3M, tied with Line 3 @ 1M (+1.5 +/- 3.4 spot by spot, better at 47, worse at 48) and Line 2b (-0.8 +/- 3.5). At 1M, with its rate still high, it was behind Line 3 (16.1, -6.2 +/- 3.6) and then climbed back to the same level. Every recipe so far ends at ~22-25, so the limit is the setup, not training length.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261116_165326` | branch from Line 3 | 257k → ~717k | 4 | linear | – | no record (killed or hung) | – |  | – |
| `20261116_173633` | continued | 717k → 2.1M | 4 | linear | 103.2 | interrupted | **16.1** @ 1.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/rlc5ue18) |
| `20261117_114712` | continued | 2.1M → 3.0M | 3 | linear | 84.7 | completed | **23.9** @ 3.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/ato3q2uf) |

<a id="line-4"></a>

### Line 4 · multi actions (move x turn x look x jump, attack held), linear LR, from scratch

`runs/natural/line4_multi-actions/`

**Question:** Every single-action recipe ends at ~22-25 logs, so the limit looks like the setup. Do multi-choice actions (choose move, turn, look and jump every step, attack always held, exact-degree turns) beat Line 3's 22.4 at 1M with the same schedule? In the fixed arena they scored 31.3 vs 30.5 (3 runs each, not significant: the arena is near its 32 cap).

**Found:** Yes: 26.6 on fixed spots at 1M, the best so far. Spot by spot +4.2 +/- 3.0 logs vs Line 3 @ 1M (same recipe, only the actions differ; better at 57 spots, worse at 36) and +1.9 +/- 3.4 vs Line 2b (not significant). It learned slower early (training avg 15.4 vs Line 3's 20.3 at 330k-423k) but kept improving after Line 3 levelled off (29.8 vs 25.5 at 900k-1M).

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261118_121806` | from scratch | 0 → ~? | 3 | linear | – | no record (killed or hung) | – | killed for low memory at ~8k steps, before any checkpoint; restarted from scratch | – |
| `20261118_124345` | from scratch | 0 → 1.0M | 3 | linear | 88.8 | completed | **26.6** @ 1.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/hg4felig) |

<a id="line-4b"></a>

### Line 4b · resume Line 4 @ 1.0M for 1M more, linear LR from 1e-4

`runs/natural/line4b_resume-1M/`

**Question:** Line 4 was still improving at the end of its 1M schedule (training avg +4.6 logs in its last 100k), unlike Line 3, which had levelled off before 3b. Does 1M more steps (1e-4 -> 0) keep improving it?

**Found:** Yes: 35.7 on fixed spots at 2M, +9.1 +/- 3.6 logs vs Line 4 @ 1.0M spot by spot (better at 62, worse at 27). Held-out world 39.6 (gap +3.9, ranges overlap): it generalises. Training avg dipped for ~600k after the LR restart, then climbed as the rate fell.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261118_162138` | branch from Line 4 | 1.0M → 2.0M | 3 | linear | 82.2 | completed | **35.7** @ 2.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/1lbhj5a9) |

<a id="line-4c"></a>

### Line 4c · resume Line 4b @ 2.0M for 1M more, linear LR from 1e-4

`runs/natural/line4c_resume-2M/`

**Question:** Line 4b was still rising at 2M. Does a third 1M (same recipe) keep improving it?

**Found:** Yes, by less: 41.2 on fixed spots at 3M, +5.5 +/- 4.5 logs vs Line 4b @ 2.0M spot by spot (better at 54, worse at 38); 0-log spots 10 -> 7. Best model so far. A second 50-episode evaluation of the same model scored 40.5 (results/eval_results_natural_50ep_check.csv, kept out of the main table).

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261118_183414` | branch from Line 4b | 2.0M → 3.0M | 3 | linear | 98.2 | completed | **41.2** @ 3.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/ms2j7prx) |

<a id="line-5"></a>

### Line 5 · wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4

`runs/natural/line5_wooded-spawns/`

**Question:** Line 4 does badly in snow and gets stuck. Do spawns in any biome with trees (snowy taiga, taiga, jungle, swamp, dark forest...; not savanna) and a new region every 5 episodes make the agent better in unfamiliar biomes, without losing forest skill? Same recipe as 4c's last 1M; Line 4d is the control (forest spawns).

**Found:** Yes, against its control (Line 4d: same 1M steps, forest spawns): wooded spots 31.6 vs 25.7, +6.0 +/- 4.8 logs spot by spot (better at 58, worse at 32); forest spots tied (40.0 vs 41.3, -1.3 +/- 5.0). 0-log episodes over both sets halved, 13 vs 26 (McNemar p = 0.007). Against its start, Line 4c, it is a tie on both sets (wooded +1.1 +/- 4.5, forest -1.2 +/- 4.7): the gain is that wooded spawns stopped the drop outside forests that more forest training causes.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261118_210945` | branch from Line 4c | 3.0M → ~3.1M | 3 | linear | – | no record (killed or hung) | – | killed by Claude Code for low system memory at ~3.09M; resumed from the 3.08M checkpoint | – |
| `20261118_213914` | continued | 3.1M → ~3.8M | 3 | linear | – | no record (killed or hung) | – | resumed with --learning-rate 9.19e-5 so the schedule matches one 1M run; hung at 3.79M (cause unknown, logs overwritten) and stopped by hand | – |
| `20261119_005447` | continued | 3.8M → 4.0M | 3 | linear | 23.2 | completed | **40.0** @ 4.0M | resumed from the 3.78M checkpoint with --learning-rate 2.186e-5 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/1v3paa89) |

<a id="line-4d"></a>

### Line 4d · resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4

`runs/natural/line4d_resume-3M/`

**Question:** Control for Line 5: the same 1M steps and recipe with forest spawns only, so Line 5's effect can be told apart from simply training longer.

**Found:** Forest score flat: 41.3 at 4M vs Line 4c's 41.2 (+0.1 +/- 4.5), so the line has levelled off. Wooded spots fell from 30.5 (4c) to 25.7: more forest-only training made it worse outside oak/birch forests. Line 5 (same steps, wooded spawns) beat it there by +6.0 +/- 4.8 and had half the 0-log episodes.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261119_112830` | branch from Line 4c | 3.0M → 4.0M | 3 | linear | 90.5 | completed | **41.3** @ 4.0M |  | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/m7mtph3f) |

<a id="line-6"></a>

### Line 6 · wooded spawns from scratch, Line 4's recipe (multi actions, linear LR from 2.5e-4, 1M)

`runs/natural/line6_wooded-scratch/`

**Question:** Line 5 showed wooded spawns help on top of a forest-trained agent. Does training on them from step 0 work as well, compared with Line 4 @ 1M (same recipe, forest spawns)?

**Found:** Stopped by hand at ~490k (not evaluated): its training average was behind Line 4's at the same point (about 10 vs 13-16 logs per episode around 400 episodes, ~400k steps, in), though wooded episodes are harder, so the two averages don't compare directly. Last checkpoint 480k.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261119_133211` | from scratch | 0 → ~480k | 3 | linear | – | no record (killed or hung) | – | stopped by hand at ~491k: no clear difference from Line 4 so far; last checkpoint ppo_chop_479952_steps.zip | – |

<a id="line-5b"></a>

### Line 5b · resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4

`runs/natural/line5b_resume-4M/`

**Question:** Line 4d showed forest training levels off by 3M. Wooded spawns give the agent harder places it hasn't mastered (31.6 wooded vs 40.0 forest), so does Line 5 keep improving with 1M more steps?

**Found:** No: wooded 31.3 vs Line 5's 31.6 (-0.3 +/- 4.5 spot by spot), forest 38.6 vs 40.0 (-1.4 +/- 4.5), both ties; 0-log spots 9 -> 7 wooded, 4 -> 6 forest. It keeps Line 5's lead over the forest-trained Line 4d on wooded spots (+5.7 +/- 4.7). Wooded training levels off by 4M too, so the remaining wooded-vs-forest gap (~8 logs) needs something other than more steps.

| Run | Start | Steps | Instances | LR | Minutes | Status | Eval (logs) | Notes | W&B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `20261119_143552` | branch from Line 5 | 4.0M → ~4.8M | 3 | linear | – | no record (killed or hung) | – | hung at ~4.82M: instance 8033's Minecraft client crashed mid-reset and deadlocked with its built-in server (thread dump in launch_logs/hang_line5b/); stopped by hand | – |
| `20261119_161818` | continued | 4.8M → 5.0M | 3 | linear | 18.0 | completed | **38.6** @ 5.0M | resumed from the 4.82M checkpoint with --learning-rate 1.80082e-5 | [open](https://forge.coreweave.com/wandb/njdevstech0100-personal/craftground-chop-trees/runs/7ozb4sir) |
<!-- AUTO-GENERATED:END -->
