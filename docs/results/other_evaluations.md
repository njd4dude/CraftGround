# Other evaluations: different rules

Scores that don't compare with the three main pages because the rules differ: MineRL's own scoring (4,000 ticks, iron axe, ends at 64 logs), the iron axe, long episodes, random spawns instead of fixed spots, and the held-out world.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 20:59._

## Other episode rules

Evaluations under rules other than our standard ones: MineRL's own scoring (4,000 ticks, iron axe, ends at 64 logs), MineRL's iron axe, and long episodes. One table per **evaluation type** (world and spots, episode length, log cap, axe); only rows in the same table are comparable. **Spots played**: fixed spots are played in file order, so 50 = the first 50 of the 100. MineRL mode runs 4 ticks per decision, so a 1,000-tick episode is 250 of its decisions.

### Natural world, fixed spots in oak/birch forests · 4,000-tick episodes, ends at 64 logs · iron axe (MineRL's)

Scored the way the MineRL project scored these models (its 24 worlds, same rules), so **In MineRL** is directly comparable.

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | In MineRL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala BC→PPO | MineRL mode | 50 | **54.0** ± 5.6 | 64 | 3 | 64 | 59.6 |
| MineRL NatureCNN BC→PPO | MineRL mode | 50 | **48.4** ± 6.6 | 64 | 5 | 64 | 59.8 |
| MineRL Impala BC | MineRL mode | 50 | **44.1** ± 6.9 | 64 | 5 | 64 | 58.8 |

### Natural world, fixed spots in oak/birch forests · 1,000-tick episodes, no log cap · iron axe (MineRL's)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max |
| --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala BC→PPO | MineRL mode | 100 | **19.6** ± 1.3 | 21 | 4 | 29 |
| MineRL NatureCNN BC→PPO | MineRL mode | 100 | **17.5** ± 1.3 | 19 | 6 | 28 |
| MineRL Impala BC | MineRL mode | 100 | **16.3** ± 1.5 | 17 | 5 | 29 |
| Random actions | MineRL mode | 100 | **0.6** ± 0.2 | 0 | 63 | 4 |

### Natural world, fixed spots in oak/birch forests · 4,000-tick episodes, no log cap · one-hit axe (ours)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | vs Line 5, spot by spot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Line 5 @ 4.0M *(reference)* | ours (114×64, 1 tick) | 50 | **118.9** ± 26.0 | 112 | 2 | 517 | – |

### Natural world, fixed spots in oak/birch forests · 4,000-tick episodes, ends at 64 logs · one-hit axe (ours)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | vs Line 5, spot by spot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Line 5 @ 4.0M *(reference)* | ours (114×64, 1 tick) | 50 | **54.6** ± 5.2 | 64 | 2 | 64 | – |

### Arena · 1,000-tick episodes, ends at 32 logs (all of them) · iron axe (MineRL's)

| Model | View & actions | Episodes | Mean logs | Median | 0-log episodes | Max |
| --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala BC→PPO | MineRL mode | 50 | **21.4** ± 0.9 | 22 | 0 | 26 |
| MineRL NatureCNN BC→PPO | MineRL mode | 50 | **20.0** ± 0.9 | 21 | 0 | 25 |
| MineRL Impala BC | MineRL mode | 50 | **16.7** ± 2.5 | 21 | 1 | 28 |
| Random actions | MineRL mode | 50 | **0.2** ± 0.1 | 0 | 41 | 1 |

## Random spawns

Every episode starts at a new spot found by the same search training uses, so agents play different spots and each mean also moves by ±3-4 logs from spawn luck: compare agents on the fixed spots first ([oak and birch forests](natural_oak_birch_forests.md), [every biome with trees](natural_all_tree_biomes.md)). This checks that a fixed-spawn score holds on spots no fixed set contains. Run with `evaluate.py <model> --env natural --random-spots [--spawn wooded]`.

### Random spawns in oak/birch forests

Rows from before 2026-11-16 were made under an older setup (render distance 6, no damage). **Arena mean** is the same agent's score in the walled arena ([Arena](arena.md)).

| Model | Line | Episodes | Mean logs | 0-log episodes | Max | Arena mean | vs random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | random actions | 100 | **1.81** ± 0.59 | 43 | 24 | 3.00 | baseline |
| Line 5b @ 5.0M | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **45.32** ± 4.33 | 2 | 102 | – | 25.0× |
| Line 4d @ 4.0M | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **44.78** ± 5.42 | 4 | 107 | – | 24.7× |
| Line 5 @ 4.0M | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **38.76** ± 4.56 | 3 | 89 | – | 21.4× |
| Line 2b @ 1.0M | linear LR from Line 2's 601k checkpoint | 100 | **25.44** ± 3.68 | 8 | 66 | – | 14.1× |
| Line 2 @ 602k | jump, constant LR · render distance 6 | 100 | **21.67** ± 3.37 | 12 | 66 | – | 12.0× |
| Line 2 @ 502k | jump, constant LR · render distance 6 | 100 | **17.96** ± 2.80 | 16 | 63 | – | 9.9× |
| Line 2 @ 1.0M | jump, constant LR · render distance 6 | 100 | **14.99** ± 2.04 | 14 | 41 | – | 8.3× |
| Line 1 @ 503k | no jump, constant LR · render distance 6 | 100 | **9.68** ± 1.95 | 15 | 43 | – | 5.3× |
| `chop_ppo_20261113_135620` | trained in the arena | 100 | **5.10** ± 0.81 | 15 | 18 | 26.20 | 2.8× |
| `chop_ppo_20261113_131702` | trained in the arena | 100 | **5.01** ± 1.22 | 33 | 30 | 31.67 | 2.8× |

### Random spawns in every biome with trees

Any biome with trees but savanna, a new region every 5 episodes (as in wooded training).

| Model | Line | Episodes | Mean logs | 0-log episodes | Max |
| --- | --- | --- | --- | --- | --- |
| Line 5b @ 5.0M | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **42.77** ± 5.20 | 3 | 139 |
| Line 4d @ 4.0M | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **37.72** ± 7.69 | 13 | 186 |
| Line 5 @ 4.0M | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **35.01** ± 6.02 | 7 | 182 |

## Held-out world

Archived 2026-11-19: agents score the same in a world they never trained in. See [held-out world](archive/held_out.md).
<!-- AUTO-GENERATED:END -->
