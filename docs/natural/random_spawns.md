# Natural world: Random spawns

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 17:04._

Every episode starts at a new spot found by the same search training uses, so agents play different spots and each mean also moves by ±3-4 logs from spawn luck: compare agents on the fixed spawns first ([Fixed forest spawns](fixed_forest_spawns.md), [Fixed wooded spawns](fixed_wooded_spawns.md)). This checks that a fixed-spawn score holds on spots no fixed set contains. Run with `evaluate.py <model> --env natural --random-spots [--spawn wooded]`.

### Random forest spawns

Oak/birch forests. Rows from before 2026-11-16 were made under an older setup (render distance 6, no damage). **Arena mean** is the same agent's score in the walled arena ([arena/](../arena/README.md)).

| Model | Line | Episodes | Mean logs | 0-log episodes | Max | Arena mean | vs random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | random actions | 100 | **1.81** ± 0.59 | 43 | 24 | 3.00 | baseline |
| Line 2b @ 1.0M | linear LR from Line 2's 601k checkpoint | 100 | **25.44** ± 3.68 | 8 | 66 | – | 14.1× |
| Line 2 @ 602k | jump, constant LR · render distance 6 | 100 | **21.67** ± 3.37 | 12 | 66 | – | 12.0× |
| Line 2 @ 502k | jump, constant LR · render distance 6 | 100 | **17.96** ± 2.80 | 16 | 63 | – | 9.9× |
| Line 2 @ 1.0M | jump, constant LR · render distance 6 | 100 | **14.99** ± 2.04 | 14 | 41 | – | 8.3× |
| Line 1 @ 503k | no jump, constant LR · render distance 6 | 100 | **9.68** ± 1.95 | 15 | 43 | – | 5.3× |
| `chop_ppo_20261113_135620` | trained in the arena | 100 | **5.10** ± 0.81 | 15 | 18 | 26.20 | 2.8× |
| `chop_ppo_20261113_131702` | trained in the arena | 100 | **5.01** ± 1.22 | 33 | 30 | 31.67 | 2.8× |
<!-- AUTO-GENERATED:END -->
