# Natural world: Earlier random-spawn evaluations

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 13:52._

Historical: every natural-world evaluation before the fixed spots (2026-11-16) spawned in random forests, so these means also move by ±3-4 logs from spawn luck, and most were made under an older setup (render distance 6, no damage). Use [Forest spots](forest_spots.md) to compare agents. **Arena mean** is the same agent's score in the walled arena ([arena/](../arena/README.md)).

| Model | Line | Arena mean | Episodes | Mean logs | Min | Max | vs random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | random actions | 3.00 | 100 | 1.81 | 0 | 24 | baseline |
| `chop_ppo_20261113_131702` | trained in the arena | 31.67 | 100 | 5.01 | 0 | 30 | 2.8× |
| `chop_ppo_20261113_135620` | trained in the arena | 26.20 | 100 | 5.10 | 0 | 18 | 2.8× |
| Line 1 @ 503k | no jump, constant LR · render distance 6 | – | 100 | 9.68 | 0 | 43 | 5.3× |
| Line 2 @ 502k | jump, constant LR · render distance 6 | – | 100 | 17.96 | 0 | 63 | 9.9× |
| Line 2 @ 602k | jump, constant LR · render distance 6 | – | 100 | 21.67 | 0 | 66 | 12.0× |
| Line 2 @ 1.0M | jump, constant LR · render distance 6 | – | 100 | 14.99 | 0 | 41 | 8.3× |
| Line 2b @ 1.0M | linear LR from Line 2's 601k checkpoint | – | 100 | 25.44 | 0 | 66 | 14.1× |
<!-- AUTO-GENERATED:END -->
