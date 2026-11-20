# Natural world: Fixed wooded spawns

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 18:03._

100 fixed spots in any biome with trees except savanna (`results/eval_spots_natural_wooded.json`, `make_eval_spots.py --spawn wooded`): snowy taiga, taiga, dark forest, jungle, swamp and more, same training world. **Forest mean** is the same agent's score on the [Fixed forest spawns](fixed_forest_spawns.md) page's oak/birch spots; the **gap** is how much it loses outside them. **0-log episodes** count the episodes that chopped nothing on each set (out of 100: the agent got stuck, lost or never found a tree). Run with `evaluate.py <model> --env natural --spots results/eval_spots_natural_wooded.json`.

| Model | Line | Episodes | Wooded mean | Forest mean | Gap | 0-log episodes (wooded) | 0-log episodes (forest) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Line 5 @ 4.0M<br>`20261119_005447` | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **31.64** ± 4.76 | 39.99 ± 4.50 | -8.35 | 9 | 4 |
| Line 5b @ 5.0M<br>`20261119_161818` | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **31.33** ± 4.81 | 38.64 ± 4.61 | -7.31 | 7 | 6 |
| Line 4c @ 3.0M<br>`20261118_183414` | resume Line 4b @ 2.0M for 1M more, linear LR from 1e-4 | 100 | **30.50** ± 5.22 | 41.18 ± 4.90 | -10.68 | 16 | 7 |
| Line 4d @ 4.0M<br>`20261119_112830` | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **25.66** ± 5.07 | 41.27 ± 5.56 | -15.61 | 18 | 8 |
<!-- AUTO-GENERATED:END -->
