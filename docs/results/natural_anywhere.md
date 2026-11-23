# Natural world: anywhere on land (100 fixed spots)

Every agent on 100 spots anywhere on land: plains, desert, savanna, mountains and forests alike, so many spots have no tree nearby (`results/eval_spots_natural_anywhere.json`), next to its oak/birch forest score.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-23 01:12._

## Results

100 fixed spots anywhere on land (`results/eval_spots_natural_anywhere.json`, made with `make_eval_spots.py --spawn anywhere`): plains, desert, savanna, mountains and forests alike, so many spots have no tree nearby; same training world. **Oak/birch mean** is the same agent's score on the [oak and birch forests](natural_oak_birch_forests.md) page; the **gap** is how much it loses outside them. **0-log episodes** count the episodes that chopped nothing on each set (out of 100: the agent got stuck, lost or never found a tree). Run with `evaluate.py <model> --env natural --spots results/eval_spots_natural_anywhere.json`.

| Model | Line | Episodes | Mean logs | vs Line 5 | Oak/birch mean | Gap | 0-log episodes (here) | 0-log episodes (oak/birch) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Line 5d @ 5.0M<br>`20261122_223446` | resume Line 5 @ 4.0M for 1M more, wooded spawns, stuck penalty 0.01 per stuck tick, linear LR from 1e-4 | 100 | **23.38** ± 5.76 | +1.8 ± 3.7 | 41.97 ± 4.99 | -18.59 | 40 | 8 |
| Line 5 @ 4.0M<br>`20261119_005447` | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **21.56** ± 5.07 | reference | 39.99 ± 4.50 | -18.43 | 36 | 4 |
| Line 5b @ 5.0M<br>`20261119_161818` | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 200 | **20.96** ± 3.43 | -0.6 ± 2.5 | 38.70 ± 3.22 | -17.74 | 70 | 11 |
| Line 5c @ 5.0M<br>`20261122_180159` | resume Line 5 @ 4.0M for 1M more, spawns anywhere on land, linear LR from 1e-4 | 100 | **20.43** ± 5.73 | -1.1 ± 4.3 | 36.59 ± 3.95 | -16.16 | 36 | 6 |
<!-- AUTO-GENERATED:END -->
