# Results: held-out world (archived)

**Archived 2026-11-19, no longer updated.** The question is answered: agents score the same in a
world they never trained in (gaps −0.5 to +3.9, all within noise), so they learned to chop trees,
not one map. What still costs score is new *biomes*, not new worlds ([every biome with trees](../natural_all_tree_biomes.md)).
The check still works for a final model: `make_eval_spots.py --held-out` (already done) and
`evaluate.py <model> --env natural --held-out`; the results go to `results/eval_results_natural.csv`.

[← Other evaluations](../other_evaluations.md)

_Tables last updated 2026-11-19 13:48._

Fixed spots in a world training never uses (seed 20261117, `results/eval_spots_natural_seed20261117.json`). **Training world** is the same agent's fixed-spot score in the world it trained in. A **gap** near 0 means the agent learned to chop trees in general; a clearly negative gap (larger than the ± ranges) means part of its score comes from knowing the training world. vs random uses the random baseline in this world.

| Model | Line | Episodes | Held-out mean | Training world | Gap | vs random |
| --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | random actions | 100 | **1.14** ± 0.38 | 1.14 ± 0.41 | +0.00 | baseline |
| Line 4b @ 2.0M<br>`20261118_162138` | resume Line 4 @ 1.0M for 1M more, linear LR from 1e-4 | 100 | **39.58** ± 4.72 | 35.73 ± 4.68 | +3.85 | **34.7×** |
| Line 2b @ 1.0M<br>`20261115_231007` | linear LR from Line 2's 601k checkpoint | 100 | **25.42** ± 3.96 | 24.74 ± 3.73 | +0.68 | **22.3×** |
| Line 3 @ 1.0M<br>`20261116_123155` | jump, linear LR, from scratch | 100 | **21.93** ± 3.25 | 22.39 ± 2.97 | -0.46 | **19.2×** |
