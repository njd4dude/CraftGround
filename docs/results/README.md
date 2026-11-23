# Results

Every agent compared, one page per environment, plus evaluations under other rules.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-22 17:57._

- [Natural world: oak and birch forests (100 fixed spots)](natural_oak_birch_forests.md): The main comparison: every agent on the same 100 spots in the natural world's oak and birch forests (`results/eval_spots_natural.json`).
- [Natural world: every biome with trees (100 fixed spots)](natural_all_tree_biomes.md): Every agent on 100 spots spread over every biome with trees (taiga, snowy taiga, jungle, dark forest, swamp, cherry grove...; no savanna; `results/eval_spots_natural_wooded.json`), next to its oak/birch forest score.
- [Natural world: anywhere on land (100 fixed spots)](natural_anywhere.md): Every agent on 100 spots anywhere on land: plains, desert, savanna, mountains and forests alike, so many spots have no tree nearby (`results/eval_spots_natural_anywhere.json`), next to its oak/birch forest score.
- [Arena: walled flat arena with 8 oaks](arena.md): Every agent in the one arena: superflat, stone-brick walls, 8 oaks of 4 logs (32 logs at most), rebuilt every episode. Only the agents' actions differ between runs.
- [Other evaluations: different rules](other_evaluations.md): Scores that don't compare with the three main pages because the rules differ: MineRL's own scoring (4,000 ticks, iron axe, ends at 64 logs), the iron axe, long episodes, random spawns instead of fixed spots, and the held-out world.
- [Held-out world (archived)](archive/held_out.md): agents score the same in a world they never trained in (written by hand)
<!-- AUTO-GENERATED:END -->
