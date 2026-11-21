# Natural world: Environment setup

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 14:34._

| Setting | Value |
| --- | --- |
| World | Normal generated world, fixed seed 20261114 (held-out evaluation world: seed 20261117, never trained in) |
| Evaluation | 100 fixed forest spots (`results/eval_spots_natural.json`, `make_eval_spots.py`); `evaluate.py --random-spots` for random spawns. Held-out world (archived: [archive/held_out.md](archive/held_out.md)): `evaluate.py <model> --env natural --held-out`. Wooded spawns: see [Fixed wooded spawns](fixed_wooded_spawns.md) |
| Spawn | A new spot every episode, on the ground in an oak/birch forest: first within 150 blocks of the last spot, else anywhere within ±25,000. `train.py --spawn wooded` (Line 5, since 2026-11-18): any biome with trees but savanna, and a world-wide search after 4 episodes near one spot |
| Rebuilt each episode | no: chopped trees stay chopped, so every episode moves to fresh forest |
| Boundaries | none; terrain loads as the agent walks |
| Render distance | 4 chunks, Minecraft heap 2G (since 2026-11-15, to fit 4 instances in memory; earlier runs: 6 chunks, 3G) |
| Tool | Unbreakable diamond axe, Efficiency X (1 hit per log) |
| Observation | RGB image 114×64 (same as the arena) |
| Actions | 13 discrete: the arena's 11 plus jump, forward_jump (since 2026-11-15; earlier runs and the random baseline use the arena's 11) |
| Reward | +1 per log chopped (any wood type) |
| Damage | Lava, fire, drowning etc. hurt and dying ends the episode; no fall damage (since 2026-11-16; earlier runs: player immune to all damage) |
| Episode length | 1000 steps (no log cap), or until the agent dies |
<!-- AUTO-GENERATED:END -->
