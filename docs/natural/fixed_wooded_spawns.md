# Natural world: Fixed wooded spawns

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 14:25._

100 fixed spots in any biome with trees except savanna (`results/eval_spots_natural_wooded.json`, `make_eval_spots.py --spawn wooded`): snowy taiga, taiga, dark forest, jungle, swamp and more, same training world. **Forest mean** is the same agent's score on the [Fixed forest spawns](fixed_forest_spawns.md) page's oak/birch spots; the **gap** is how much it loses outside them. **0-log episodes** count the episodes that chopped nothing on each set (out of 100: the agent got stuck, lost or never found a tree). Run with `evaluate.py <model> --env natural --spots results/eval_spots_natural_wooded.json`.

| Model | Line | Episodes | Wooded mean | Forest mean | Gap | 0-log episodes (wooded) | 0-log episodes (forest) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO + fine-tune @ 2.0M ticks<br>`ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` @ 615,292 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 492,412 decisions = 1.97M ticks, from MineRL Impala PPO (see [MineRL](../minerl/README.md)) | 100 | **38.54** ± 4.73 | 44.66 ± 3.73 | -6.12 | 5 | 2 |
| MineRL Impala BC + fine-tune @ 1.0M ticks<br>`ft_impala-bc_wooded_one-hit-axe_20261120_220917` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 253,692 decisions = 1.01M ticks, from MineRL Impala BC (see [MineRL](../minerl/README.md)) | 100 | **37.48** ± 4.16 | 40.43 ± 3.14 | -2.95 | 6 | 3 |
| MineRL Impala PPO + fine-tune @ 1.0M ticks<br>`ft_impala-ppo_wooded_one-hit-axe_20261120_124557` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 252,412 decisions = 1.01M ticks, from MineRL Impala PPO (see [MineRL](../minerl/README.md)) | 100 | **37.22** ± 4.48 | 43.35 ± 3.16 | -6.13 | 5 | 2 |
| Line 5 @ 4.0M<br>`20261119_005447` | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **31.64** ± 4.76 | 39.99 ± 4.50 | -8.35 | 9 | 4 |
| Line 5b @ 5.0M<br>`20261119_161818` | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **31.33** ± 4.81 | 38.64 ± 4.61 | -7.31 | 7 | 6 |
| Line 4c @ 3.0M<br>`20261118_183414` | resume Line 4b @ 2.0M for 1M more, linear LR from 1e-4 | 100 | **30.50** ± 5.22 | 41.18 ± 4.90 | -10.68 | 16 | 7 |
| MineRL Impala PPO<br>`model_20261112_130747_ppo_bc_impala_all_kl05` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL](../minerl/README.md)): ImpalaCNN BC on all VPT versions, then 120k PPO steps (target_kl 0.05) | 100 | **25.72** ± 2.84 | 29.41 ± 2.44 | -3.69 | 6 | 3 |
| Line 4d @ 4.0M<br>`20261119_112830` | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **25.66** ± 5.07 | 41.27 ± 5.56 | -15.61 | 18 | 8 |
| MineRL Impala BC<br>`model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL](../minerl/README.md)): ImpalaCNN behavioural cloning on all VPT versions, no RL (the Impala PPO's starting point) | 100 | **21.15** ± 2.67 | 22.18 ± 2.36 | -1.03 | 8 | 6 |
<!-- AUTO-GENERATED:END -->
