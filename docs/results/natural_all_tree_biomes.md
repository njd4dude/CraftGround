# Natural world: every biome with trees (100 fixed spots)

Every agent on 100 spots spread over every biome with trees (taiga, snowy taiga, jungle, dark forest, swamp, cherry grove...; no savanna; `results/eval_spots_natural_wooded.json`), next to its oak/birch forest score.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-22 20:35._

## Results

100 fixed spots in every biome with trees except savanna (`results/eval_spots_natural_wooded.json`, made with `make_eval_spots.py --spawn wooded`): snowy taiga, taiga, dark forest, jungle, swamp and more, same training world. **Oak/birch mean** is the same agent's score on the [oak and birch forests](natural_oak_birch_forests.md) page; the **gap** is how much it loses outside them. **0-log episodes** count the episodes that chopped nothing on each set (out of 100: the agent got stuck, lost or never found a tree). Run with `evaluate.py <model> --env natural --spots results/eval_spots_natural_wooded.json`.

| Model | Line | Episodes | Mean logs | vs Line 5 | Oak/birch mean | Gap | 0-log episodes (here) | 0-log episodes (oak/birch) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Impala BC→PPO + CG 1.5M<br>`ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` @ 495,292 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 372,412 decisions = 1.49M ticks, from Impala BC→PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **39.01** ± 4.77 | +7.4 ± 4.0 | 44.54 ± 3.61 | -5.53 | 7 | 3 |
| Impala BC→PPO + CG 2.0M<br>`ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` @ 615,292 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 492,412 decisions = 1.97M ticks, from Impala BC→PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **38.54** ± 4.73 | +6.9 ± 4.0 | 44.66 ± 3.73 | -6.12 | 5 | 2 |
| Impala BC + CG 1.0M<br>`ft_impala-bc_wooded_one-hit-axe_20261120_220917` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 253,692 decisions = 1.01M ticks, from Impala BC (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **37.48** ± 4.16 | +5.8 ± 3.7 | 40.43 ± 3.14 | -2.95 | 6 | 3 |
| Impala BC→PPO + CG 1.0M<br>`ft_impala-ppo_wooded_one-hit-axe_20261120_124557` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 252,412 decisions = 1.01M ticks, from Impala BC→PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **37.22** ± 4.48 | +5.6 ± 3.8 | 43.35 ± 3.16 | -6.13 | 5 | 2 |
| Line 5c @ 5.0M<br>`20261122_180159` | resume Line 5 @ 4.0M for 1M more, spawns anywhere on land, linear LR from 1e-4 | 100 | **35.37** ± 4.90 | +3.7 ± 3.5 | 36.59 ± 3.95 | -1.22 | 7 | 6 |
| Line 5 @ 4.0M<br>`20261119_005447` | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **31.64** ± 4.76 | reference | 39.99 ± 4.50 | -8.35 | 9 | 4 |
| Line 5b @ 5.0M<br>`20261119_161818` | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **31.33** ± 4.81 | -0.3 ± 4.5 | 38.64 ± 4.61 | -7.31 | 7 | 6 |
| Line 4c @ 3.0M<br>`20261118_183414` | resume Line 4b @ 2.0M for 1M more, linear LR from 1e-4 | 100 | **30.50** ± 5.22 | -1.1 ± 4.5 | 41.18 ± 4.90 | -10.68 | 16 | 7 |
| Impala BC→PPO, zero-shot<br>`model_20261112_130747_ppo_bc_impala_all_kl05` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL transfer](../experiments/minerl_transfer.md)): ImpalaCNN BC on all VPT versions, then 120k PPO steps (target_kl 0.05) | 100 | **25.72** ± 2.84 | -5.9 ± 3.6 | 29.41 ± 2.44 | -3.69 | 6 | 3 |
| Line 4d @ 4.0M<br>`20261119_112830` | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **25.66** ± 5.07 | -6.0 ± 4.8 | 41.27 ± 5.56 | -15.61 | 18 | 8 |
| Impala BC, zero-shot<br>`model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL transfer](../experiments/minerl_transfer.md)): ImpalaCNN behavioural cloning on all VPT versions, no RL (the Impala BC→PPO's starting point) | 100 | **21.15** ± 2.67 | -10.5 ± 3.7 | 22.18 ± 2.36 | -1.03 | 8 | 6 |
| Scratch Impala (MineRL mode) + CG 1.0M, Line 4 PPO settings<br>`scratch_impala_rl-settings_20261121_231545` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 250,720 decisions = 1.00M ticks, from scratch (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **18.45** ± 2.63 | -13.2 ± 3.8 | 18.95 ± 1.89 | -0.50 | 8 | 7 |
| Scratch Impala (MineRL mode) + CG 1.0M, gentle PPO settings<br>`scratch_impala_20261121_193751` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 252,282 decisions = 1.01M ticks, from scratch (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **14.28** ± 2.05 | -17.4 ± 3.8 | 14.57 ± 1.38 | -0.29 | 11 | 5 |
<!-- AUTO-GENERATED:END -->
