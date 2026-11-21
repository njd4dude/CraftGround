# Results: forest spots

The main comparison: every agent on the same 100 oak/birch forest spots in the natural world.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-21 14:49._

## Results

Every agent here played the same 100 forest spots (`results/eval_spots_natural.json`, one episode each, same starting direction, current setup), so differences come from the agents, not from where they spawned. Best first. **± is the 95% range** of the mean: two agents whose ranges overlap a lot may not really differ. **vs Line 5** is the average difference per spot with its 95% range; a range that doesn't include 0 is a real difference. MineRL rows play in MineRL mode (their own view and actions) under the same rules. Random-spawn scores are in [Other rules](other_rules.md#random-spawns).

| Model | Line | Episodes | Mean logs | Median | 0-log episodes | Max | vs Line 5 | vs random |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | random actions | 100 | **1.14** ± 0.41 | 0 | 65 | 11 | -38.9 ± 4.4 | baseline |
| MineRL Impala PPO + fine-tune @ 2.0M ticks<br>`ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` @ 615,292 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 492,412 decisions = 1.97M ticks, from MineRL Impala PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **44.66** ± 3.73 | 49 | 2 | 78 | +4.7 ± 3.9 | **39.2×** |
| MineRL Impala PPO + fine-tune @ 1.5M ticks<br>`ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` @ 495,292 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 372,412 decisions = 1.49M ticks, from MineRL Impala PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **44.54** ± 3.61 | 47 | 3 | 85 | +4.5 ± 4.2 | **39.1×** |
| MineRL Impala PPO + fine-tune @ 1.0M ticks<br>`ft_impala-ppo_wooded_one-hit-axe_20261120_124557` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 252,412 decisions = 1.01M ticks, from MineRL Impala PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **43.35** ± 3.16 | 46 | 2 | 71 | +3.4 ± 4.0 | **38.0×** |
| Line 4d @ 4.0M<br>`20261119_112830` | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **41.27** ± 5.56 | 40 | 8 | 109 | +1.3 ± 5.0 | **36.2×** |
| Line 4c @ 3.0M<br>`20261118_183414` | resume Line 4b @ 2.0M for 1M more, linear LR from 1e-4 | 100 | **41.18** ± 4.90 | 41 | 7 | 94 | +1.2 ± 4.7 | **36.1×** |
| MineRL Impala BC + fine-tune @ 1.0M ticks<br>`ft_impala-bc_wooded_one-hit-axe_20261120_220917` | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 253,692 decisions = 1.01M ticks, from MineRL Impala BC (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **40.43** ± 3.14 | 43 | 3 | 68 | +0.4 ± 3.5 | **35.5×** |
| Line 5 @ 4.0M<br>`20261119_005447` | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **39.99** ± 4.50 | 40 | 4 | 89 | reference | **35.1×** |
| Line 5b @ 5.0M<br>`20261119_161818` | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **38.64** ± 4.61 | 39 | 6 | 94 | -1.4 ± 4.5 | **33.9×** |
| MineRL Impala PPO + fine-tune @ 480k ticks<br>`ft_impala-ppo_wooded_one-hit-axe_20261120_124557` @ 242,874 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 119,994 decisions = 0.48M ticks, from MineRL Impala PPO (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **38.48** ± 3.03 | 42 | 3 | 68 | -1.5 ± 3.7 | **33.8×** |
| MineRL Impala BC + fine-tune @ 480k ticks<br>`ft_impala-bc_wooded_one-hit-axe_20261120_193516` @ 120,000 steps | **MineRL mode**, trained here (natural, wooded spawns, one-hit axe) for 120,000 decisions = 0.48M ticks, from MineRL Impala BC (see [MineRL transfer](../experiments/minerl_transfer.md)) | 100 | **37.17** ± 2.95 | 38 | 4 | 60 | -2.8 ± 3.7 | **32.6×** |
| Line 4b @ 2.0M<br>`20261118_162138` | resume Line 4 @ 1.0M for 1M more, linear LR from 1e-4 | 100 | **35.73** ± 4.68 | 37 | 10 | 97 | -4.3 ± 4.3 | **31.3×** |
| MineRL Impala PPO<br>`model_20261112_130747_ppo_bc_impala_all_kl05` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL transfer](../experiments/minerl_transfer.md)): ImpalaCNN BC on all VPT versions, then 120k PPO steps (target_kl 0.05) | 100 | **29.41** ± 2.44 | 32 | 3 | 49 | -10.6 ± 3.8 | **25.8×** |
| Line 4 @ 1.0M<br>`20261118_124345` | multi actions (move x turn x look x jump, attack held), linear LR, from scratch | 100 | **26.60** ± 3.42 | 28 | 9 | 67 | -13.4 ± 4.0 | **23.3×** |
| Line 2b @ 1.0M<br>`20261115_231007` | linear LR from Line 2's 601k checkpoint | 100 | **24.74** ± 3.73 | 23 | 13 | 69 | -15.2 ± 3.8 | **21.7×** |
| MineRL NatureCNN PPO<br>`model_20261111_163211_ppo_bc_6xx-1762_8xx-1957_plus20_kl05` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL transfer](../experiments/minerl_transfer.md)): NatureCNN BC on 6.x + 8.x VPT (40 epochs), then 120k PPO steps (target_kl 0.05) | 100 | **24.31** ± 2.44 | 26 | 5 | 46 | -15.7 ± 4.0 | **21.3×** |
| Line 3c @ 3.0M<br>`20261117_114712` | 3M linear LR schedule, from Line 3 @ 257k | 100 | **23.93** ± 3.59 | 21 | 9 | 69 | -16.1 ± 4.1 | **21.0×** |
| Line 3 @ 1.0M<br>`20261116_123155` | jump, linear LR, from scratch | 100 | **22.39** ± 2.97 | 23 | 6 | 53 | -17.6 ± 3.6 | **19.6×** |
| MineRL Impala BC<br>`model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont` | **MineRL mode** (64×64 view, 4 ticks per decision; see [MineRL transfer](../experiments/minerl_transfer.md)): ImpalaCNN behavioural cloning on all VPT versions, no RL (the Impala PPO's starting point) | 100 | **22.18** ± 2.36 | 24 | 6 | 43 | -17.8 ± 3.8 | **19.5×** |
| Line 3b @ 1.5M<br>`20261116_145213` | fine-tune Line 3 @ 1.0M for 500k more at a small LR | 100 | **21.86** ± 3.08 | 20 | 14 | 53 | -18.1 ± 3.5 | **19.2×** |
| Line 2 @ 602k<br>`20261115_185527` @ 601,536 steps | jump, constant LR · render distance 6 | 100 | **18.07** ± 2.95 | 16 | 17 | 59 | -21.9 ± 3.8 | **15.9×** |
| Line 3c @ 1.0M<br>`20261116_173633` @ 996,932 steps | 3M linear LR schedule, from Line 3 @ 257k | 100 | **16.15** ± 2.12 | 15 | 10 | 47 | -23.8 ± 3.8 | **14.2×** |
| Line 2 @ 1.0M<br>`20261115_192015` | jump, constant LR · render distance 6 | 100 | **14.98** ± 2.09 | 14 | 11 | 43 | -25.0 ± 4.0 | **13.1×** |
| Line 3 @ 257k<br>`20261115_235545` | jump, linear LR, from scratch | 100 | **12.35** ± 1.95 | 11 | 10 | 37 | -27.6 ± 3.9 | **10.8×** |
<!-- AUTO-GENERATED:END -->
