# Natural world: Fixed forest spawns

[← Natural world index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-19 17:04._

Every agent here played the same 100 forest spots (`results/eval_spots_natural.json`, one episode each, same starting direction, current setup), so differences come from the agents, not from where they spawned. Best first. **± is the 95% range** of the mean: two agents whose ranges overlap a lot may not really differ. Older random-spawn scores are in [Random spawns](random_spawns.md).

| Model | Line | Episodes | Mean logs | Median | 0-log episodes | Max | vs random |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random actions (baseline) | random actions | 100 | **1.14** ± 0.41 | 0 | 65 | 11 | baseline |
| Line 4d @ 4.0M<br>`20261119_112830` | resume Line 4c @ 3.0M for 1M more, forest spawns, linear LR from 1e-4 | 100 | **41.27** ± 5.56 | 40 | 8 | 109 | **36.2×** |
| Line 4c @ 3.0M<br>`20261118_183414` | resume Line 4b @ 2.0M for 1M more, linear LR from 1e-4 | 100 | **41.18** ± 4.90 | 41 | 7 | 94 | **36.1×** |
| Line 5 @ 4.0M<br>`20261119_005447` | wooded spawns, from Line 4c @ 3.0M, 1M more, linear LR from 1e-4 | 100 | **39.99** ± 4.50 | 40 | 4 | 89 | **35.1×** |
| Line 5b @ 5.0M<br>`20261119_161818` | resume Line 5 @ 4.0M for 1M more, wooded spawns, linear LR from 1e-4 | 100 | **38.64** ± 4.61 | 39 | 6 | 94 | **33.9×** |
| Line 4b @ 2.0M<br>`20261118_162138` | resume Line 4 @ 1.0M for 1M more, linear LR from 1e-4 | 100 | **35.73** ± 4.68 | 37 | 10 | 97 | **31.3×** |
| Line 4 @ 1.0M<br>`20261118_124345` | multi actions (move x turn x look x jump, attack held), linear LR, from scratch | 100 | **26.60** ± 3.42 | 28 | 9 | 67 | **23.3×** |
| Line 2b @ 1.0M<br>`20261115_231007` | linear LR from Line 2's 601k checkpoint | 100 | **24.74** ± 3.73 | 23 | 13 | 69 | **21.7×** |
| Line 3c @ 3.0M<br>`20261117_114712` | 3M linear LR schedule, from Line 3 @ 257k | 100 | **23.93** ± 3.59 | 21 | 9 | 69 | **21.0×** |
| Line 3 @ 1.0M<br>`20261116_123155` | jump, linear LR, from scratch | 100 | **22.39** ± 2.97 | 23 | 6 | 53 | **19.6×** |
| Line 3b @ 1.5M<br>`20261116_145213` | fine-tune Line 3 @ 1.0M for 500k more at a small LR | 100 | **21.86** ± 3.08 | 20 | 14 | 53 | **19.2×** |
| Line 2 @ 602k<br>`20261115_185527` @ 601,536 steps | jump, constant LR · render distance 6 | 100 | **18.07** ± 2.95 | 16 | 17 | 59 | **15.9×** |
| Line 3c @ 1.0M<br>`20261116_173633` @ 996,932 steps | 3M linear LR schedule, from Line 3 @ 257k | 100 | **16.15** ± 2.12 | 15 | 10 | 47 | **14.2×** |
| Line 2 @ 1.0M<br>`20261115_192015` | jump, constant LR · render distance 6 | 100 | **14.98** ± 2.09 | 14 | 11 | 43 | **13.1×** |
| Line 3 @ 257k<br>`20261115_235545` | jump, linear LR, from scratch | 100 | **12.35** ± 1.95 | 11 | 10 | 37 | **10.8×** |
<!-- AUTO-GENERATED:END -->
