# MineRL in CraftGround: Results

[← MineRL in CraftGround index](README.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-20 14:55._

One table per **evaluation type**: the world and spots, episode length, log cap and axe. Only rows in the same table are comparable. Logs per episode; **±** is the 95% range of the mean. **vs Line 5** is the average difference per fixed spot over the spots both played, with its 95% range (a range that doesn't include 0 is a real difference). Our agents' 1,000-tick rows are their standard evaluations from the natural-world report (MineRL rows of that type are also shown there). **Spots played**: fixed spots are played in file order, so 50 = the first 50 of the 100; means over 50 and 100 spots can differ by which spots they include, so compare those rows with the spot-by-spot column, which only uses shared spots. MineRL mode runs 4 ticks per decision, so a 1,000-tick episode is 250 of its decisions.

### Natural world, fixed forest spots · 4,000-tick episodes, ends at 64 logs · iron axe (MineRL's)

Scored the way the MineRL project scored these models (its 24 worlds, same rules), so **In MineRL** is directly comparable.

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | In MineRL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO | MineRL mode | 50 | **54.0** ± 5.6 | 64 | 3 | 64 | 59.6 |
| MineRL NatureCNN PPO | MineRL mode | 50 | **48.4** ± 6.6 | 64 | 5 | 64 | 59.8 |
| MineRL Impala BC | MineRL mode | 50 | **44.1** ± 6.9 | 64 | 5 | 64 | 58.8 |

### Natural world, fixed forest spots · 1,000-tick episodes, no log cap · one-hit axe (ours)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | vs Line 5, spot by spot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO + fine-tune @ 252k | MineRL mode | 100 | **43.4** ± 3.2 | 46 | 2 | 71 | +3.4 ± 4.0 (51 better / 43 worse, 100 spots) |
| Line 4d @ 4.0M | ours (114×64, 1 tick) | 100 | **41.3** ± 5.6 | 40 | 8 | 109 | +1.3 ± 5.0 (50 better / 44 worse, 100 spots) |
| Line 4c @ 3.0M | ours (114×64, 1 tick) | 100 | **41.2** ± 4.9 | 41 | 7 | 94 | +1.2 ± 4.7 (49 better / 48 worse, 100 spots) |
| Line 5 @ 4.0M *(reference)* | ours (114×64, 1 tick) | 100 | **40.0** ± 4.5 | 40 | 4 | 89 | – |
| MineRL Impala PPO + fine-tune @ 120k | MineRL mode | 100 | **38.5** ± 3.0 | 42 | 3 | 68 | -1.5 ± 3.7 (46 better / 48 worse, 100 spots) |
| MineRL Impala PPO | MineRL mode | 100 | **29.4** ± 2.4 | 32 | 3 | 49 | -10.6 ± 3.8 (28 better / 67 worse, 100 spots) |
| MineRL NatureCNN PPO | MineRL mode | 100 | **24.3** ± 2.4 | 26 | 5 | 46 | -15.7 ± 4.0 (20 better / 74 worse, 100 spots) |

### Natural world, fixed forest spots · 1,000-tick episodes, no log cap · iron axe (MineRL's)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max |
| --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO | MineRL mode | 100 | **19.6** ± 1.3 | 21 | 4 | 29 |
| MineRL NatureCNN PPO | MineRL mode | 100 | **17.5** ± 1.3 | 19 | 6 | 28 |
| MineRL Impala BC | MineRL mode | 100 | **16.3** ± 1.5 | 17 | 5 | 29 |
| Random actions | MineRL mode | 100 | **0.6** ± 0.2 | 0 | 63 | 4 |

### Natural world, fixed forest spots · 4,000-tick episodes, no log cap · one-hit axe (ours)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | vs Line 5, spot by spot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Line 5 @ 4.0M *(reference)* | ours (114×64, 1 tick) | 50 | **118.9** ± 26.0 | 112 | 2 | 517 | – |

### Natural world, fixed forest spots · 4,000-tick episodes, ends at 64 logs · one-hit axe (ours)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | vs Line 5, spot by spot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Line 5 @ 4.0M *(reference)* | ours (114×64, 1 tick) | 50 | **54.6** ± 5.2 | 64 | 2 | 64 | – |

### Natural world, fixed wooded spots · 1,000-tick episodes, no log cap · one-hit axe (ours)

| Model | View & actions | Spots played | Mean logs | Median | 0-log episodes | Max | vs Line 5, spot by spot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO + fine-tune @ 252k | MineRL mode | 100 | **37.2** ± 4.5 | 39 | 5 | 134 | +5.6 ± 3.8 (56 better / 41 worse, 100 spots) |
| Line 5 @ 4.0M *(reference)* | ours (114×64, 1 tick) | 100 | **31.6** ± 4.8 | 31 | 9 | 101 | – |
| Line 4c @ 3.0M | ours (114×64, 1 tick) | 100 | **30.5** ± 5.2 | 26 | 16 | 88 | -1.1 ± 4.5 (43 better / 51 worse, 100 spots) |
| Line 4d @ 4.0M | ours (114×64, 1 tick) | 100 | **25.7** ± 5.1 | 19 | 18 | 102 | -6.0 ± 4.8 (32 better / 58 worse, 100 spots) |

### Arena · 1,000-tick episodes, ends at 32 logs (all of them) · one-hit axe (ours)

| Model | View & actions | Episodes | Mean logs | Median | 0-log episodes | Max |
| --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO | MineRL mode | 50 | **27.8** ± 1.6 | 30 | 0 | 32 |
| MineRL NatureCNN PPO | MineRL mode | 50 | **26.4** ± 0.9 | 27 | 0 | 30 |

### Arena · 1,000-tick episodes, ends at 32 logs (all of them) · iron axe (MineRL's)

| Model | View & actions | Episodes | Mean logs | Median | 0-log episodes | Max |
| --- | --- | --- | --- | --- | --- | --- |
| MineRL Impala PPO | MineRL mode | 50 | **21.4** ± 0.9 | 22 | 0 | 26 |
| MineRL NatureCNN PPO | MineRL mode | 50 | **20.0** ± 0.9 | 21 | 0 | 25 |
| MineRL Impala BC | MineRL mode | 50 | **16.7** ± 2.5 | 21 | 1 | 28 |
| Random actions | MineRL mode | 50 | **0.2** ± 0.1 | 0 | 41 | 1 |
<!-- AUTO-GENERATED:END -->
