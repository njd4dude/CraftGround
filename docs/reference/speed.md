# Reference: environment speed

How many env steps per second the tree-chopping environments run at, measured by
`benchmark.py`. Rebuild these tables by running it again; each run adds rows.

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-17 14:05._

## How it was measured

`benchmark.py` starts N Minecraft instances side by side, plays 200 untimed steps per instance, then times 3,000 steps per instance with **random actions and no model**, so these are the environments' own speeds: the most that training or evaluation can reach. Episode resets happen during the timed steps just as in training and are included. **Reset share** is the part of instance time spent inside reset() (spawn search, terrain loading); since instances step in lockstep, a slow reset also stalls the others, so its real cost is higher. **RAM/instance** is the rise in used system RAM divided by N.

Machine: AMD Ryzen 7 7700X 8-Core Processor, 8 cores / 16 threads, 31 GB RAM

## Arena (walled superflat arena, `chop/envs/arena.py`)

Latest measurement per instance count.

| Instances | Steps/s total | Claimed in train.py | Per instance | vs 1 instance | Resets | Avg reset (s) | Min / median / max (s) | Reset share | RAM/instance (GB) | Launch (s) | Measured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **487** | 451 | 487 | 1.00× | 3 | 0.00 | – | 0% | 2.30 | 77 | 2026-11-14 18:05 |
| 2 | **713** | 714 | 357 | 1.46× | 6 | 0.00 | – | 0% | 2.40 | 93 | 2026-11-14 18:06 |
| 4 | **977** | 858 | 244 | 2.01× | 12 | 0.01 | – | 0% | 2.38 | 110 | 2026-11-14 18:08 |
| 6 | **1046** | – | 174 | 2.15× | 18 | 0.01 | – | 0% | 2.43 | 142 | 2026-11-14 18:10 |

## Natural (forest in a normal world, `chop/envs/natural.py`)

Latest measurement per instance count.

| Instances | Steps/s total | Per instance | vs 1 instance | Resets | Avg reset (s) | Min / median / max (s) | Reset share | RAM/instance (GB) | Launch (s) | Measured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **48** | 48 | 1.00× | 10 | 18.64 | 1.59 / 8.10 / 86.41 | 90% | 2.84 | 143 | 2026-11-15 14:45 |
| 2 | **120** | 60 | 2.50× | 20 | 8.89 | 1.61 / 6.68 / 43.33 | 53% | 2.82 | 152 | 2026-11-15 14:51 |
| 4 | **235** | 59 | 4.90× | 40 | 5.69 | 1.63 / 4.42 / 41.63 | 33% | 3.03 | 182 | 2026-11-15 15:38 |
| 6 | **263** | 44 | 5.48× | 18 | 7.83 | – | 34% | 2.84 | 640 | 2026-11-14 23:19 |

## All runs

Every measurement, newest first.

| When | World | Instances | Steps/s total | Per instance | Status |
| --- | --- | --- | --- | --- | --- |
| 2026-11-15 15:38 | natural | 4 | 235 | 59 | ok |
| 2026-11-15 14:56 | natural | 4 | 161 | 40 | ok |
| 2026-11-15 14:51 | natural | 2 | 120 | 60 | ok |
| 2026-11-15 14:45 | natural | 1 | 48 | 48 | ok |
| 2026-11-15 14:42 | natural | 1 | – | – | failed: RuntimeError |
| 2026-11-14 23:19 | natural | 6 | 263 | 44 | ok |
| 2026-11-14 23:04 | natural | 6 | – | – | failed: 1 of 6 instances did not start (Gradle: Failed to remap minecraft), rest hung (stopped by hand) |
| 2026-11-14 18:38 | natural | 6 | – | – | failed: 1 of 6 instances did not start, then the run hung in reset (stopped by hand) |
| 2026-11-14 18:34 | natural | 4 | 165 | 41 | ok |
| 2026-11-14 18:31 | natural | 2 | 101 | 50 | ok |
| 2026-11-14 18:27 | natural | 1 | 38 | 38 | ok |
| 2026-11-14 18:10 | arena | 6 | 1046 | 174 | ok |
| 2026-11-14 18:08 | arena | 4 | 977 | 244 | ok |
| 2026-11-14 18:06 | arena | 2 | 713 | 357 | ok |
| 2026-11-14 18:05 | arena | 1 | 487 | 487 | ok |
<!-- AUTO-GENERATED:END -->

## Notes

Write your observations here. This section is never overwritten.

### Natural: rows were measured with different forest-check timeouts (2026-11-15)

The natural table mixes two versions of the natural world (`natural_env.py` then, `chop/envs/natural.py` now), so its rows are not directly comparable:

| Row | `CHECK_TIMEOUT_S` | Measured |
| --- | --- | --- |
| 1, 2 instances | 3.0 s | 2026-11-15 14:45, 14:51 |
| 4 instances | **0.75 s** | 2026-11-15 15:38 |
| 6 instances | 3.0 s (older run, 3,000 steps) | 2026-11-14 23:19 |

The 4-instance row is faster (235 steps/s, against 161 for the same setup with 3.0 s at
2026-11-15 14:56) **only because of the lower timeout**. A spot that fails the forest check gets
no reply, so each miss waits out the timeout; cutting it from 3.0 s to 0.75 s cut the cost of a miss
from ~3.1 s to ~1.0 s, and the median reset from 7.3 s to 4.4 s. Nothing else changed. For the same
reason the "vs 1 instance" column overstates the 4-instance speedup. The per-try pass rate stayed
the same (26% → 24%, from `results/benchmark_resets.csv`), so the shorter timeout is not visibly
rejecting real forests. Rerun 1, 2 and 6 instances with 0.75 s to make the table consistent.
