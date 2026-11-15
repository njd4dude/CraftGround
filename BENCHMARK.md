# CraftGround Benchmark: Environment Speed

How many env steps per second the tree-chopping environments run at, measured by
`benchmark.py`. Rebuild these tables by running it again; each run adds rows.

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-14 23:31._

## How it was measured

`benchmark.py` starts N Minecraft instances side by side, plays 200 untimed steps per instance, then times 3,000 steps per instance with **random actions and no model**, so these are the environments' own speeds: the most that training or evaluation can reach. Episode resets happen during the timed steps just as in training and are included. **Reset share** is the part of instance time spent inside reset() (spawn search, terrain loading); since instances step in lockstep, a slow reset also stalls the others, so its real cost is higher. **RAM/instance** is the rise in used system RAM divided by N.

Machine: AMD Ryzen 7 7700X 8-Core Processor, 8 cores / 16 threads, 31 GB RAM

## Arena (walled superflat arena, `tree_env.py`)

Latest measurement per instance count.

| Instances | Steps/s total | Claimed in train.py | Per instance | vs 1 instance | Resets | Avg reset (s) | Reset share | RAM/instance (GB) | Launch (s) | Measured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **487** | 451 | 487 | 1.00× | 3 | 0.00 | 0% | 2.30 | 77 | 2026-11-14 18:05 |
| 2 | **713** | 714 | 357 | 1.46× | 6 | 0.00 | 0% | 2.40 | 93 | 2026-11-14 18:06 |
| 4 | **977** | 858 | 244 | 2.01× | 12 | 0.01 | 0% | 2.38 | 110 | 2026-11-14 18:08 |
| 6 | **1046** | – | 174 | 2.15× | 18 | 0.01 | 0% | 2.43 | 142 | 2026-11-14 18:10 |

## Natural (forest in a normal world, `natural_env.py`)

Latest measurement per instance count.

| Instances | Steps/s total | Per instance | vs 1 instance | Resets | Avg reset (s) | Reset share | RAM/instance (GB) | Launch (s) | Measured |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **38** | 38 | 1.00× | 3 | 24.42 | 93% | 2.93 | 117 | 2026-11-14 18:27 |
| 2 | **101** | 50 | 2.66× | 6 | 10.15 | 51% | 2.94 | 152 | 2026-11-14 18:31 |
| 4 | **165** | 41 | 4.34× | 12 | 8.96 | 37% | 2.20 | 140 | 2026-11-14 18:34 |
| 6 | **263** | 44 | 6.92× | 18 | 7.83 | 34% | 2.84 | 640 | 2026-11-14 23:19 |

## All runs

Every measurement, newest first.

| When | World | Instances | Steps/s total | Per instance | Status |
| --- | --- | --- | --- | --- | --- |
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
