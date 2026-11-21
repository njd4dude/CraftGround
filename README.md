# CraftGround: chopping trees with RL

Reinforcement learning (PPO, Stable-Baselines3) agents that chop as many logs as they can in one
Minecraft episode (1,000 game ticks, ~50 s), run in [CraftGround](https://github.com/yhs0602/CraftGround).
Two worlds: a small walled **arena** with 8 oaks, and a **natural** generated world where every
episode starts in a different forest. Models trained in a separate MineRL project can also run
here unchanged, in **MineRL mode** (their own 64×64 view and actions).

## Best agents so far

Logs per episode on the 100 fixed forest spots / 100 fixed wooded spots (natural world):

| Agent | Forest | Wooded |
| --- | --- | --- |
| MineRL Impala PPO, fine-tuned here for 2M ticks | **44.7** | **38.5** |
| MineRL Impala BC, fine-tuned here for 1M ticks | 40.4 | 37.5 |
| Line 5: pure RL here, 4M ticks | 40.0 | 31.6 |

What's behind these, and everything else learned so far: **[docs/FINDINGS.md](docs/FINDINGS.md)**.

## Docs

| Folder | What's in it |
| --- | --- |
| [FINDINGS.md](docs/FINDINGS.md) | **Start here.** What works, what doesn't, the best agent, written by hand |
| [results/](docs/results/forest_spots.md) | Every agent compared, one page per set of rules: [forest spots](docs/results/forest_spots.md) (the main comparison), [wooded spots](docs/results/wooded_spots.md), [arena](docs/results/arena.md), [other rules](docs/results/other_rules.md) (MineRL's own scoring, random spawns, held-out world) |
| [experiments/](docs/experiments/natural_lines.md) | Why each run was made: [natural-world training lines](docs/experiments/natural_lines.md), [arena experiments](docs/experiments/arena_experiments.md), [MineRL transfer](docs/experiments/minerl_transfer.md) |
| [reference/](docs/reference/environments.md) | [How the environments are set up](docs/reference/environments.md), [their speed](docs/reference/speed.md), [starting several Minecraft instances](docs/reference/parallel_launch.md) |
| [logs/](docs/logs/evaluations.md) | Every [evaluation](docs/logs/evaluations.md) and [training](docs/logs/training.md) run, newest first |

The tables in `results/`, `experiments/`, `reference/environments.md` and `logs/` are rebuilt from
the CSVs in `results/` after every run; text outside their generated blocks is kept.
`FINDINGS.md`, `experiments/minerl_transfer.md` and this README are written by hand.

## Running things

Run scripts through `run.bat` from the repository root: it sets up the compiler and the
`exp_craftground` conda env.

```
run.bat train.py --env natural                     # train in the natural world (3 Minecraft instances)
run.bat train.py --env natural --resume <model.zip> --timesteps 1000000   # continue a run
run.bat evaluate.py <model.zip> --env natural      # score it on the 100 fixed forest spots
run.bat evaluate.py <model.zip> --env natural --spots results/eval_spots_natural_wooded.json
run.bat evaluate.py <model.zip>                    # score it in the arena
run.bat experiments.py                             # rebuild the docs tables by hand
run.bat play.py                                    # play the env yourself, seeing what the agent sees
```

MineRL models (`runs/minerl/`) are detected automatically and run in MineRL mode; add
`--axe instant` to play them with our one-hit axe. Every flag is in `--help`.

| Where | What |
| --- | --- |
| `chop/` | The code: `envs/` (arena, natural world, MineRL mode), `training/`, `evaluation/`, `reports/` (builds the docs), `infra/` (launching instances) |
| `train.py`, `evaluate.py`, ... | Thin entry points into `chop/` |
| `runs/` | Trained models, checkpoints and videos (not in git) |
| `results/` | Every evaluation and training run as CSV rows, the fixed spot sets, and the JSON files naming lines and experiments |
| `tools/` | One-off checks and diagnostics ([tools/README.md](tools/README.md)) |
