# CraftGround: chopping trees with RL

Reinforcement learning (PPO, Stable-Baselines3) agents that chop as many logs as they can in one
Minecraft episode (1,000 game ticks, ~50 s), run in [CraftGround](https://github.com/yhs0602/CraftGround).
Two worlds: a small walled **arena** with 8 oaks, and a **natural** generated world where every
episode starts in a different forest. Models trained in a separate MineRL project can also run
here unchanged, in **MineRL mode** (their own 64×64 view and actions).

## Best agents so far

Logs per episode in the natural world, on two fixed sets of 100 spots: oak and birch forests, and
every biome with trees (taiga, snowy taiga, jungle, dark forest, swamp...):

| Agent | Oak/birch forests | Every biome with trees |
| --- | --- | --- |
| Impala BC→PPO + CG 2M (MineRL project's BC + PPO model, then 2M ticks of PPO here) | **44.7** | **38.5** |
| Impala BC + CG 1M (MineRL project's BC model, then 1M ticks of PPO here) | 40.4 | 37.5 |
| Line 5d: pure RL here, 5M ticks, last 1M with a penalty for being stuck | 42.0 | 39.3 |
| Line 5: pure RL here, 4M ticks | 40.0 | 31.6 |

What's behind these, and everything else learned so far: **[docs/FINDINGS.md](docs/FINDINGS.md)**.

## Docs

| Folder | What's in it |
| --- | --- |
| [FINDINGS.md](docs/FINDINGS.md) | **Start here.** What works, what doesn't, the best agent, written by hand |
| [results/](docs/results/README.md) | Every agent compared, one page per environment: [natural world, oak and birch forests](docs/results/natural_oak_birch_forests.md) (the main comparison), [natural world, every biome with trees](docs/results/natural_all_tree_biomes.md), [arena](docs/results/arena.md); and [other evaluations](docs/results/other_evaluations.md) under different rules (MineRL's own scoring, random spawns, held-out world) |
| [experiments/](docs/experiments/README.md) | Why each run was made: [natural-world training lines](docs/experiments/natural_lines.md), [arena experiments](docs/experiments/arena_experiments.md), [MineRL transfer](docs/experiments/minerl_transfer.md) |
| [reference/](docs/reference/README.md) | [How the environments are set up](docs/reference/environments.md), [their speed](docs/reference/speed.md), [starting several Minecraft instances](docs/reference/parallel_launch.md) |
| [logs/](docs/logs/README.md) | Every [evaluation](docs/logs/evaluations.md) and [training](docs/logs/training.md) run, newest first |

The tables in `docs/results/`, `docs/experiments/`, `docs/reference/environments.md` and `docs/logs/`
are rebuilt from the CSVs in the top-level `results/` folder after every run; text outside their
generated blocks is kept. `docs/FINDINGS.md`, `docs/experiments/minerl_transfer.md` and this README
are written by hand.

## Running things

Run scripts through `run.bat` from the repository root: it sets up the compiler and the
`exp_craftground` conda env.

```
run.bat train.py --env natural                     # train in the natural world (3 Minecraft instances)
run.bat train.py --env natural --resume <model.zip> --timesteps 1000000   # continue a run
run.bat evaluate.py <model.zip> --env natural      # score it on the 100 oak/birch forest spots
run.bat evaluate.py <model.zip> --env natural --spots results/eval_spots_natural_wooded.json   # ...or every biome with trees
run.bat evaluate.py <model.zip>                    # score it in the arena
run.bat experiments.py                             # rebuild the docs tables by hand
run.bat play.py                                    # play the env yourself, seeing what the agent sees
run.bat play_human.py                              # play evaluation spots with normal controls (human ceiling check)
```

MineRL models (`runs/minerl/`) are detected automatically and run in MineRL mode; add
`--axe instant` to play them with our one-hit axe. Every flag is in `--help`.

| Where | What |
| --- | --- |
| `chop/` | The code: `chop/envs/` (arena, natural world, MineRL mode), `chop/training/`, `chop/evaluation/`, `chop/reports/` (builds the docs), `chop/infra/` (launching instances) |
| `train.py`, `evaluate.py`, ... | Thin entry points into `chop/` |
| `runs/` | Trained models, checkpoints and videos (not in git) |
| `results/` | Every evaluation and training run as CSV rows, the fixed spot sets, and the JSON files naming lines and experiments |
| `tools/` | One-off checks and diagnostics ([tools/README.md](tools/README.md)) |
