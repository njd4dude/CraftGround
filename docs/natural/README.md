# CraftGround Experiments: Chop Trees in a Natural World

Goal: chop as many logs as possible in one episode, starting in a real generated
forest instead of the walled arena (chop/envs/natural.py). Arena results are in
[arena/](../arena/README.md); the summary of findings is [FINDINGS.md](../FINDINGS.md).

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-20 23:49._

- [Fixed forest spawns](fixed_forest_spawns.md): 100 fixed spots in oak/birch forests, one row per agent: the main comparison
- [Fixed wooded spawns](fixed_wooded_spawns.md): 100 fixed spots in any biome with trees (snowy taiga, taiga, jungle, swamp...), next to each agent's forest score
- [Training lines](lines.md): every training line: what it tested, what it found, its pieces
- [Evaluation runs](evaluation_runs.md): one row per evaluate.py run, newest first
- [Random spawns](random_spawns.md): a new random spot every episode, forest and wooded: checks the fixed-spawn scores hold
- [Environment setup](setup.md): world, spawns, actions, reward, episode length
<!-- AUTO-GENERATED:END -->

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.
