# CraftGround Experiments: Chop Trees (arena)

Goal: train an agent to chop as many oak logs as possible in one episode. Natural-world results
are in [natural/](../natural/README.md); the summary of findings is [FINDINGS.md](../FINDINGS.md).

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-20 14:55._

- [Experiments](experiments.md): each experiment's question, what it found, and its runs with their scores
- [Other runs](other_runs.md): runs no experiment lists yet
- [Setup](setup.md): arena, trees, actions, reward, episode length, random baseline
<!-- AUTO-GENERATED:END -->

## Notes

Write your observations, ideas and next steps here. This section is never overwritten.

- **2026-11-13:** `check_env.py` passed. A scripted agent chopped 3 logs per episode and the trees regrew after reset. Rewards sometimes show up 1–2 steps after the attack that earned them.
- **2026-11-13, run 1 (`chop_ppo_20261113_131702`):** PPO learned fast. Logs per episode went 3.5 → 18 → 25 → 28 over the first 70 episodes and finished at a last-20 average of 31.4/32. The task may now be too easy, so next ideas are harder layouts, more trees or random tree positions.
- **2026-11-13:** Parallel envs: `train.py` runs 6 Minecraft instances by default (`--n-envs`); `--n-steps` is the total across all envs, so each update is still ~2048 steps. Raw env benchmark (random actions): 1 env 451, 2 → 714, 4 → 858, 8 → 1076 steps/s, with 8 using 92% CPU and 17 GB RAM. The CPU is the limit; the GPU stays under 40%.
- **2026-11-17:** Speed with 3 instances: single actions ~490 steps/s, multi ~370 (turning the camera looks like the slow part, not verified), 4 ticks per step ~220 steps/s (~880 ticks/s). Each arena instance uses ~2.4 GB, so 3 is the safe number on this machine when anything else is open.
