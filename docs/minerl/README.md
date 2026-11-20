# CraftGround Experiments: MineRL Models in CraftGround

Models trained in the MineRL project (behavioural cloning on VPT videos, then PPO), run in
CraftGround's arena and natural world with MineRL's own view and actions ("MineRL mode",
`chop/envs/minerl.py`). Our own agents are in [natural/](../natural/README.md) and
[arena/](../arena/README.md); the summary of findings is [FINDINGS.md](../FINDINGS.md).

<!-- AUTO-GENERATED:START (rebuilt by experiments.py, edits here are overwritten) -->
_Tables last updated 2026-11-20 14:55._

- [Results](results.md): one table per evaluation type (world, spots, episode length, log cap, axe)
- [Evaluation runs](evaluation_runs.md): one row per evaluate.py run, newest first, with its settings
- [Training runs](training_runs.md): MineRL-mode fine-tunes and from-scratch runs (train.py)
- [MineRL mode](setup.md): what it copies from MineRL, how it was checked, what still differs
<!-- AUTO-GENERATED:END -->

## Notes

Written by hand; this section is never overwritten. Every number names its evaluation type, the
heading of its table in [Results](results.md). _Last updated 2026-11-20._

### The questions

1. **How well does a MineRL model do here, with and without fine-tuning, in the arena and in the
   natural world?** Natural world: answered below, both ways. Arena: zero-shot only.
2. **Can pure RL here match or beat it?** Not so far: fine-tuned for 1M ticks, the MineRL model ties
   our best pure-RL agents on forest spots and beats them on wooded ones; pure RL needed 3-4M ticks to
   get there. PPO from scratch in MineRL's own spaces (the fairest version) isn't run yet.
3. **How much does exactly matching the observations and actions matter?** One test so far (the axe).
   Next: hand hidden, our 114×64 view, 1 tick per decision.
4. **Does a fine-tuned MineRL model still improve after switching from discrete to continuous
   controls?** Not started: needs a continuous action head.

### What we found (zero-shot: no fine-tuning, 2026-11-19)

1. **The MineRL models transfer.** Scored as MineRL scored them (*fixed forest spots · 4,000-tick
   episodes, ends at 64 logs · iron axe*, first 50 spots), they keep 75-91% of their MineRL score:
   Impala PPO 54.0 (59.6 in MineRL), NatureCNN PPO 48.4 (59.8), Impala BC 44.1 (58.8). This is a newer
   Minecraft, a world they never saw, and no fine-tuning.
2. **Impala PPO transfers best**, and PPO on top of BC still helps here: 54.0 vs its BC start's 44.1.
   NatureCNN PPO scored the same as Impala PPO in MineRL but loses more here (48.4).
3. **The arena and the natural world are about equally easy for them** (*1,000-tick, iron axe*):
   Impala PPO 21.4 in the arena (out of 32), 19.6 on the forest spots. Walls and flat ground didn't
   confuse them, though they never saw either.
4. **The axe alone is worth +6 to +10 logs** (*1,000-tick*, iron vs one-hit axe): Impala PPO 19.6 → 29.4
   on the forest spots, 21.4 → 27.8 in the arena. The model's behaviour carries over; its pace was
   set by the slower axe. The first answer to question 3: one env difference changes the score a lot.
5. **Our pure-RL agents are still ahead of the best one zero-shot.** Same rules as our own reports
   (*fixed forest spots · 1,000-tick, no cap · one-hit axe*): Line 5 40.0, Impala PPO 29.4, so
   −10.6 ± 3.8 spot by spot (Impala better at 28 spots, worse at 67). It rarely gets stuck, though:
   3 empty episodes to Line 5's 4.
6. **Line 5 over 4,000 ticks** (*fixed forest spots, one-hit axe*, first 50 spots): 118.9 uncapped,
   about 3× its 1,000-tick 40, not 4×: it slows as nearby trees run out (median 112; one episode
   reached 517 by walking into a dark oak forest, whose 2×2 trunks hold many logs). With MineRL's
   64 cap: 54.6, reaching 64 on 37 of 50 spots. That matches Impala PPO's 54.0, but Impala had the
   slower iron axe, so they're in different tables and this isn't a like-for-like comparison.

### What we found (fine-tuned, 2026-11-20)

The Impala PPO, fine-tuned here: natural world, wooded spawns, one-hit axe, linear LR from 1e-4,
the model's own PPO settings (batch 64, 10 epochs, ent 0.001, target_kl 0.05), 252k decisions =
1.0M ticks (run `runs/minerl_mode/natural/ft_impala-ppo_wooded_one-hit-axe_20261120_124557`,
continued from the 11:57 run's 60k checkpoint after a low-memory stop; 2 instances).
All scores: *fixed spots · 1,000-tick, no cap · one-hit axe*, 100 spots.

7. **Fine-tuning works fast.** Forest spots: 29.4 zero-shot → 38.5 at 0.48M ticks → **43.4** at
   1.0M ticks (+13.9 ± 3.1 over zero-shot), the highest forest mean of any agent so far. Our pure-RL
   history reached 26.6 at 1M ticks (Line 4), 35.7 at 2M (4b), 41.2 at 3M (4c).
8. **It ties our best on forests and beats it on wooded spots.** Forest: +3.4 ± 4.0 vs Line 5
   (+2.1 ± 5.0 vs Line 4d): ahead, but not a significant difference. Wooded: **37.2 vs Line 5's 31.6,
   +5.6 ± 3.8**, better at 56 spots, worse at 41: a real difference. Fewest 0-log forest episodes of
   any agent (2; Line 5: 4).
9. **What it cost.** 1M ticks here, on top of what the MineRL model already had: BC on VPT videos plus
   120k PPO decisions (~0.5M ticks) in MineRL. Line 5 had 4M ticks of PPO here and nothing else. So in
   game time simulated here, starting from MineRL needed about a quarter of pure RL's; the
   demonstrations and the MineRL training are the rest of the price.

### How the numbers relate

- **Timesteps are decisions.** Our agents decide every tick; MineRL mode decides every 4 ticks. So
  a 1,000-tick episode is 1,000 decisions for Line 5 and 250 for a MineRL model, and 1M training
  steps in MineRL mode is 4M ticks of play.
- **MineRL's own scores** are over 24 worlds (seeds 1-24), 4,000 ticks, iron axe, ending at 64 logs,
  with mobs. Ours are over fixed spots in one world (seed 20261114), peaceful.

### Next

1. Continue the fine-tune (it was still climbing: 38.5 → 43.4 between 0.48M and 1M ticks) to see where
   it levels off; pure RL levelled off at ~41 by 3M ticks.
2. The cheap matching tests (zero-shot, ~10 min each): hand hidden, our 114×64 view, 1 tick per decision.
3. PPO from scratch in MineRL's spaces (question 2, fair version).
4. A continuous action head (question 4), starting from this fine-tuned model.
