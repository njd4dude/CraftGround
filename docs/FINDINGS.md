# Findings: Chop Trees with PPO in CraftGround

What we've learned so far, as plain statements. Each one gives the key numbers and links to
the table that supports it. The tables rebuild themselves after every run
([project README](../README.md) for how to run things):

- [results/](results/README.md): every agent compared, one page per environment
(natural world: oak and birch forests; natural world: every biome with trees; arena) plus other evaluations
- [experiments/](experiments/README.md): why each run was made: natural-world training lines,
arena experiments, and the MineRL transfer
- [reference/](reference/README.md): how the environments are set up, their speed, and starting
several Minecraft instances
- [logs/](logs/README.md): every evaluation and training run

This file is written by hand: update it when a result changes a statement.
_Last updated 2026-11-24._

**How to read the numbers.** Scores are logs chopped per 1000-tick episode. Natural-world
scores are over the same 100 fixed spots for every model, either in oak and birch forests ("oak/birch
spots", the main set) or spread over every biome with trees ("all-biome spots"), so models can be compared
spot by spot: "+4.2 ± 3.0" is the average difference per spot with its 95% range, and a range
that doesn't include 0 counts as a real difference. Use 100 spots to compare models: 50 gives about
the same mean but a ~1.6× wider range, since one episode on the same spot can differ by ~14
logs between runs (Line 4c: 40.5 over 50 vs 41.2 over 100). Arena scores are over 50 episodes, out of 32.

**Which model is which.** Every model named here ("Line 4c", "Impala BC→PPO + CG 1M"...)
has its model file listed in the **Models named here** table at the end of this page.

## Best agent so far

**Impala BC→PPO + CG 3.5M, 1 tick per decision, target_kl 0.02** (Impala BC→PPO + CG 2.9M, 1 tick per
decision + 500k ticks with smaller policy updates; findings 7 and 8 under
[MineRL models in CraftGround](#minerl-models-in-craftground)): **47.7 logs** on the all-biome spots
(48.7 without the unfair spots; median episode 51.5), +7.2 ± 3.9 over Impala BC→PPO + CG 2M. Not run on
the oak/birch spots yet.
Model: `runs/minerl_finetuned/ft5_impala-ppo_wooded_one-hit-axe_1tick_small-updates_20261124_131835/model.zip`
(run it with `evaluate.py <model> --env natural --axe instant`; 1 tick per decision comes from its config).
[All-biome spots](results/natural_all_tree_biomes.md), [MineRL transfer](experiments/minerl_transfer.md)

The best model before 2026-11-24, and the best on the oak/birch spots:

**Impala BC→PPO + CG 2M**: the MineRL project's Impala BC→PPO model (BC on VPT videos, then 120k PPO steps in MineRL), trained here in CraftGround for 2M more ticks (MineRL mode, wooded spawns = any biome with trees, one-hit axe, gentle PPO settings; names explained under [MineRL models in CraftGround](#minerl-models-in-craftground)):
**44.7 logs** on the oak/birch spots and **38.5** on the all-biome spots. Impala BC→PPO + CG 2M beats Line 5 on both:
+4.7 ± 3.9 and +6.9 ± 4.0. Impala BC→PPO + CG levelled off after 1-1.5M ticks (Impala BC→PPO + CG 1M: 43.4).
Model: `runs/minerl_finetuned/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_615292_steps.zip`
(a checkpoint: the run froze 10k steps before its end and never wrote `model.zip`; run it with
`evaluate.py <model> --env natural --axe instant`).
[Oak/birch spots](results/natural_oak_birch_forests.md), [All-biome spots](results/natural_all_tree_biomes.md), [MineRL transfer](experiments/minerl_transfer.md)

The best pure-RL agent:

**Line 5d**, multi actions, 5M steps (Line 5 + 1M with wooded spawns and a penalty for being
stuck, finding 10 below): **42.0 logs** per episode on the oak/birch spots (Line 5: 40.0, +2.0 ± 4.5,
a tie) and **39.3** on the all-biome spots, where it beats Line 5 by +7.6 ± 4.4 and ties
Impala BC→PPO + CG 2M (−1.3 ± 4.0, both of that model's all-biome evaluations averaged).
Model: `runs/natural/line5d_stuck-penalty/20261122_223446/model.zip`.

The best pure-RL agent before the stuck penalty was **Line 5**, 4M steps (Line 4c + 1M with wooded
spawns): **40.0** oak/birch, **31.6** all-biome, beating the forest-trained Line 4d there by +6.0 ± 4.8.
Model: `runs/natural/line5_wooded-spawns/20261119_005447/model.zip`.
[Oak/birch spots](results/natural_oak_birch_forests.md), [All-biome spots](results/natural_all_tree_biomes.md)

## What works

1. **Multi actions beat single actions in the natural world.** Choosing move, turn (exact
   degrees), look and jump every step with attack held scored 26.6, against 22.4 for the
   same recipe with one action per step: +4.2 ± 3.0 logs spot by spot (better at 57 spots,
   worse at 36). It learned slower at first and kept improving after single actions levelled
   off. [Line 4](experiments/natural_lines.md) vs [Line 3](experiments/natural_lines.md)
2. **Multi-action agents keep improving with more training.** Resuming Line 4 for 1M more
   steps (linear LR restarted at 1e-4) went 26.6 → 35.7 at 2M (+9.1 ± 3.6 spot by spot), and
   another 1M went to 41.2 at 3M (+5.5 ± 4.5), and a third 1M (Line 4d) to 41.3 (+0.1 ± 4.5):
   it levels off at ~41 by 3M. Single actions levelled off much earlier, at ~22–25 (see below).
   [Line 4b](experiments/natural_lines.md), [Line 4c](experiments/natural_lines.md),
   [Line 4d](experiments/natural_lines.md)
3. **Training in every biome with trees, not just oak/birch forests.** With spawns in any biome with
   trees (snowy taiga, taiga, jungle, swamp, dark forest...; `train.py --spawn wooded`), Line 5
   beat its control (Line 4d, same 1M steps on oak/birch spawns) by +6.0 ± 4.8 logs on the all-biome
   spots and tied it on the oak/birch spots (−1.3 ± 5.0). It halved the 0-log episodes, 13 vs 26 over both
   sets (p = 0.007): it gets stuck or lost much less. More forest-only training made the agent
   *worse* outside them (all-biome spots 30.5 → 25.7). Another 1M steps with wooded spawns (Line 5b)
   changed nothing (all-biome −0.3 ± 4.5, oak/birch −1.4 ± 4.5): it levels off by 4M too.
   [Line 5](experiments/natural_lines.md), [Line 5b](experiments/natural_lines.md),
   [All-biome spots](results/natural_all_tree_biomes.md)
4. **Jump is needed on natural terrain.** Adding jump and forward_jump nearly doubled the
   score at ~500k steps: 18.0 vs 9.7 (random spawns). [Line 1](experiments/natural_lines.md),
   [Line 2](experiments/natural_lines.md)
5. **A learning rate that decays to 0 keeps the gains.** From the same 600k model, a linear
   decay rose to 24.7 while a constant rate fell to 15.0: +9.8 ± 3.5 spot by spot.
   [Line 2b](experiments/natural_lines.md)
6. **Agents generalise to a world they never trained in.** On a held-out world, scores match
   the training world: Line 2b +0.7, Line 3 −0.5, Line 4b +3.9 logs (ranges overlap). They learned to chop trees, not to
   memorise one map. [Held-out world (archived)](results/archive/held_out.md)
7. **1 game tick per decision is best.** It beats 4 ticks for both action sets on the fixed
   arena (single 30.5 vs 24.0, multi 31.3 vs 27.8), whether runs are matched on decisions,
   game time or wall time. [Arena experiment 4](experiments/arena_experiments.md)
8. **The arena is easy.** At 1 tick per step, PPO reaches 30–32 of 32 logs in 200k steps
   (~7 minutes) with either action set. Good for quick checks, too easy to rank good setups.
   [Arena experiment 1](experiments/arena_experiments.md), [experiment 3](experiments/arena_experiments.md)
9. **Fixed evaluation spots make comparisons possible.** With random spawns, a model's mean
   moves by ±3–4 logs from spawn luck alone; every model now plays the same 100 spots.
   [Oak/birch spots](results/natural_oak_birch_forests.md)
10. **A small penalty for being stuck.** A tick is "stuck" when the player moved < 1 block over the
   last 2 s (40 ticks) without chopping, e.g. jumping at a wall too high to climb. Line 5b spent
   26–31% of its ticks stuck, and the more stuck ticks an episode has, the fewer logs (correlation
   −0.5 to −0.76). Line 5d (Line 5 + 1M steps, wooded spawns, 0.01 reward off per stuck tick;
   `train.py --stuck-penalty 0.01`) against its control Line 5b (same without the penalty): stuck
   23% of ticks on every spot set, and on the all-biome spots 39.3 vs 31.1 logs (+8.2 ± 4.5) with
   half as many episodes stuck ≥ 50% of the time (11 vs 23). Oak/birch 42.0 vs 38.8 (+3.2 ± 4.5)
   and anywhere 23.4 vs 20.9 (+2.5 ± 3.6) are ties. 0-log episodes didn't fall: those spots
   mostly have no tree in reach. The penalty is small (~2.5 logs' worth per episode); a bigger one
   is untested. [Line 5d](experiments/natural_lines.md)

## What doesn't work

1. **Holding each action for 4 ticks (action repeat).** Worse in every comparison (see above).
   The 4-tick single agent skips the bottom log of most trees. It also takes about twice as
   long to train, since each decision simulates 4 ticks.
   [Arena experiment 4](experiments/arena_experiments.md)
2. **A constant learning rate past ~600k.** Line 2 peaked at 18.1 and slid to 15.0 by 1M
   (−3.1 ± 2.6 spot by spot). [Line 2](experiments/natural_lines.md)
3. **Training single-action agents longer.** Every single-action recipe ends at ~22–25 logs:
   500k more steps changed nothing (−0.5 ± 3.1), and one 3M schedule tied a 1M one
   (23.9 vs 22.4, +1.5 ± 3.4). The limit was the setup, which multi actions then passed.
   [Line 3b](experiments/natural_lines.md), [Line 3c](experiments/natural_lines.md)
4. **Using arena-trained agents in the natural world.** The best arena agent
   (`fixed_multi_1tick_r2`, 31.7 of 32 in the arena) scores 5.7 on the oak/birch spots: 5× random
   actions (1.1) but −34.3 ± 4.2 vs Line 5, with 31 of 100 episodes at 0 logs (median 3). The
   arena's 8 oaks stand at the same spots every episode on flat ground, so arena agents never learn
   to search for trees or climb. The transfer only works the other way: Impala BC→PPO + CG 2M,
   trained on wooded spawns and never in the arena, scores 29.9 of 32 there (min 23), above every
   4-tick arena agent (27.8) and Impala BC→PPO, zero-shot (27.8, min 3).
   [Oak/birch spots](results/natural_oak_birch_forests.md), [Arena](results/arena.md)
5. **Pretraining in the arena, then training in the natural world.** Line 7 started from an arena
   agent (multi actions with jump, 200k steps, 9 minutes, 28.0 in the arena) and trained with Line 4's
   exact setup; only the starting weights differ. Spot by spot vs Line 4 at the same natural steps:
   260k −2.1 ± 1.2 (10.7 vs 12.8), 500k +4.8 ± 2.1 (17.8 vs 13.1), 1M −1.2 ± 2.7 (25.4 vs 26.6, a
   tie). Line 7's head start (10.8 with no natural training) was gone by 260k: Line 7 stayed at ~11
   while unlearning arena habits, and Line 4 caught up. Side result: with jump, the arena agent scores
   10.8 in the natural world untrained, against 5.7 for `fixed_multi_1tick_r2` without jump, though
   the jump agent scores less in the arena (28.0 vs 31.7; one run each).
   [Line 7](experiments/natural_lines.md), [Oak/birch spots](results/natural_oak_birch_forests.md)
6. **Spawning anywhere on land instead of only where trees grow.** Line 5c (Line 5 + 1M steps with
   spawns anywhere on land: plains, desert, savanna, mountains... included) tied its control Line 5b
   (Line 5 + 1M steps, wooded spawns) on all three spot sets: anywhere-on-land 20.4 vs 21.1
   (−0.7 ± 4.5), oak/birch 36.6 vs 38.6 (−2.0 ± 4.2), all-biome 35.4 vs 31.3 (+4.0 ± 4.7, close).
   Line 5c didn't reduce 0-log episodes on the anywhere-on-land spots (Line 5c 36, Line 5b 37,
   Line 5 36 of 100); 28 of those spots are 0 for all three agents, probably with no tree in reach.
   [Line 5c](experiments/natural_lines.md), [Anywhere-on-land spots](results/natural_anywhere.md)

## MineRL models in CraftGround

Models trained in the MineRL project run here in "MineRL mode": their own 64×64 view with the hand,
`[move, yaw, pitch]` actions, 4 ticks per decision. Each model's name lists its training stages:
**BC** = behavioural cloning on VPT videos (MineRL project, no simulator); **→PPO** = then 120k PPO
steps in MineRL's simulator; **+ CG &lt;ticks&gt;** = then PPO here in CraftGround (MineRL mode, wooded
spawns, one-hit axe); **zero-shot** = no CraftGround training; **Scratch** = random starting weights,
CraftGround training only. Stage 3 used the **gentle PPO settings** (LR 1e-4, entropy 0.001, batch 64,
10 epochs, target_kl 0.05) unless a name says **Line 4 PPO settings** (LR 2.5e-4, entropy 0.01, batch
256, 4 epochs). The full name table and every result: [MineRL transfer](experiments/minerl_transfer.md).

1. **The MineRL models transfer.** With no training here the three zero-shot models keep 75-91% of
   their MineRL score (scored MineRL's way), but trail our agents under our rules: Impala BC→PPO,
   zero-shot 29.4 vs Line 5's 40.0 (−10.6 ± 3.8).
2. **Impala BC→PPO + CG beats pure RL, in a quarter of the game time.** Impala BC→PPO + CG 2M: 44.7
   oak/birch / 38.5 all-biome (Line 5 after 4M ticks: 40.0 / 31.6; +4.7 ± 3.9 and +6.9 ± 4.0). Impala
   BC→PPO + CG levels off after 1-1.5M ticks; pure RL levelled off at ~41 after 3M.
3. **MineRL's simulator isn't needed.** Impala BC + CG 1M (never run in MineRL's simulator) ties Line 5
   on the oak/birch spots (40.4, +0.4 ± 3.5) and beats Line 5 on the all-biome spots (37.5, +5.8 ± 3.7).
   Stage 2 (120k PPO steps in MineRL) is worth ~3 logs on the oak/birch spots (Impala BC→PPO + CG 1M
   43.4 vs Impala BC + CG 1M 40.4).
4. **The pretraining (BC on VPT) is what makes the MineRL models good.** Scratch Impala (MineRL mode) +
   CG 1M, gentle PPO settings (same network, PPO settings and 1M ticks as Impala BC→PPO + CG 1M, random
   starting weights) scores 14.6 oak/birch / 14.3 all-biome: −28.8 ± 2.9 and −22.9 ± 3.5 vs Impala
   BC→PPO + CG 1M. Scratch Impala (MineRL mode) + CG 1M, Line 4 PPO settings scores 18.9 / 18.4
   (+4.4 ± 1.9 / +4.2 ± 2.2 over the gentle-settings Scratch run), still ~21 behind Impala BC + CG 1M.
5. **MineRL mode learns faster than our setup per decision, slower per tick.** With Line 4's PPO
   settings in both, Scratch Impala (MineRL mode) + CG 1M vs Line 4 (our view, actions and NatureCNN,
   also from scratch): at the same ~250k decisions, 18.9 vs Line 4 @ 260k steps 12.8 (+6.2 ± 1.7); at
   the same 1M game ticks, 18.9 vs Line 4 @ 1M steps 26.6 (−7.7 ± 2.9), since MineRL mode decides once
   every 4 ticks.
6. **The stuck penalty doesn't help Impala BC→PPO + CG.** Impala BC→PPO + CG 3M, stuck penalty (Impala
   BC→PPO + CG 2M + 1M ticks with Line 5d's 0.01 per stuck tick) ties Impala BC→PPO + CG 2M on the
   all-biome spots: 41.0 vs 40.5, +0.4 ± 3.1 spot by spot (better at 50 spots, worse at 45; 200 episodes
   each). Impala BC→PPO + CG 2M was already stuck only ~120 ticks per episode (Impala BC→PPO + CG 3M,
   stuck penalty: 113), so the penalty had little to fix. Its 0-log episodes (15 of 200, against 11) are
   the same few spots every time: all-biome spots 5, 27 and 56 score 0 for every agent (`results/spot_notes.json`:
   a cave drop and a ravine floor, both tagged unfair because the actions can't place or dig blocks; and
   sea ice with the trees behind the agent). Without the two unfair spots the scores are 41.8 and 41.4.
   [All-biome spots](results/natural_all_tree_biomes.md)
7. **One tick per decision is a small, unproven gain.** Impala BC→PPO + CG 2M switched to 1 tick per
   decision (each decision turns a quarter as far, so turning speed per tick stays MineRL's) loses
   nothing with no training (40.6 all-biome, 100 episodes). Impala BC→PPO + CG 2.9M, 1 tick per decision
   (+1M decisions = 1M ticks at 1 tick, gamma 0.9975 = 0.99 per 4 ticks) scores 42.7 all-biome (43.6
   without the unfair spots): +2.2 ± 3.5 over Impala BC→PPO + CG 2M (better at 57 spots, worse at 40;
   200 episodes each), the highest all-biome score so far but within noise. The median episode rose
   from 42 to 48 logs, while 0-log episodes stayed the same (12 of 200, against 11) and a few spots
   collapsed (spot 85: 74/68 → 0/5). Typical spots got faster; the failures didn't change. Training at
   1 tick runs at about the same ticks per second per instance as at 4: the game, not PPO, sets the speed.
   [All-biome spots](results/natural_all_tree_biomes.md)
8. **Smaller policy updates: +5 logs in 500k ticks.** In stage 3 the KL early stop (`target_kl`), not the
   learning rate, set how far each PPO update moved the policy: Adam moves every weight by about the
   learning rate per step, so the pretrained network passed the stop (1.5 × 0.05) within the first of
   its 10 epochs in many updates (211 of 243 updates of Impala BC→PPO + CG 2.9M, 1 tick per decision
   stopped early; approx_kl 0.03-0.05, clip fraction 0.15-0.35). Lowering the learning rate from 1e-4 to
   3e-5 alone left approx_kl the same (0.02-0.05 over 6 updates). A check before any update found the
   stored and recomputed log-probabilities equal, so the large KL is real, not a bookkeeping bug.
   Impala BC→PPO + CG 3.5M, 1 tick per decision, target_kl 0.02 (Impala BC→PPO + CG 2.9M, 1 tick per
   decision + 500k ticks, target_kl 0.02, linear LR from 3e-5, 4 epochs; approx_kl ~0.011, clip ~0.10)
   scores 47.7 all-biome: +5.0 ± 3.5 over Impala BC→PPO + CG 2.9M, 1 tick per decision (better at 57
   spots, worse at 37; 200 episodes each), with the median episode at 51.5. Its 0-log episodes didn't
   change (12 of 200). The three settings changed together; target_kl is the one that binds (106 of
   122 updates still stopped early).
   [All-biome spots](results/natural_all_tree_biomes.md)

## Not proven yet

1. **Multi vs single actions in the arena.** Multi is ahead, 31.3 vs 30.5 over 3 runs each,
   but the groups overlap (p = 0.2); both are near the 32 cap.
   [Arena experiment 3](experiments/arena_experiments.md)
2. **Line 4 vs the previous best (Line 2b).** +1.9 ± 3.4 spot by spot: ahead, not significant.
3. **Why 4 ticks is worse.** Likely coarser control (it can only correct every 4 ticks) and
   wasted ticks (a 1-tick axe swing held for 4). Untested; a 2-tick run or the 4-tick videos
   would tell.

## Facts about the environment

1. **CraftGround shows actions late.** In the image, a camera move appears 1 tick late,
   walking 2, an attack's result (and its reward) 3. The player's position is on time. The
   game waits for each action, so the delay is fixed, not random. Agents learn around it;
   it isn't worth fixing with action repeat. (`tools/check_timing.py`)
2. **Speed.** With 3 instances: arena ~490 steps/s with single actions and ~370 with multi;
   natural world ~190 steps/s with either (resets dominate there: Line 3 and Line 4 both took
   ~89 minutes for 1M steps). [Speed](reference/speed.md)
3. **Memory.** An arena instance uses ~2.4 GB; a natural one grows to ~4 GB (2G Java heap,
   render distance 4). 3 instances is the safe number on this 32 GB machine. Claude Code stops
   background runs when memory runs low, which killed several runs. In MineRL mode 3 natural instances
   plus training plus open apps (Cursor, browsers: ~6 GB) was too much (2026-11-20); MineRL-mode runs use
   2, at almost the same speed (larger rollouts per instance).
4. **An instance can freeze without an error.** On 2026-11-20 a natural-world instance stopped answering
   during a training run; nothing was logged and the trainer waited 80 minutes. Long runs since then go
   through a watchdog script that kills a run whose log is silent for 20 minutes and resumes it from the
   latest checkpoint (saved every 20k steps).
5. **Waiting for the terrain to draw after a teleport matters.** After each reset's teleport the
   natural env idles until the view has stopped changing for 1.5 s (`_wait_for_render`, chop/envs/natural.py);
   in a third of resets terrain is still appearing 0.5–3.5 s after the teleport. Line 5d evaluated without
   the wait (`evaluate.py --no-render-wait`, 2026-11-23; rows in `results/eval_results_render_wait_test.csv`,
   kept out of the reports) scored lower on both spot sets: oak/birch 37.2 vs 42.0 (−4.8 ± 4.6), all-biome
   35.4 vs 39.3 (−3.9 ± 3.7), with more stuck ticks (27% vs 23%). The wait is most of a fixed-spot reset
   (resets took 39% of evaluation time with it, 15% without), but it stays.

## Bugs found, and what they affected

- **Resumed runs recorded the wrong PPO settings** (fixed 2026-11-20). A resumed model keeps its own
batch size, epochs, entropy and target_kl, but config.json recorded train.py's defaults. Training was
never affected; the two Impala BC→PPO + CG configs were corrected by hand (`config_note`).
- **train.py crashed starting from a behaviour-cloned model** (fixed 2026-11-20): `--target-kl` had no
default to record, and a never-trained model has no episode buffers to clear. Both crashed at launch,
before any training.

1. **Dug holes stayed in the arena** (fixed 2026-11-17). Pits from earlier episodes trapped
   agents, so arena results before the fix aren't valid comparisons
   ([experiment 2](experiments/arena_experiments.md) is marked preliminary). The arena now
   refills the ground every reset.
2. **Held attack stopped working after a reset.** CraftGround only presses attack when it goes
   from off to on, so an agent holding attack (multi actions) couldn't chop after its first
   episode. Fixed before any multi-action run was trained.

## Models named here

The model file behind each agent this page names. Score: mean logs on the 100 oak/birch spots
(1,000 ticks, one-hit axe) unless noted. Every evaluation of every model: [logs](logs/evaluations.md).

| Named here as | Score | Model file |
| --- | --- | --- |
| **Impala BC→PPO + CG 3.5M, 1 tick per decision, target_kl 0.02** (best) | 47.7 (all-biome spots; not run on oak/birch) | `runs/minerl_finetuned/ft5_impala-ppo_wooded_one-hit-axe_1tick_small-updates_20261124_131835/model.zip` |
| **Impala BC→PPO + CG 2M** (best on oak/birch) | 44.7 | `runs/minerl_finetuned/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_615292_steps.zip` |
| Impala BC→PPO + CG 2.9M, 1 tick per decision | 42.7 (all-biome spots; not run on oak/birch) | `runs/minerl_finetuned/ft4_impala-ppo_wooded_one-hit-axe_1tick_20261123_212345/checkpoints/ppo_chop_1595292_steps.zip` |
| Impala BC→PPO + CG 3M, stuck penalty | 41.0 (all-biome spots; not run on oak/birch) | `runs/minerl_finetuned/ft3_impala-ppo_wooded_one-hit-axe_stuck-penalty_20261123_181405/model.zip` |
| Impala BC→PPO + CG 1M | 43.4 | `runs/minerl_finetuned/ft_impala-ppo_wooded_one-hit-axe_20261120_124557/model.zip` |
| Impala BC + CG 1M | 40.4 | `runs/minerl_finetuned/ft_impala-bc_wooded_one-hit-axe_20261120_220917/model.zip` |
| Impala BC→PPO, zero-shot | 29.4 | `runs/minerl/model_20261112_130747_ppo_bc_impala_all_kl05/final.zip` |
| Scratch Impala (MineRL mode) + CG 1M, Line 4 PPO settings | 18.9 | `runs/minerl_finetuned/scratch_impala_rl-settings_20261121_231545/model.zip` |
| Scratch Impala (MineRL mode) + CG 1M, gentle PPO settings | 14.6 | `runs/minerl_finetuned/scratch_impala_20261121_193751/model.zip` |
| Impala BC, zero-shot | 22.2 | `runs/minerl/model_20261112_101203_bc_impala_6xx-1763_7xx-2256_8xx-1971_9xx-832_10xx-777_cont/final.zip` |
| **Line 5d** (best pure RL; stuck penalty) | 42.0 | `runs/natural/line5d_stuck-penalty/20261122_223446/model.zip` |
| Line 5 | 40.0 | `runs/natural/line5_wooded-spawns/20261119_005447/model.zip` |
| Line 5b | 38.6 (38.8 re-run with stuck ticks) | `runs/natural/line5b_resume-4M/20261119_161818/model.zip` |
| Line 4d | 41.3 | `runs/natural/line4d_resume-3M/20261119_112830/model.zip` |
| Line 4, 1M steps | 26.6 | `runs/natural/line4_multi-actions/20261118_124345/model.zip` |
| Line 4 @ 260k steps (same number of decisions as the Scratch Impala (MineRL mode) runs) | 12.8 | `runs/natural/line4_multi-actions/20261118_124345/checkpoints/ppo_chop_259974_steps.zip` |
| Line 4c | 41.2 | `runs/natural/line4c_resume-2M/20261118_183414/model.zip` |
| Line 4b | 35.7 | `runs/natural/line4b_resume-1M/20261118_162138/model.zip` |
| Line 4 | 26.6 | `runs/natural/line4_multi-actions/20261118_124345/model.zip` |
| Line 3 | 22.4 | `runs/natural/line3_jump-linear-lr-scratch/20261116_123155/model.zip` |
| Line 3b | 21.9 | `runs/natural/line3b_finetune-500k/20261116_145213/model.zip` |
| Line 3c (the 3M schedule) | 23.9 | `runs/natural/line3c_linear-lr-3M/20261117_114712/model.zip` |
| Line 2b | 24.7 | `runs/natural/line2b_linear-lr-from-601k/20261115_231007/model.zip` |
| Line 2 at its 602k peak | 18.1 | `runs/natural/line2_jump-constant-lr/20261115_185527/checkpoints/ppo_chop_601536_steps.zip` |
| Line 2 at 1M | 15.0 | `runs/natural/line2_jump-constant-lr/20261115_192015/model.zip` |
| Line 2 at ~500k (random spawns, finding 4) | 18.0* | `runs/natural/line2_jump-constant-lr/20261115_173930/model.zip` |
| Line 1 (random spawns, finding 4) | 9.7* | `runs/natural/line1_no-jump/20261115_161832/model.zip` |
| `fixed_multi_1tick_r2` (best arena agent; arena 31.7) | 5.7 | `runs/arena/fixed_multi_1tick_r2_20261117_220328/model.zip` |
| Line 7, 1M natural steps (arena-pretrained) | 25.4 | `runs/natural/line6_arena-pretrained/20261122_150832/model.zip` |
| Line 7 @ 500k / 260k natural steps | 17.8 / 10.7 | `runs/natural/line6_arena-pretrained/20261122_150832/checkpoints/ppo_chop_700458_steps.zip` / `ppo_chop_460482_steps.zip` (the step count includes the 200,508 arena steps) |
| Line 7's arena start (multi + jump, 3 instances; arena 28.0) | 10.8 | `runs/arena/arena_pretrain_multi_jump_3envs_20261122_145451/model.zip` |
| Same with 6 instances (arena 25.0; not run in the natural world) | – | `runs/arena/arena_pretrain_multi_jump_20261122_144237/model.zip` |
| Line 4 @ 500k steps | 13.1 | `runs/natural/line4_multi-actions/20261118_124345/checkpoints/ppo_chop_499950_steps.zip` |

\* random spawns, not the fixed spots (older evaluations, before fixed spots existed).

