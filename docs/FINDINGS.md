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
_Last updated 2026-11-21._

**How to read the numbers.** Scores are logs chopped per 1000-tick episode. Natural-world
scores are over the same 100 fixed spots for every model, either in oak and birch forests ("oak/birch
spots", the main set) or spread over every biome with trees ("all-biome spots"), so models can be compared
spot by spot: "+4.2 ± 3.0" is the average difference per spot with its 95% range, and a range
that doesn't include 0 counts as a real difference. Use 100 spots to compare models: 50 gives about
the same mean but a ~1.6× wider range, since one episode on the same spot can differ by ~14
logs between runs (Line 4c: 40.5 over 50 vs 41.2 over 100). Arena scores are over 50 episodes, out of 32.

## Best agent so far

**The MineRL Impala PPO, fine-tuned here for 2M ticks** (MineRL mode, wooded spawns = any biome with trees, one-hit axe):
**44.7 logs** on the oak/birch spots and **38.5** on the all-biome spots. It beats Line 5 on both:
+4.7 ± 3.9 and +6.9 ± 4.0. It levelled off after 1-1.5M ticks (43.4 at 1M).
[Oak/birch spots](results/natural_oak_birch_forests.md), [All-biome spots](results/natural_all_tree_biomes.md), [MineRL transfer](experiments/minerl_transfer.md)

The best pure-RL agent:

**Line 5**, multi actions, 4M steps (Line 4c + 1M with wooded spawns): **40.0 logs** per
episode on the oak/birch spots (tied with Line 4c's 41.2 and Line 4d's 41.3) and **31.6** on
the all-biome spots, where it beats the forest-trained Line 4d by +6.0 ± 4.8. Fewest 0-log
episodes of our pure-RL agents. [Oak/birch spots](results/natural_oak_birch_forests.md), [All-biome spots](results/natural_all_tree_biomes.md)

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
4. **Using arena-trained agents in the natural world.** Agents scoring 26–32 in the arena get

   ~5 logs in forests (random actions: 1.8; random spawns, older setup). The arena doesn't
    teach enough to transfer.
    [Random spawns](results/other_evaluations.md)

## MineRL models in CraftGround

Models trained in the MineRL project (behavioural cloning on VPT videos, then PPO) run here unchanged
in "MineRL mode": their own 64×64 view with the hand, `[move, yaw, pitch]` actions, 4 ticks per
decision. Details and every table: [MineRL transfer](experiments/minerl_transfer.md).

1. **They transfer.** With no training here they keep 75-91% of their MineRL score (scored MineRL's way),

   but trail our agents under our rules: Impala PPO 29.4 vs Line 5's 40.0 (−10.6 ± 3.8).
2. **Fine-tuned here, they beat pure RL, in a quarter of the game time.** Impala PPO: 44.7 oak/birch /

   38.5 all-biome after 2M ticks (Line 5 after 4M: 40.0 / 31.6; +4.7 ± 3.9 and +6.9 ± 4.0). It levels off
    after 1-1.5M ticks; pure RL levelled off at ~41 after 3M.
3. **MineRL's simulator isn't needed.** The BC-only model (never run in MineRL), fine-tuned for 1M

   ticks, ties Line 5 on the oak/birch spots (40.4, +0.4 ± 3.5) and beats it on the all-biome spots (37.5,
    +5.8 ± 3.7). The MineRL PPO start is worth ~3 logs on the oak/birch spots.

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

## Bugs found, and what they affected

- **Resumed runs recorded the wrong PPO settings** (fixed 2026-11-20). A resumed model keeps its own
batch size, epochs, entropy and target_kl, but config.json recorded train.py's defaults. Training was
never affected; the two MineRL fine-tune configs were corrected by hand (`config_note`).
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
