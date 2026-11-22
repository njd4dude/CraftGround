# Experiments: MineRL models in CraftGround

Models trained in the MineRL project (behavioural cloning on VPT videos, then PPO), run here in "MineRL mode" (their own view and actions; set up in [Environments](../reference/environments.md)): how well they transfer, and what fine-tuning them here gives. Written by hand. _Last updated 2026-11-21._

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

**How these are scored, unless a line says otherwise:** natural world, one of the two sets of 100
fixed spots (oak and birch forests, or every biome with trees), 1,000-tick episodes, no log cap, our
one-hit axe: the same rules as our own agents' results, so the numbers compare directly with Line 5.
"±" is the 95% range of a spot-by-spot difference; a range that doesn't include 0 is a real
difference. Tables: [oak and birch forests](../results/natural_oak_birch_forests.md),
[every biome with trees](../results/natural_all_tree_biomes.md), [arena](../results/arena.md);
MineRL's own scoring and the iron axe: [other evaluations](../results/other_evaluations.md).

## Main results

Fine-tuning ran in MineRL mode (the model's own view and actions, 4 ticks per decision) on wooded
spawns (`--spawn wooded`: any biome with trees), like Line 5. Training is given in game ticks, so it
compares with our agents' steps. **Both MineRL models start from BC:** "BC→PPO" is the BC model below
plus 120k PPO steps in MineRL's simulator; "BC" is behavioural cloning alone.

| Model | Trained here | Oak/birch forests | Every biome with trees |
| --- | --- | --- | --- |
| MineRL Impala BC→PPO | 0 (zero-shot) | 29.4 | 25.7 |
| | 0.48M ticks | 38.5 | |
| | 1M ticks | 43.4 | 37.2 |
| | 1.5M ticks | 44.5 | 39.0 |
| | **2M ticks** | **44.7** | **38.5** |
| MineRL Impala BC (no MineRL simulator) | 0 (zero-shot) | 22.2 | 21.1 |
| | 0.48M ticks | 37.2 | |
| | 1M ticks | 40.4 | 37.5 |
| Pure RL (Line 4 → 4b → 4c → 5) | 1M / 2M / 3M ticks | 26.6 / 35.7 / 41.2 | |
| | **Line 5**, 4M ticks | **40.0** | **31.6** |
| PPO from scratch in MineRL mode (ImpalaCNN, no BC) | 1M ticks = 252k decisions | 14.6 | 14.3 |
| Line 4 at the same number of decisions | 260k steps = 0.26M ticks | 12.8 | |

## What we found

1. **MineRL models transfer without fine-tuning.** Scored the way MineRL scored them (4,000 ticks, iron
   axe, ends at 64 logs, first 50 oak/birch spots) they keep 75-91% of their MineRL score: Impala BC→PPO
   54.0 (59.6 in MineRL), NatureCNN BC→PPO 48.4 (59.8), Impala BC 44.1 (58.8). The arena is no harder for
   them than oak/birch forests (Impala BC→PPO, iron axe: 21.4 vs 19.6).
2. **Zero-shot, our pure-RL agents are ahead:** Impala BC→PPO 29.4 vs Line 5's 40.0 on the oak/birch
   spots (−10.6 ± 3.8).
3. **Fine-tuned, the MineRL model is our best agent.** At 2M ticks it beats Line 5 on both spot sets:
   oak/birch +4.7 ± 3.9, every biome with trees +6.9 ± 4.0. It levels off after 1-1.5M ticks: the
   1.5M and 2M checkpoints score the same on both sets (oak/birch 44.5 vs 44.7, +0.1 ± 3.8; every biome
   with trees 39.0 vs 38.5, −0.5 ± 3.7), so either is "the best model". Pure RL levelled off at ~41
   after 3M.
4. **MineRL's simulator isn't needed.** The BC model (VPT videos only, never run in MineRL), fine-tuned
   the same way for 1M ticks, ties Line 5 on the oak/birch spots (+0.4 ± 3.5) and beats it on every
   biome with trees (+5.8 ± 3.7). The 120k PPO steps in MineRL are worth ~3 logs on the oak/birch spots
   (−2.9 ± 2.6 for BC) and nothing on every biome with trees (+0.3 ± 3.3).
5. **The axe matters a lot; it is one setting that differs from MineRL.** Zero-shot, the one-hit axe
   gives +6 to +10 logs over MineRL's iron axe (Impala BC→PPO 19.6 → 29.4 on the oak/birch spots): the model's
   aim-and-hold behaviour just finishes each log sooner. The one-hit axe is a diamond axe, so it also
   looks different in the hand.
6. **The pretraining is what makes the MineRL model good, not MineRL's view and actions.** PPO from
   scratch in MineRL mode, the same network, settings, spawns and 1M ticks as the BC→PPO fine-tune but
   from random weights, scores 14.6 on the oak/birch spots and 14.3 on every biome with trees:
   −28.8 ± 2.9 and −22.9 ± 3.5 vs the fine-tune at 1M ticks (−25.9 ± 2.8 and −23.2 ± 2.9 vs the BC
   fine-tune). Starting from VPT videos is worth ~25-29 logs.
7. **MineRL's view and actions learn about as fast as ours per decision, slower per tick.** Compared
   with Line 4 (pure RL, our setup) on the oak/birch spots:
   - **Same decisions** (~250k): 14.6 vs Line 4 at 260k steps 12.8, +1.8 ± 1.6 (better on 58 spots,
     worse on 35): slightly ahead.
   - **Same game time** (1M ticks): 14.6 vs 26.6, −12.0 ± 3.1. MineRL mode makes one decision every
     4 ticks, so in 1M ticks it gets a quarter of Line 4's decisions, samples and PPO updates.
   Two things still favour Line 4: it trained on oak/birch spawns (this run: wooded), and it used our
   PPO settings (LR 2.5e-4, entropy 0.01) where this run kept the fine-tune's gentler ones (LR 1e-4,
   entropy 0.001), which suit a model that already knows the task.
8. **Over 4,000 ticks Line 5 averages 118.9,** about 3× its 1,000-tick score, not 4×: it slows as nearby
   trees run out. With MineRL's 64-log cap it scores 54.6.

## The questions

1. **How good is a MineRL model here, with and without fine-tuning?** Answered (1-3). Arena: zero-shot only.
2. **Can pure RL here match or beat it?** No (3, 4, 6). PPO from scratch in MineRL's own spaces is
   ~29 logs behind the fine-tune at 1M ticks; per decision it learns about as fast as our setup (7).
   Not yet run: the same with our PPO settings (LR 2.5e-4, entropy 0.01).
3. **How much does matching the observations and actions matter?** One test so far, the axe (5).
   Not yet run: hand hidden, our 114×64 view, 1 tick per decision.
4. **Does a fine-tuned model still improve after switching to continuous controls?** Not started.

## Next

1. The matching tests for question 3 (zero-shot, ~10 min each).
2. Optional: PPO from scratch in MineRL's spaces with our PPO settings, to see how much of the gap
   to Line 4 at 1M ticks was the gentle fine-tune settings (question 2).
3. A continuous action head on the 2M-tick model (question 4).
4. Optional: the BC fine-tune to 2M ticks, to see whether it catches up the last ~3 oak/birch logs.

## Training runs, every piece

All in `runs/minerl_finetuned/`: wooded spawns, one-hit axe, MineRL's PPO settings (batch 64,
10 epochs, entropy 0.001, target_kl 0.05), linear learning rate, 2 instances (3 ran out of memory).
`train.py` lists only runs that ended normally in the [training log](../logs/training.md), so all pieces are here.
Decisions are counted from the MineRL model's start; 1 decision = 4 ticks. The from-scratch run uses
`train.py --actions minerl --network impala` with the same settings.

| Model | Piece (folder) | From | Decisions | Ended |
| --- | --- | --- | --- | --- |
| Impala BC→PPO | `ft_impala-ppo_wooded_one-hit-axe_20261120_115737` | MineRL Impala BC→PPO, LR 1e-4 | 0 → 60k | stopped at low memory |
| | `ft_impala-ppo_wooded_one-hit-axe_20261120_124557` | its 60k checkpoint | 60k → 252k | completed: **the 1M-tick model** (`model.zip`) |
| | `ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` | the 1M-tick model, LR 1e-4 again | 252k → 492k | froze 10k decisions from the end; **the 2M-tick model** is `runs/minerl_finetuned/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_615292_steps.zip` |
| Impala BC | `ft_impala-bc_wooded_one-hit-axe_20261120_193516` | MineRL Impala BC, LR 1e-4, 5-update value warm-up | 0 → 188k | paused by hand |
| | `ft_impala-bc_wooded_one-hit-axe_20261120_220917` | its 180k checkpoint | 180k → 254k | completed: **the BC 1M-tick model** (`model.zip`) |
| From scratch (Impala) | `scratch_impala_20261121_170105` | random weights, LR 1e-4 | 0 → 241k | an instance froze; a watchdog killed it after 20 silent minutes |
| | `scratch_impala_20261121_193751` | its 240k checkpoint | 240k → 252k | completed: **the from-scratch 1M-tick model** (`model.zip`) |

Two BC launches crashed on `train.py` bugs before training (fixed; see the bug list in [Findings](../FINDINGS.md)).
