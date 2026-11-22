# Experiments: MineRL models in CraftGround

Models trained in the MineRL project (behavioural cloning on VPT videos, then PPO), run here in "MineRL mode" (their own view and actions; set up in [Environments](../reference/environments.md)): how well they transfer, and what training them further here in CraftGround gives. Written by hand. _Last updated 2026-11-22._

[← Project README](../../README.md) · [Findings](../FINDINGS.md)

**How these are scored, unless a line says otherwise:** natural world, one of the two sets of 100
fixed spots (oak and birch forests, or every biome with trees), 1,000-tick episodes, no log cap, our
one-hit axe: the same rules as our own agents' results, so the numbers compare directly with Line 5.
"±" is the 95% range of a spot-by-spot difference; a range that doesn't include 0 is a real
difference. Tables: [oak and birch forests](../results/natural_oak_birch_forests.md),
[every biome with trees](../results/natural_all_tree_biomes.md), [arena](../results/arena.md);
MineRL's own scoring and the iron axe: [other evaluations](../results/other_evaluations.md).

## Models in this doc

Every model has one name, and the name lists its training stages. A MineRL model can have up to three:

1. **BC:** behavioural cloning on VPT videos (human Minecraft play from YouTube), in the MineRL project.
   No reward, no simulator.
2. **→PPO:** 120k PPO steps in MineRL's simulator (target_kl 0.05), in the MineRL project.
3. **+ CG &lt;ticks&gt;:** PPO here in CraftGround, in MineRL mode (64×64 view with the hand,
   `[move, yaw, pitch]` actions, 4 ticks per decision), natural world, wooded spawns (`--spawn wooded`:
   any biome with trees, like Line 5), one-hit axe. Counted in game ticks, so it compares with our
   lines' steps (1 step = 1 tick).

"Impala" / "NatureCNN" is the network. "Zero-shot" means no stage 3.

| Name | Stage 1: BC on VPT | Stage 2: PPO in MineRL | Stage 3: PPO in CraftGround | PPO settings in stage 3 |
| --- | --- | --- | --- | --- |
| **Impala BC, zero-shot** | yes | – | – | – |
| **Impala BC→PPO, zero-shot** | yes | 120k steps | – | – |
| **NatureCNN BC→PPO, zero-shot** | yes | 120k steps | – | – |
| **Impala BC + CG 1M** | yes | – | 1M ticks | gentle |
| **Impala BC→PPO + CG 2M** (our best agent) | yes | 120k steps | 2M ticks | gentle |
| **Scratch Impala (MineRL mode) + CG 1M, gentle PPO settings** | – (random weights) | – | 1M ticks | gentle |
| **Scratch Impala (MineRL mode) + CG 1M, Line 4 PPO settings** | – (random weights) | – | 1M ticks | Line 4's |
| **Line 4 / Line 5** (pure RL, our setup) | – | – | 1M / 4M ticks, our 114×64 view and actions, 1 tick per decision, NatureCNN | Line 4's |

"Impala BC→PPO + CG 1M" is the 1M-tick checkpoint of the same training, and so on.

- **Gentle PPO settings** (meant for a model that already knows the task): LR 1e-4 linear, entropy
  0.001, batch 64, 10 epochs, target_kl 0.05, n_steps 4094.
- **Line 4 PPO settings** (SB3 defaults, used by all our lines): LR 2.5e-4 linear, entropy 0.01,
  batch 256, 4 epochs, no target_kl, n_steps 2048.

## Main results

| Model | Oak/birch forests | Every biome with trees |
| --- | --- | --- |
| Impala BC→PPO, zero-shot | 29.4 | 25.7 |
| Impala BC→PPO + CG 480k | 38.5 | |
| Impala BC→PPO + CG 1M | 43.4 | 37.2 |
| Impala BC→PPO + CG 1.5M | 44.5 | 39.0 |
| **Impala BC→PPO + CG 2M** | **44.7** | **38.5** |
| Impala BC, zero-shot | 22.2 | 21.1 |
| Impala BC + CG 480k | 37.2 | |
| Impala BC + CG 1M | 40.4 | 37.5 |
| Scratch Impala (MineRL mode) + CG 1M, gentle PPO settings | 14.6 | 14.3 |
| Scratch Impala (MineRL mode) + CG 1M, Line 4 PPO settings | 18.9 | 18.4 |
| Line 4 @ 260k steps (same number of decisions as the Scratch runs: ~250k) | 12.8 | |
| Line 4 @ 1M steps (same game time as the Scratch runs) | 26.6 | |
| Line 4b @ 2M / Line 4c @ 3M steps | 35.7 / 41.2 | |
| **Line 5** @ 4M steps (best pure RL) | **40.0** | **31.6** |

## What we found

1. **MineRL models transfer without fine-tuning.** Scored the way MineRL scored them (4,000 ticks, iron
   axe, ends at 64 logs, first 50 oak/birch spots), the three zero-shot models keep 75-91% of their
   MineRL score: Impala BC→PPO, zero-shot 54.0 (59.6 in MineRL); NatureCNN BC→PPO, zero-shot 48.4
   (59.8); Impala BC, zero-shot 44.1 (58.8). The arena is no harder for Impala BC→PPO, zero-shot than
   oak/birch forests (iron axe: 21.4 vs 19.6).
2. **Zero-shot, our pure-RL agents are ahead:** Impala BC→PPO, zero-shot 29.4 vs Line 5's 40.0 on the
   oak/birch spots (−10.6 ± 3.8).
3. **Impala BC→PPO + CG 2M is our best agent.** Impala BC→PPO + CG 2M beats Line 5 on both spot sets:
   oak/birch +4.7 ± 3.9, every biome with trees +6.9 ± 4.0. Stage 3 levels off after 1-1.5M ticks:
   Impala BC→PPO + CG 1.5M and + CG 2M score the same on both sets (oak/birch 44.5 vs 44.7,
   +0.1 ± 3.8; every biome with trees 39.0 vs 38.5, −0.5 ± 3.7), so either is "the best model". Pure
   RL (Line 4 → 4b → 4c) levelled off at ~41 after 3M ticks.
4. **MineRL's simulator isn't needed.** Impala BC + CG 1M (VPT videos only, never run in MineRL's
   simulator) ties Line 5 on the oak/birch spots (+0.4 ± 3.5) and beats Line 5 on every biome with
   trees (+5.8 ± 3.7). Stage 2 (the 120k PPO steps in MineRL) is worth ~3 logs on the oak/birch spots
   (Impala BC + CG 1M vs Impala BC→PPO + CG 1M: −2.9 ± 2.6) and nothing on every biome with trees
   (+0.3 ± 3.3).
5. **The axe matters a lot; the axe is one setting that differs from MineRL.** Zero-shot, our one-hit
   axe gives +6 to +10 logs over MineRL's iron axe (Impala BC→PPO, zero-shot: 19.6 → 29.4 on the
   oak/birch spots): the model's aim-and-hold behaviour just finishes each log sooner. The one-hit axe
   is a diamond axe, so the axe also looks different in the hand.
6. **Stages 1-2 (pretraining) are what make the MineRL models good, not MineRL's view and actions.**
   Scratch Impala (MineRL mode) + CG 1M, gentle PPO settings has the same network, PPO settings, spawns
   and 1M ticks as Impala BC→PPO + CG 1M, but random starting weights. Scratch Impala (MineRL mode) +
   CG 1M, gentle PPO settings scores 14.6 on the oak/birch spots and 14.3 on every biome with trees:
   −28.8 ± 2.9 and −22.9 ± 3.5 vs Impala BC→PPO + CG 1M (−25.9 ± 2.8 and −23.2 ± 2.9 vs Impala BC +
   CG 1M). Starting from VPT videos is worth ~25-29 logs.
7. **The gentle PPO settings hurt training from scratch.** Scratch Impala (MineRL mode) + CG 1M, Line 4
   PPO settings scores 18.9 oak/birch / 18.4 every biome with trees: +4.4 ± 1.9 and +4.2 ± 2.2 over
   Scratch Impala (MineRL mode) + CG 1M, gentle PPO settings. Scratch Impala (MineRL mode) + CG 1M,
   Line 4 PPO settings is still ~21 logs behind Impala BC + CG 1M (40.4) and ~25 behind Impala BC→PPO
   + CG 1M (43.4), so finding 6 holds.
8. **MineRL mode learns faster than our setup per decision, slower per tick.** With PPO settings
   matched (Line 4's), Scratch Impala (MineRL mode) + CG 1M, Line 4 PPO settings vs Line 4 (our view,
   actions and NatureCNN, also from scratch) on the oak/birch spots:
   - **Same number of decisions** (~250k): 18.9 vs Line 4 @ 260k steps 12.8, +6.2 ± 1.7 (better on 71
     spots, worse on 22). MineRL mode's view, actions and ImpalaCNN together learn clearly faster per
     decision.
   - **Same game time** (1M ticks): 18.9 vs Line 4 @ 1M steps 26.6, −7.7 ± 2.9 (better on 23 spots,
     worse on 69). MineRL mode makes one decision every 4 ticks, so in 1M ticks MineRL mode gets a
     quarter of Line 4's decisions, samples and PPO updates.
   With the gentle settings the same comparison gave +1.8 ± 1.6 and −12.0 ± 3.1. Line 4 trained on
   oak/birch spawns; both Scratch runs trained on wooded spawns. The difference can't be pinned on the
   view, the actions, the 4 ticks per decision or ImpalaCNN alone, because all four change together.
9. **Over 4,000 ticks Line 5 averages 118.9,** about 3× Line 5's 1,000-tick score, not 4×: Line 5
   slows as nearby trees run out. With MineRL's 64-log cap Line 5 scores 54.6.

## The questions

1. **How good is a MineRL model here, with and without stage 3?** Answered (1-3). Arena: zero-shot only.
2. **Can pure RL here match or beat Impala BC→PPO + CG?** No (3, 4, 6). Scratch Impala (MineRL mode)
   + CG 1M is ~29 logs behind Impala BC→PPO + CG 1M with the gentle PPO settings and ~25 behind with
   Line 4's PPO settings (7). Per decision, Scratch Impala (MineRL mode) learns faster than Line 4; per
   tick, Scratch Impala (MineRL mode) learns slower (8).
3. **How much does matching the observations and actions matter?** One test so far, the axe (5).
   Not yet run: hand hidden, our 114×64 view, 1 tick per decision.
4. **Does Impala BC→PPO + CG 2M still improve after switching to continuous controls?** Not started.

## Next

1. The matching tests for question 3 (zero-shot, ~10 min each).
2. A continuous action head on Impala BC→PPO + CG 2M (question 4).
3. Optional: Impala BC + CG to 2M ticks, to see whether Impala BC + CG catches up the last ~3
   oak/birch logs to Impala BC→PPO + CG.

## Training runs, every piece

All in `runs/minerl_finetuned/`: wooded spawns, one-hit axe, the gentle PPO settings unless the row
says otherwise, 2 instances (3 ran out of memory).
`train.py` lists only runs that ended normally in the [training log](../logs/training.md), so all pieces are here.
Decisions are counted from the start of stage 3; 1 decision = 4 ticks. The Scratch runs use
`train.py --actions minerl --network impala`.

| Model | Piece (folder) | From | Decisions | Ended |
| --- | --- | --- | --- | --- |
| Impala BC→PPO + CG | `ft_impala-ppo_wooded_one-hit-axe_20261120_115737` | Impala BC→PPO, zero-shot, LR 1e-4 | 0 → 60k | stopped at low memory |
| | `ft_impala-ppo_wooded_one-hit-axe_20261120_124557` | its 60k checkpoint | 60k → 252k | completed: **Impala BC→PPO + CG 1M** (`model.zip`) |
| | `ft2_impala-ppo_wooded_one-hit-axe_20261120_153415` | Impala BC→PPO + CG 1M, LR 1e-4 again | 252k → 492k | froze 10k decisions from the end; **Impala BC→PPO + CG 2M** is `runs/minerl_finetuned/ft2_impala-ppo_wooded_one-hit-axe_20261120_153415/checkpoints/ppo_chop_615292_steps.zip` |
| Impala BC + CG | `ft_impala-bc_wooded_one-hit-axe_20261120_193516` | Impala BC, zero-shot, LR 1e-4, 5-update value warm-up | 0 → 188k | paused by hand |
| | `ft_impala-bc_wooded_one-hit-axe_20261120_220917` | its 180k checkpoint | 180k → 254k | completed: **Impala BC + CG 1M** (`model.zip`) |
| Scratch Impala (MineRL mode) + CG, gentle PPO settings | `scratch_impala_20261121_170105` | random weights, LR 1e-4 | 0 → 241k | an instance froze; a watchdog killed it after 20 silent minutes |
| | `scratch_impala_20261121_193751` | its 240k checkpoint | 240k → 252k | completed: **Scratch Impala (MineRL mode) + CG 1M, gentle PPO settings** (`model.zip`) |
| Scratch Impala (MineRL mode) + CG, Line 4 PPO settings | `scratch_impala_rl-settings_20261121_210129` | random weights, LR 2.5e-4 | 0 → ~225k | an instance froze; the watchdog killed it |
| | `scratch_impala_rl-settings_20261121_231545` | its 220k checkpoint | 220k → 250k | completed: **Scratch Impala (MineRL mode) + CG 1M, Line 4 PPO settings** (`model.zip`) |

Two BC launches crashed on `train.py` bugs before training (fixed; see the bug list in [Findings](../FINDINGS.md)).
