"""Train a PPO agent to chop as many trees as possible.

    python train.py                       # train with defaults
    python train.py --timesteps 500000    # train longer
    python train.py --resume runs/arena/chop_ppo_20261113_131800/checkpoints/ppo_chop_100000_steps.zip
    python train.py --no-wandb            # train without logging to Weights & Biases
    python train.py --learning-rate 1e-4 --n-epochs 8 --notes "slower lr, more epochs"
    python train.py --n-envs 4            # run 4 Minecraft instances instead of 6
    python train.py --env natural                            # train in a forest in a normal world (3 instances)
    python train.py --env natural --resume runs/arena/chop_ppo_20261113_135620/model.zip   # fine-tune an arena model there
    python train.py --env natural --no-jump                  # natural world with the arena's 11 actions (no jump)

--env picks the world: arena (chop/envs/arena.py, walled superflat arena, the default)
or natural (chop/envs/natural.py, a fresh forest spot every episode). Both have the
same images, but the natural world adds jump and forward_jump (13 actions, see
common.JUMP_ACTIONS) because of its hills and ledges. 11-action models still --resume
there (jump is turned off automatically to match); --no-jump forces the 11 actions.
Without jump the two worlds share the same actions, so --resume works across them. Natural runs get their
own folders (runs/natural/<line>/<time>/, grouped by training line: --resume continues the
model's line, --line names a new one), CSV (results/training_runs_natural.csv) and
report (EXPERIMENTS_NATURAL.md). The natural world is slower (about a third of
the arena's steps/s in evaluation) and needs more memory per instance, so it
defaults to 3 instances (see N_ENVS_NATURAL).

Training runs --n-envs Minecraft instances side by side (default 6, natural 3). Each
needs ~2.4 GB of RAM in the arena and a lot of CPU. In the natural world each grows to
~4.2 GB (memprobe.py: Java fills its 2G heap cap, native memory levels off after ~10
minutes, plus ~0.5 GB of Gradle launcher processes), so 4 would leave almost no RAM free. Measured raw env
speed on this machine (8-core 7700X, BENCHMARK.md): arena 1 -> 487, 2 -> 713,
4 -> 977, 6 -> 1046 steps/s; natural 4 -> 235 steps/s with the current 0.75 s
forest-check timeout (1 -> 48, 2 -> 120, 6 -> 263 were measured with the old 3 s
timeout, so they understate it). Training runs ~20% below these raw speeds
(model inference each step, PPO updates): 4 natural instances trained at ~173.

Each instance runs from its own copy of CraftGround's game folder
(~/craftground_instances/mc121_port<port>, see chop/infra/instances.py), because instances
started together in CraftGround's one shared folder sometimes collide and one
fails to start, which hangs the whole run. --shared-folder goes back to the
shared folder. The first run on a new port is ~75 s slower while its copy builds.
--n-steps is the rollout size summed over all instances, so changing --n-envs
keeps each PPO update the same size (each instance collects n_steps / n_envs).

The learning rate falls linearly from --learning-rate to 0 over each run (--lr-schedule,
default linear). With a constant rate the natural-world jump model peaked at ~600k steps
(21.7 logs per episode) and slid to 15.0 by 1M. A resumed run starts the schedule over.

Every PPO setting in PPO_SETTINGS has a flag (--n-steps, --batch-size, --n-epochs,
--learning-rate, --gamma, --ent-coef, --clip-range). Use --notes to say what's
special about a run (e.g. a reward change); it shows up in EXPERIMENTS.md.

Live graphs (logs chopped per episode, reward, losses) go to Weights & Biases,
project "craftground-chop-trees". Log in once first with:  wandb login

Every run gets one folder with everything in it:

    runs/arena/chop_ppo_20261113_131800/      (natural world: runs/natural/<line>/<time>/)
        config.json            this run's settings, changes from the defaults, notes
        model.zip              final model (saved on finish, Ctrl+C or crash)
        checkpoints/           a model every 20,000 steps
        monitor_0.monitor.csv  one line per training episode (one file per instance)
        wandb/                 W&B's local copy of the run

Press Ctrl+C at any time: the current model is saved before exiting.
Each run is added to the Training runs table in EXPERIMENTS.md (or
EXPERIMENTS_NATURAL.md), with a link to its W&B page.
"""

from chop.training.cli import main

if __name__ == "__main__":
    main()
