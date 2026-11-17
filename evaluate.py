"""Measure how many logs an agent chops per episode.

    python evaluate.py --random                              # baseline: random actions
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip               # a trained model
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --episodes 10 # quick check (default 100)
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --record 0    # no videos
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --record 34   # video of every episode (3 x 34 >= 100)
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --env natural # forest in a normal world
    python evaluate.py runs/arena/chop_ppo_20261113_131800/model.zip --env natural --held-out  # a world never trained in

Runs 100 episodes by default so the mean is stable. Every run counts toward the
agent's score in EXPERIMENTS.md (EXPERIMENTS_NATURAL.md for --env natural), so to just grab a couple of videos use a small
--episodes, e.g. --episodes 2.

Episodes are spread over --n-envs Minecraft instances running side by side
(default 6, natural world 3, never more than --episodes), each running from its own copy of the
game folder (chop/infra/instances.py; --shared-folder for CraftGround's one shared folder,
where instances started together sometimes collide and hang the run). Each instance gets a fixed share of the
episodes decided up front (100 over 6 = 17,17,17,17,16,16) and only its first
share counts, so no half-played episode is ever dropped. (Stopping at "the first
100 to finish" would favour short episodes, and here short means good, because
an episode ends early once every log is chopped.)

(The model path for each training run is listed in EXPERIMENTS.md.)

21 episodes are saved as MP4 videos by default, picked by score once every episode is
done so they show where the agent succeeds and fails: the 7 best (file names start
top1..top7, top1 = most logs), the 7 worst (low1..low7, low1 = fewest) and the 7 around
the median (mid1..mid7, lowest first). --record N picks N per group (--record 0 for no
videos); when 3N covers every episode, all of them are saved. Ties go to the earlier
episode. All episodes still count toward the score; only the videos are limited.
Choosing after the fact means every episode's frames are kept until the end: they're
written as compressed .npz files to a temporary _frames_<time> folder next to the videos
(roughly 5-15 MB per episode) and deleted once the chosen ones are encoded. The encoding
happens after Minecraft has closed, so it doesn't slow the playing or the timing.

Each video is ~5 MB at real speed. The videos show what the agent sees,
enlarged 4x, with the step, logs chopped and current action written on each frame.
They open in any video player and go in the model's run folder under
videos/arena/ or videos/natural/, depending on --env (random agent:
runs/baselines/random/videos/<env>/).

Videos play at real speed by default. The agent acts every tick, so that looks frantic;
--video-fps adds slower copies: --video-fps 20 5 also saves a quarter-speed copy
(each step shown for 1/5 s, file name ends in _5fps).

At the end it prints how long the run took and how fast it went: launch time
(starting Minecraft), playing time, env steps per second (total and per
instance), and how much of each instance's time went into resets. Those numbers
are saved with the results too.

Compare the trained model against the random baseline: that gap is what the
agent actually learned. Every run is appended to results/eval_results.csv
and the tables in EXPERIMENTS.md are refreshed.
"""

from chop.evaluation.cli import main

if __name__ == "__main__":
    main()
