"""Tree-chopping RL on CraftGround (Minecraft).

    chop.envs        the environments: arena (walled superflat) and natural (generated world)
    chop.infra       running Minecraft instances: per-port game folders, launching, closing
    chop.training    PPO training (train.py is its command line)
    chop.evaluation  scoring agents and recording videos (evaluate.py is its command line)
    chop.reports     the CSVs in results/ and the reports built from them (EXPERIMENTS*.md)

The scripts in the repository root (train.py, evaluate.py, make_eval_spots.py, play.py,
benchmark.py) are the commands; one-off diagnostics live in tools/. Everything runs from the
repository root (run.bat does that), since results/ and runs/ are relative paths.
"""
