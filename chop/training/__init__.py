"""PPO training (train.py is the command line).

    cli        the command line: settings, run folder, W&B, the training loop
    settings   PPO defaults and the environment settings saved with every run
    schedule   the linear learning-rate schedule
    callbacks  what gets logged during training (logs per episode, speed) and to W&B
    lines      which natural-world training line a run belongs to
"""
