"""Learning-rate schedules."""


def linear_lr(lr0: float, start_steps: int, end_steps: int):
    """Learning rate falling linearly from lr0 at start_steps to 0 at end_steps (this run's span).

    SB3 calls it with progress_remaining = 1 - steps / end_steps, counted from step 0 even when
    resuming, so a resumed run would otherwise start part-way down. Converting back to steps keeps
    the full lr0 at this run's first update whatever the model's age."""
    span = max(end_steps - start_steps, 1)

    def schedule(progress_remaining: float) -> float:
        steps = (1.0 - progress_remaining) * end_steps
        return lr0 * min(1.0, max(0.0, 1.0 - (steps - start_steps) / span))
    return schedule
