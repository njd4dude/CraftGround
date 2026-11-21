"""Spot-by-spot comparisons on the fixed evaluation spots."""

from chop.reports.markdown import mean_ci

# Our reference agent: every fixed-spot table compares each agent with it, spot by spot.
REFERENCE = "runs/natural/line5_wooded-spawns/20261119_005447/model.zip"
REFERENCE_NAME = "Line 5"


def per_spot(rows: list[dict]) -> dict[int, float]:
    """Score per fixed spot from evaluation rows, averaged if a spot was played more than once."""
    acc: dict[int, list[int]] = {}
    for r in rows:
        spots = (r.get("spot_per_episode") or "").split()
        for s, x in zip(spots, r["per_episode"].split()):
            if s.isdigit():
                acc.setdefault(int(s), []).append(int(x))
    return {s: sum(v) / len(v) for s, v in acc.items()}


def paired(a: dict[int, float], b: dict[int, float], short: bool = False) -> str:
    """Mean difference a - b over the spots both played, with its 95% range (and better / worse counts)."""
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return "–"
    d = [a[s] - b[s] for s in shared]
    mean, ci = mean_ci(d)
    better, worse = sum(x > 0 for x in d), sum(x < 0 for x in d)
    if short:
        return f"{mean:+.1f} ± {ci:.1f}"
    return f"{mean:+.1f} ± {ci:.1f} ({better} better / {worse} worse, {len(shared)} spots)"
