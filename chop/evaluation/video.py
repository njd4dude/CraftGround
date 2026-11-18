"""Videos of evaluation episodes: what the agent saw, with step / logs / action on every frame.

Which episodes get a video is decided once all are played (the best, worst and median ones,
pick_video_episodes), so EpisodeRecorder keeps every episode's small frames on disk until then:

    rec = EpisodeRecorder(video_dir, stamp, speeds, label, max_logs)   # label(action) -> text
    rec.add_frame(i, image, step, logs, action)    # every step of instance i
    rec.end_episode(i, episode)                     # written to _frames_<stamp>/ep<N>.npz in the background
    rec.finish(played_all)                          # after the envs are closed
    rec.encode(scores, per_group)                   # the chosen episodes -> MP4, then the frames are deleted
"""

import shutil
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

VIDEO_SCALE = 4
VIDEO_FPS = 20  # one step is one game tick (1/20 s), so videos play at real Minecraft speed
MAX_PENDING = 8  # episodes waiting to be written before playing pauses to let writing catch up
RECORD_PER_GROUP = 7  # videos of the best, worst and median episodes, this many each

ActionLabel = Callable[[object], str]  # one action (an index, or a (move, turn, look) list) -> its text


def video_frame(image: np.ndarray, step: int, logs: int, action, label: ActionLabel,
                max_logs: int | None = None) -> np.ndarray:
    """Enlarge the agent's view and write step / logs / action across the top."""
    h, w = image.shape[:2]
    frame = Image.fromarray(image).resize((w * VIDEO_SCALE, h * VIDEO_SCALE), Image.NEAREST)
    draw = ImageDraw.Draw(frame)
    draw.rectangle([0, 0, frame.width, 16], fill=(0, 0, 0))
    cap = f"/{max_logs}" if max_logs else ""
    draw.text((4, 2), f"step {step:4d}   logs {logs:2d}{cap}   {label(action)}", fill=(255, 255, 255))
    return np.asarray(frame)


def save_video(frames: list[np.ndarray], path: Path, fps: int = VIDEO_FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # H.264 + yuv420p plays in Windows' Media Player, VS Code, browsers and VLC.
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  saved video {path} ({path.stat().st_size / 1e6:.1f} MB)")


def save_episode_videos(raw: list[tuple], base: Path, speeds: list[int], label: ActionLabel,
                        max_logs: int | None = None, step_ticks: int = 1) -> None:
    """Build the annotated frames from the raw (image, step, logs, action) records and write
    one video per speed (file name ends in _<fps>fps except at real speed).
    speeds are game ticks per second; with step_ticks ticks per frame (action repeat) the video
    plays at speed / step_ticks frames per second, so real speed stays real speed."""
    frames = [video_frame(*r, label=label, max_logs=max_logs) for r in raw]
    for fps in speeds:
        speed = "" if fps == VIDEO_FPS else f"_{fps}fps"
        save_video(frames, base.with_name(f"{base.name}{speed}.mp4"), fps=max(1, round(fps / step_ticks)))


def save_raw_episode(raw: list[tuple], path: Path) -> None:
    """Keep one episode's small frames + labels on disk until we know whether it gets a video."""
    images, steps, logs, actions = zip(*raw)
    np.savez_compressed(path, images=np.stack(images), steps=np.array(steps),
                        logs=np.array(logs), actions=np.array(actions))


def encode_saved_episode(path: Path, base: Path, speeds: list[int], label: ActionLabel,
                         max_logs: int | None = None, step_ticks: int = 1) -> None:
    with np.load(path) as d:
        raw = list(zip(d["images"], d["steps"].tolist(), d["logs"].tolist(), d["actions"].tolist()))
    save_episode_videos(raw, base, speeds, label, max_logs, step_ticks)


def pick_video_episodes(results: dict[int, int], per_group: int) -> list[tuple[str, int]]:
    """(label, episode) for the per_group best, worst and median-most episodes.
    Labels: top1 = most logs, low1 = fewest, mid1..midN = around the median, lowest first.
    Ties go to the earlier episode; an episode in two groups is saved once (top, then low, then mid)."""
    if per_group <= 0:
        return []
    order = sorted(results, key=lambda e: (results[e], e))           # fewest logs first
    best = sorted(results, key=lambda e: (-results[e], e))[:per_group]
    worst = order[:per_group]
    start = max(0, min(len(order) - per_group, len(order) // 2 - per_group // 2))
    middle = order[start:start + per_group]
    picked, seen = [], set()
    for prefix, group in (("top", best), ("low", worst), ("mid", middle)):
        for rank, ep in enumerate(group, 1):
            if ep not in seen:
                seen.add(ep)
                picked.append((f"{prefix}{rank}", ep))
    return picked


class EpisodeRecorder:
    """Collects every episode's frames per instance and, at the end, encodes the chosen ones.
    Disabled (does nothing) when enabled is False, so callers needn't check."""

    def __init__(self, video_dir: Path, stamp: str, speeds: list[int], label: ActionLabel,
                 max_logs: int | None, n_envs: int, enabled: bool = True, step_ticks: int = 1):
        self.video_dir, self.stamp, self.speeds = video_dir, stamp, speeds
        self.label, self.max_logs, self.step_ticks = label, max_logs, step_ticks
        self.enabled = enabled and bool(speeds)
        self.frames_dir = video_dir / f"_frames_{stamp}"  # every episode's frames until the videos are chosen
        self._raw: list[list[tuple]] = [[] for _ in range(n_envs)]  # small frames; enlarged only when encoding
        self._writer = ThreadPoolExecutor(max_workers=2)  # writes frames while Minecraft keeps playing
        self._pending: list[Future] = []
        if self.enabled:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def add_frame(self, i: int, image: np.ndarray, step: int, logs: int, action) -> None:
        if self.enabled:
            self._raw[i].append((np.array(image), step, logs, np.asarray(action).tolist()))

    def end_episode(self, i: int, episode: int) -> None:
        """Instance i finished `episode` (0-based): write its frames, start a fresh list."""
        if self.enabled:
            self._pending.append(self._writer.submit(save_raw_episode, self._raw[i],
                                                     self.frames_dir / f"ep{episode + 1}.npz"))
        self._raw[i] = []

    def throttle(self) -> None:
        """Wait while too many episodes are queued for writing, so they don't pile up in memory."""
        while sum(not f.done() for f in self._pending) > MAX_PENDING:
            next(f for f in self._pending if not f.done()).result()

    def finish(self, played_all: bool) -> None:
        """Wait for the frame writes. If the run stopped early there's no score, so no videos either:
        wait without raising (the error that stopped the run is the one to report) and delete the frames."""
        for f in self._pending:
            if played_all:
                f.result()  # surface any write error
            else:
                f.exception()
        self._writer.shutdown()
        if not played_all:
            self.discard()

    def discard(self) -> None:
        shutil.rmtree(self.frames_dir, ignore_errors=True)

    def encode(self, scores: dict[int, int], per_group: int) -> None:
        """Encode the best, worst and median episodes (per_group each), then delete all frames."""
        if not self.enabled:
            return
        picked = pick_video_episodes(scores, per_group)
        print(f"\nEncoding {len(picked)} videos (best, worst and median episodes) to {self.video_dir}")
        try:
            with ThreadPoolExecutor(max_workers=4) as pool:  # Minecraft is closed, so the CPU is free
                jobs = [pool.submit(encode_saved_episode, self.frames_dir / f"ep{ep + 1}.npz",
                                    self.video_dir / f"eval_{self.stamp}_{tag}_ep{ep + 1}_{scores[ep]}logs",
                                    self.speeds, self.label, self.max_logs, self.step_ticks)
                        for tag, ep in picked]
                for job in jobs:
                    job.result()
        finally:
            self.discard()
