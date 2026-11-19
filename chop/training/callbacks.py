"""What training logs: logs chopped per episode, speed per PPO iteration, and SB3's own stats, to W&B."""

import time

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter
from stable_baselines3.common.vec_env import VecEnvWrapper


class WandbOutputFormat(KVWriter):
    """Sends SB3's training stats (losses, rewards, fps...) to W&B instead of TensorBoard."""

    def write(self, key_values, key_excluded, step=0):
        data = {k: v for k, v in key_values.items() if isinstance(v, (int, float, np.number))}
        if data and wandb.run is not None:
            wandb.log({"global_step": step, **data})

    def close(self):
        pass


class LogsChoppedCallback(BaseCallback):
    """Logs every finished episode's log count to W&B and the console.
    ticks_per_step: the env's action repeat, so chop/game_ticks compares runs with different repeats
    on game time (set it as the x-axis in W&B).
    Every hist_every episodes it also logs chop/logs_per_episode_hist, the distribution of those
    episodes' log counts (one bin per log); W&B shows it as a histogram that changes over steps."""

    def __init__(self, ticks_per_step: int = 1, hist_every: int = 50):
        super().__init__()
        self.ticks_per_step = ticks_per_step
        self.hist_every = hist_every
        self.recent: list[int] = []
        self.window: list[int] = []
        self.episodes = 0
        self.best = 0

    def _log_histogram(self) -> None:
        counts, edges = np.histogram(self.window, bins=np.arange(0, max(self.window) + 2))
        wandb.log({
            "global_step": self.num_timesteps,
            "chop/logs_per_episode_hist": wandb.Histogram(np_histogram=(counts, edges)),
        })
        self.window = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                logs = info.get("logs_chopped", 0)
                self.episodes += 1
                self.best = max(self.best, logs)
                self.recent = (self.recent + [logs])[-20:]
                avg = float(np.mean(self.recent))
                if wandb.run is not None:
                    wandb.log({
                        "global_step": self.num_timesteps,
                        "chop/logs_per_episode": logs,
                        "chop/logs_per_episode_avg20": avg,
                        "chop/episode": self.episodes,
                        "chop/game_ticks": self.num_timesteps * self.ticks_per_step,
                    })
                    self.window.append(logs)
                    if len(self.window) >= self.hist_every:
                        self._log_histogram()
                print(f"[{self.num_timesteps:>8} steps] episode done: {logs:2d} logs (last-20 avg {avg:.1f})")
        return True


class StepTimer(VecEnvWrapper):
    """Adds up the wall time spent inside env.step (all Minecraft instances, including resets)."""

    def __init__(self, venv):
        super().__init__(venv)
        self.seconds = 0.0
        self._t = 0.0

    def step_async(self, actions):
        self._t = time.perf_counter()
        self.venv.step_async(actions)

    def step_wait(self):
        result = self.venv.step_wait()
        self.seconds += time.perf_counter() - self._t
        return result

    def reset(self):
        return self.venv.reset()


class SpeedCallback(BaseCallback):
    """Logs current training speed per PPO iteration (SB3's time/fps averages over the whole run).

    time/env_fps              env steps/s counting only time inside env.step: pure Minecraft speed
    time/rollout_fps          env steps/s while collecting, summed over all instances
                              (Minecraft + policy inference + callbacks, incl. checkpoint saves)
    time/rollout_fps_per_env  rollout_fps / n_envs (one instance's effective speed)
    time/iter_fps             env steps/s over rollout + PPO update (real training throughput);
                              measured start-to-start, so it describes the previous iteration
    time/update_seconds       seconds from rollout end to the next rollout start: the PPO update
                              (backprop over all epochs/minibatches); also describes the previous iteration
    """

    def __init__(self, n_envs: int, step_timer: StepTimer):
        super().__init__()
        self.n_envs = n_envs
        self.step_timer = step_timer
        self.rollout_start = None
        self.rollout_end = None
        self.rollout_start_steps = 0

    def _on_rollout_start(self) -> None:
        now = time.perf_counter()
        if self.rollout_start is not None:
            steps = self.num_timesteps - self.rollout_start_steps
            self.logger.record("time/iter_fps", steps / (now - self.rollout_start))
        if self.rollout_end is not None:
            self.logger.record("time/update_seconds", now - self.rollout_end)
        self.rollout_start = now
        self.rollout_start_steps = self.num_timesteps
        self.step_timer.seconds = 0.0

    def _on_rollout_end(self) -> None:
        steps = self.num_timesteps - self.rollout_start_steps
        if self.step_timer.seconds > 0:
            self.logger.record("time/env_fps", steps / self.step_timer.seconds)
        self.rollout_end = time.perf_counter()
        fps = steps / (self.rollout_end - self.rollout_start)
        self.logger.record("time/rollout_fps", fps)
        self.logger.record("time/rollout_fps_per_env", fps / self.n_envs)

    def _on_step(self) -> bool:
        return True
