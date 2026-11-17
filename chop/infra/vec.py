"""Launch N Minecraft instances side by side as one VecEnv, and close them again.

Every script that runs several instances does the same three things, which used to be
copied into each of them:

    ports = instance_ports(args.port, n)                       # 8023, 8033, 8043, ...
    env = launch_vec_env([factory(p) for p in ports], ports)   # clear leftovers, start them
    try:
        ...
    finally:
        close_vec_env(env, ports)                              # close with a time limit, kill what's left

A factory builds one env inside its worker process; game_folder(port) gives it the
instance's own copy of the game folder (see instances.py for why each port has one).
"""

from typing import Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from chop.infra.instances import clear_stale_loom_locks, close_vec_env, env_path_for, kill_instance_processes

__all__ = ["instance_ports", "game_folder", "launch_vec_env", "close_vec_env"]

PORT_STEP = 10  # instance i listens on first_port + 10 * i


def instance_ports(first_port: int, n: int) -> list[int]:
    return [first_port + PORT_STEP * i for i in range(n)]


def game_folder(port: int, shared_folder: bool = False) -> str | None:
    """env_path for the instance on this port: its own game folder (made and built on first use),
    or None for CraftGround's one shared folder (--shared-folder, the old behaviour)."""
    return None if shared_folder else env_path_for(port, verbose=True)


def launch_vec_env(factories: list[Callable[[], gym.Env]], ports: list[int],
                   shared_folder: bool = False) -> VecEnv:
    """Clear what a killed run may have left on these ports, then start one env per factory:
    each in its own worker process (SubprocVecEnv), or in this process when there is only one.
    Minecraft itself starts on the first reset."""
    if not shared_folder:
        kill_instance_processes(ports, verbose=True)  # leftovers of a killed run would block these ports
    clear_stale_loom_locks()  # also needed with a shared folder; see instances.py
    return SubprocVecEnv(factories) if len(factories) > 1 else DummyVecEnv(factories)
