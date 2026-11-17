"""CraftGround startup smoke test.

Launches the Minecraft environment, resets it, takes a few random steps,
and shuts it down. Requires a JDK (21 for Minecraft 1.21) on PATH.
"""

import craftground
from craftground import InitialEnvironmentConfig
from craftground.initial_environment_config import WorldType

NUM_STEPS = 20

config = InitialEnvironmentConfig(
    image_width=114,
    image_height=64,
    world_type=WorldType.SUPERFLAT,
)
env = craftground.make(initial_env_config=config, port=8023)

try:
    obs, info = env.reset()
    keys = list(obs.keys()) if hasattr(obs, "keys") else type(obs).__name__
    print("reset OK, observation:", keys)

    for _ in range(NUM_STEPS):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            obs, info = env.reset()
    print(f"{NUM_STEPS} steps OK")
finally:
    env.close()
