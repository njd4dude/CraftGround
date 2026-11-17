"""Running Minecraft instances.

    instances    each port's own copy of the game folder, and killing/closing instances cleanly
    launch_lock  start instances one at a time (an alternative fix for startup collisions)
    vec          launch N instances as one VecEnv and close them again
"""
