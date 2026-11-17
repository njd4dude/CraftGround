"""The tree-chopping environments.

    common   what both worlds share: image size, episode length, axe, actions
    arena    walled superflat arena with 8 placed oaks (rebuilt every episode)
    natural  a normal generated world, a new forest spot every episode

ENVS maps the --env names used on the command line to their module; each has make_env().
"""

from chop.envs import arena, natural

ENVS = {"arena": arena, "natural": natural}
