from brittlestar_gym_environment import BrittleStarEnv
from cpg import *
from render import *

env = BrittleStarEnv()

frames = []
while not (env.is_terminated() | env.is_truncated()):

    # TODO: use model
    action = jnp.array([
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, # R
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # X
        5 # omega
    ])
    env.step(action)

    frame = env.render()
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")