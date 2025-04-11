from stable_baselines3 import PPO

from brittlestar_gym_environment import BrittleStarEnv
import jax.numpy as jnp
from render import *

model = PPO.load("trained_agent.zip")

env = BrittleStarEnv()
observation, info = env.reset()


# action, _states = model.predict(observation, deterministic=True)
env.modulate_cpg(jnp.array([-0.778954, -0.277990, 0.094271, -0.578112, -1.130316, 0.448246, 1.224521, -0.109453, 0.572109, 0.263016, -0.744902, -0.151820, 0.283899, -0.713379, -0.041819, -0.198762, -0.349436, 0.017592, 0.217051, 0.450029, 5.064455]))

frames = []
while not env.is_terminated() and not env.is_truncated():
    env.single_step()

    frame = env.render()
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")