from stable_baselines3 import PPO

from brittlestar_gym_environment import BrittleStarEnv
import jax.numpy as jnp
from render import *

model = PPO.load("trained_agent.zip")

env = BrittleStarEnv()
observation, info = env.reset()


# action, _states = model.predict(observation, deterministic=True)
env.modulate_cpg(jnp.array([
    0.118032, -0.803417, -0.557202, 0.814711, -0.067402, 0.499765, -0.111335, -0.701821, 0.972904, 0.59254,
    0.828742, 0.46721, 0.790916, -0.968669, 0.264438, -0.900014, -0.322913, 0.186337, -0.403329, -0.889394,
    4.196488
]))

frames = []
while not env.is_terminated() and not env.is_truncated():
    env.single_step()

    frame = env.render()
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")