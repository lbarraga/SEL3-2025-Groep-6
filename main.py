from stable_baselines3 import PPO

from brittlestar_gym_environment import BrittleStarEnv
import jax.numpy as jnp
from render import *

model = PPO.load("trained_agent.zip")

env = BrittleStarEnv()
observation, info = env.reset()


# action, _states = model.predict(observation, deterministic=True)
env.modulate_cpg(jnp.array([-2.163345, -1.929613, 1.388149, -1.741568, 0.945254, -1.186500, 0.295644, -0.849766, -0.689628, -0.579965, -0.503659, -1.648075, 1.043123, -1.978117, 0.188540, -0.380110, -0.944715, 0.171568, -1.004629, -1.398436, 1.207692]))

frames = []
while not env.is_terminated() and not env.is_truncated():
    env.single_step()

    frame = env.render()
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")