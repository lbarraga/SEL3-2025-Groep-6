from stable_baselines3 import PPO

from brittlestar_gym_environment import BrittleStarEnv
from render import *

model = PPO.load("trained_agent_10k.zip")

env = BrittleStarEnv()
observation, info = env.reset()


action, _states = model.predict(observation, deterministic=True)
env.modulate_cpg(action)

frames = []
while not env.is_terminated() and not env.is_truncated():


    env.single_step()

    frame = env.render()
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")