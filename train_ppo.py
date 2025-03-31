from stable_baselines3 import PPO

from brittlestar_gym_environment import BrittleStarEnv
from render import show_video

# Create the environment
env = BrittleStarEnv()

# Create the agent with custom hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,           # Learning rate
    n_steps=1024,                 # Steps to run for each environment per update
    batch_size=64,                # Minibatch size
    n_epochs=10,                  # Number of epochs when optimizing the surrogate loss
    gamma=0.99,                   # Discount factor
    gae_lambda=0.95,              # Factor for trade-off of bias vs variance for GAE
    clip_range=0.2,               # Clipping parameter for PPO
    verbose=1,
    device="cpu",                           # Use CPU explicitly
    policy_kwargs=dict(net_arch=[16, 16])   # Custom policy architecture
)

# Train the agent
model.learn(total_timesteps=10000, progress_bar=True)


# Test the agent
observation, info = env.reset()
frames = []

terminated = False
truncated = False
step = 0
while step < 500 and not terminated and not truncated:
    action, _states = model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())
    step += 1

# Save video of trained agent
show_video(images=frames, sim_time=100, path="trained_agent.mp4")