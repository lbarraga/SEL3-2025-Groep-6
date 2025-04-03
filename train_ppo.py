import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from brittlestar_gym_environment import BrittleStarEnv
from render import show_video


class ImprovedWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ImprovedWandbCallback, self).__init__(verbose)
        # Initialize episode tracking variables
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = []
        self.total_episodes = 0
        self.dist_to_target = []

    def _on_step(self) -> bool:
        # Get info from the current step
        info = self.locals.get("infos", [{}])[0]
        reward = self.locals.get("rewards", [0])[0]
        done = self.locals.get("dones", [False])[0]

        # Track rewards for current episode
        self.current_episode_rewards.append(reward)

        # Track distance to target
        if "distance_to_target" in info:
            self.dist_to_target.append(info["distance_to_target"])

        # When episode ends, log episodic metrics
        if done:
            self.total_episodes += 1
            self.episode_rewards.append(sum(self.current_episode_rewards))
            self.episode_lengths.append(len(self.current_episode_rewards))

            # Log episode metrics to WandB
            wandb.log({
                "episode": self.total_episodes,
                "episode_reward": self.episode_rewards[-1],
                "episode_length": self.episode_lengths[-1],
                "average_episode_reward": np.mean(self.episode_rewards[-100:]),
                "final_distance_to_target": self.dist_to_target[-1] if self.dist_to_target else None,
                "min_distance_to_target": min(self.dist_to_target) if self.dist_to_target else None,
            })

            # Reset episode tracking
            self.current_episode_rewards = []
            self.dist_to_target = []

        # Still log some step-based metrics
        wandb.log({
            "learning_rate": self.model.learning_rate,
            "steps": self.num_timesteps,
        })

        return True

# Initialize wandb with your project name
wandb.init(project="brittle_star_rl")

# Initialize improved WandB callback
wandb_callback = ImprovedWandbCallback()

# Create the environment
env = BrittleStarEnv()

# Create the agent with custom hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-4,           # Learning rate
    n_steps=5,                 # Steps to run for each environment per update
    batch_size=10,                # Minibatch size
    n_epochs=7,                  # Number of epochs when optimizing the surrogate loss
    gamma=0.99,                   # Discount factor
    gae_lambda=0.95,              # Factor for trade-off of bias vs variance for GAE
    clip_range=0.2,               # Clipping parameter for PPO
    verbose=1,
    device="cpu",                           # Use CPU explicitly
    policy_kwargs=dict(net_arch=[16, 16])   # Custom policy architecture
)

# Train the agent
model.learn(total_timesteps=2_000, progress_bar=True, callback=wandb_callback)
model.save("trained_agent")

# Test the agent
observation, info = env.reset()
action, _states = model.predict(observation, deterministic=True)
env.modulate_cpg(action)
frames = []

terminated = False
truncated = False
step = 0
while step < 500 and not terminated and not truncated:
    observation = env.single_step()
    frames.append(env.render())
    step += 1

# Save video of trained agent
show_video(images=frames, sim_time=100, path="trained_agent.mp4")

wandb.finish()