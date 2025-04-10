from stable_baselines3 import PPO

from brittlestar_gym_environment import BrittleStarEnv
from wandb_ppo_callback import SimpleBrittleStarCallback

env = BrittleStarEnv(seed=55)

# Create the agent with improved hyperparameters for one-shot parameter selection
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1e-3,  # Higher learning rate for one-shot decision
    n_steps=2,  # Match to number of environments
    n_epochs=10,  # More epochs to extract more from limited data
    gamma=0.99,  # Standard discount factor
    gae_lambda=0.95,  # Standard GAE lambda
    clip_range=0.2,  # Standard clipping parameter
    ent_coef=0.05,  # Added entropy coefficient for better exploration
    verbose=1,
    device="cpu",  # Using CPU explicitly
    policy_kwargs=dict(net_arch=[16, 16])
)

model.learn(total_timesteps=1000, progress_bar=True, callback=SimpleBrittleStarCallback())
model.save("trained_agent")
