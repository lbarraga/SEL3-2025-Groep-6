import wandb
from stable_baselines3 import PPO

from brittle_star_gym_environment import BrittleStarGymEnv
from jax_env import JAXVecEnv
from config import MAX_STEPS_PER_PPO_EPISODE  # Assuming you have this
from render import show_video  # Assuming you have this
from wandb.integration.sb3 import WandbCallback

WANDB_PROJECT_NAME = "ppo_brittle_star_jaxenv"

# --- Hyperparameters ---
N_ENVS = 2
LEARNING_RATE = 3e-4
N_STEPS = 8
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
DEVICE = "cpu"  # Or "cuda" if you have it set up
POLICY_KWARGS = dict(net_arch=[16, 16])
TOTAL_TIMESTEPS = 250_000

# --- Create the JAXVecEnv ---
print("Create the JaxVecEnv")
jax_vec_env = JAXVecEnv(BrittleStarGymEnv, num_envs=N_ENVS)

logger = wandb.init(
    project=WANDB_PROJECT_NAME,
    config={
        "n_envs": N_ENVS,
        "learning_rate": LEARNING_RATE,
        "n_steps": N_STEPS,
        "batch_size": BATCH_SIZE,
        "n_epochs": N_EPOCHS,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_range": CLIP_RANGE,
        "device": DEVICE,
        "policy_kwargs": POLICY_KWARGS,
    },
    monitor_gym=True,
    sync_tensorboard=True,
)

# --- Create the PPO agent ---
print("Create the PPO agent")
model = PPO(
    "MlpPolicy",
    jax_vec_env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    verbose=0,
    device=DEVICE,
    policy_kwargs=POLICY_KWARGS,
    tensorboard_log=f"runs/{logger.id}",
)

# --- Train the agent ---
print("Train the model")
trained_model = model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    progress_bar=True,
    callback=WandbCallback(gradient_save_freq=100, verbose=2),
)
trained_model.save("trained_model_jaxenv")
logger.finish()

# --- Test the agent ---
print("Test the agent")
env = BrittleStarGymEnv()
observation, info = env.reset()
frames = []
terminated = False
truncated = False
step = 0

action, _states = trained_model.predict(observation, deterministic=True)
# Assuming your BrittleStarGymEnv's step takes action directly for single env
next_observation, reward, terminated, truncated, info = env.step(action)
observation = next_observation

while step < 500 and not terminated and not truncated:
    action, _states = trained_model.predict(observation, deterministic=True)
    next_observation, reward, terminated, truncated, info = env.step(action)
    observation = next_observation
    frames.append(env.render())
    step += 1

# --- Save video ---
show_video(images=frames, sim_time=100, path="trained_agent_jaxenv.mp4")