import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from brittle_star_gym_environment import BrittleStarGymEnv
from groep6.config import MAX_STEPS_PER_PPO_EPISODE
from groep6.render import show_video
from wandb.integration.sb3 import WandbCallback

WANDB_PROJECT_NAME = "ppo_brittle_star_gym_bounds"

# Create the environment
env = BrittleStarGymEnv()

N_ENVS = 4
LEARNING_RATE = 3e-4
N_STEPS = 8
BATCH_SIZE = 64
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
DEVICE = "cpu"
POLICY_KWARGS = dict(net_arch=[16, 16])

def make_env():
    gym_env = BrittleStarGymEnv()
    return Monitor(gym_env)

vec_env = make_vec_env(make_env, n_envs=N_ENVS)

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

# Create the agent with custom hyperparameters
model = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=LEARNING_RATE,           # Learning rate
    n_steps=N_STEPS,                 # Steps to run for each environment per update
    batch_size=BATCH_SIZE,                # Minibatch size
    n_epochs=N_EPOCHS,                  # Number of epochs when optimizing the surrogate loss
    gamma=GAMMA,                   # Discount factor
    gae_lambda=GAE_LAMBDA,              # Factor for trade-off of bias vs variance for GAE
    clip_range=CLIP_RANGE,               # Clipping parameter for PPO
    verbose=0,
    device=DEVICE,                           # Use CPU explicitly
    policy_kwargs=POLICY_KWARGS,   # Custom policy architecture
    tensorboard_log=f"runs/{logger.id}",
)

# Train the agent
trained_model = model.learn(total_timesteps=1_250_000, progress_bar=True, callback=WandbCallback(gradient_save_freq=100, verbose=2))
trained_model.save("trained_model")

logger.finish()

# trained_model = model.load("trained_model")
# trained_model.device = "cpu"  # Explicitly set device to CPU

# Test the agent
observation, info = env.reset()
frames = []

terminated = False
truncated = False
step = 0

action, _states = trained_model.predict(observation, deterministic=True)
env.sim_state = env.sim_state.replace(cpg_state=env.modulate_cpg(env.sim_state.cpg_state, action))

while step < 500 and not terminated and not truncated:
    if step % MAX_STEPS_PER_PPO_EPISODE == 0:
        observation = env.get_observation()
        action, _states = trained_model.predict(observation, deterministic=True)
        env.sim_state = env.sim_state.replace(cpg_state=env.modulate_cpg(env.sim_state.cpg_state, action))

    env.sim_state = env.simulation_single_step_logic(env.sim_state)
    frames.append(env.render())
    step += 1

# Save video of trained agent
show_video(images=frames, sim_time=100, path="../trained_agent.mp4")