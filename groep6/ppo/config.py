from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from groep6.ppo.brittle_star_gym_environment import BrittleStarGymEnv

WANDB_PROJECT_NAME = "ppo_brittle_star_gym_bounds"

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

def make_model(logger_id = None):
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,  # Learning rate
        n_steps=N_STEPS,  # Steps to run for each environment per update
        batch_size=BATCH_SIZE,  # Minibatch size
        n_epochs=N_EPOCHS,  # Number of epochs when optimizing the surrogate loss
        gamma=GAMMA,  # Discount factor
        gae_lambda=GAE_LAMBDA,  # Factor for trade-off of bias vs variance for GAE
        clip_range=CLIP_RANGE,  # Clipping parameter for PPO
        verbose=0,
        device=DEVICE,  # Use CPU explicitly
        policy_kwargs=POLICY_KWARGS,  # Custom policy architecture
        tensorboard_log=f"runs/{logger_id}",
    )
    return model
