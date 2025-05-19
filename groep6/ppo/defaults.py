from stable_baselines3 import PPO

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

TOTAL_TIMESTEPS = 1_250_000
MODEL_FILE = "trained_model.zip"