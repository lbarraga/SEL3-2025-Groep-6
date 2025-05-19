import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback

from groep6.ppo.brittle_star_gym_environment import BrittleStarGymEnv
from groep6.ppo.config import (
    WANDB_PROJECT_NAME, N_ENVS, LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS, GAMMA,
    GAE_LAMBDA, CLIP_RANGE, DEVICE, POLICY_KWARGS, TOTAL_TIMESTEPS, MODEL_FILE
)


def make_env():
    gym_env = BrittleStarGymEnv()
    return Monitor(gym_env)


def train_ppo(
        n_envs: int = N_ENVS, learning_rate: float = LEARNING_RATE, n_steps: int = N_STEPS, batch_size: int = BATCH_SIZE,
        n_epochs: int = N_EPOCHS, gamma: float = GAMMA, gae_lambda: float = GAE_LAMBDA, clip_range: float = CLIP_RANGE,
        device: str = DEVICE, policy_kwargs: dict = POLICY_KWARGS, timesteps: int = TOTAL_TIMESTEPS,
        model_file: str = MODEL_FILE, wandb_project_name: str = WANDB_PROJECT_NAME
):
    """
    Trains a Proximal Policy Optimization (PPO) agent.

    Args:
        n_envs (int): Number of parallel environments to use for training.
        learning_rate (float): The learning rate for the PPO optimizer.
        n_steps (int): The number of steps to run for each environment per update.
                       This is (n_envs * n_steps) total steps per update.
        batch_size (int): The minibatch size for optimizing the PPO objective.
        n_epochs (int): The number of epochs to run when optimizing the surrogate loss.
        gamma (float): The discount factor for future rewards.
        gae_lambda (float): Factor for trade-off of bias vs variance for Generalized
                            Advantage Estimation (GAE).
        clip_range (float): Clipping parameter for PPO. This limits the change in
                            the policy update.
        device (str): The device to use for training ('cpu', 'cuda', 'auto').
        policy_kwargs (dict): Additional arguments to pass to the policy network.
                              For example, to specify the network architecture.
        timesteps (int): The total number of timesteps to train the agent.
        model_file (str): The filename to save the trained model.
        wandb_project_name (str): The name of the Weights and Biases project for logging.

    Returns:
        stable_baselines3.PPO: The trained PPO model.
    """
    vec_env = make_vec_env(make_env, n_envs=n_envs)

    logger = wandb.init(
        project=wandb_project_name,
        config={
            "n_envs": n_envs,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "device": device,
            "policy_kwargs": policy_kwargs,
        },
        monitor_gym=True,
        sync_tensorboard=True,
    )

    # Train the agent
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=0,
        device=device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=f"runs/{logger.id}",
    )

    trained_model = model.learn(
        total_timesteps=timesteps, progress_bar=True, callback=WandbCallback(gradient_save_freq=100, verbose=2)
    )
    trained_model.save(model_file)

    logger.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the model using PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--n_envs", type=int, default=N_ENVS,
                        help="Number of environments to train with")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training the model")
    parser.add_argument("--n_steps", type=int, default=N_STEPS,
                        help="Steps to run for each environment per update")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS,
                        help="Number of epochs when optimizing the surrogate loss")
    parser.add_argument("--gamma", type=float, default=GAMMA,
                        help="Discount factor for surrogate loss")
    parser.add_argument("--gae_lambda", type=float, default=GAE_LAMBDA,
                        help="Factor for trade-off of bias vs variance for GAE")
    parser.add_argument("--clip_range", type=float, default=CLIP_RANGE,
                        help="Clip parameter for PPO")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Which device to run on: cpu, gpu, auto")
    parser.add_argument("--policy_kwargs", type=dict, default=POLICY_KWARGS,
                        help="Custom policy architecture")
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS,
                        help="Number of timesteps to train for")
    parser.add_argument("--model_file", type=str, default=MODEL_FILE,
                        help="Filename to save the trained model")
    parser.add_argument("--wandb_project_name", type=str, default=WANDB_PROJECT_NAME,
                        help="Weights and Biases project name for logging")

    args = parser.parse_args()

    train_ppo(**vars(args))
