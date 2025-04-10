from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

import wandb

from brittlestar_gym_environment import BrittleStarEnv
from wandb_ppo_callback import SimpleBrittleStarCallback


# Keep the helper function definition outside
def make_env(seed=0):
    def _init():
        env = BrittleStarEnv(seed=seed)
        return env

    return _init


if __name__ == '__main__':  # <-- ADD THIS GUARD

    # Number of parallel environments (e.g., number of CPU cores - 1)
    num_cpu = 9  # Adjust based on your system

    # Create the vectorized environment INSIDE the guard
    env = SubprocVecEnv([make_env(seed=i) for i in range(num_cpu)])

    wandb.init(project="brittle-star-ppo")

    # --- Hyperparameters ---
    p_n_steps = 512
    p_batch_size = 64  # PPO internal minibatch size
    p_n_epochs = 10
    p_total_timesteps = 100_000  # Increase significantly

    # Create the model INSIDE the guard
    model = PPO(
        "MlpPolicy",
        env,  # Pass the VecEnv
        n_steps=p_n_steps,
        batch_size=p_batch_size,
        n_epochs=p_n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        verbose=1,
        device="cpu",  # Or "cuda" if using GPU
        policy_kwargs=dict(net_arch=[16, 16])  # Or [64, 64]
    )

    # Setup and run training INSIDE the guard
    callback = SimpleBrittleStarCallback()
    model.learn(total_timesteps=p_total_timesteps, progress_bar=True, callback=callback)
    model.save("trained_agent")

    # Logging INSIDE the guard
    artifact = wandb.Artifact("brittle_star_model", type="model")
    artifact.add_file("trained_agent.zip")
    wandb.log_artifact(artifact)
    wandb.finish()  # Good practice to finish wandb run

    # Close the environment INSIDE the guard
    env.close()
