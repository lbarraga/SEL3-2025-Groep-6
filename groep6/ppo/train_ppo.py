import wandb
from wandb.integration.sb3 import WandbCallback

from groep6.ppo.config import (
    WANDB_PROJECT_NAME, N_ENVS, LEARNING_RATE, N_STEPS, BATCH_SIZE, N_EPOCHS, GAMMA, \
    GAE_LAMBDA, CLIP_RANGE, DEVICE, POLICY_KWARGS, make_model
)

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

# Train the agent
model = make_model(logger=logger)
trained_model = model.learn(total_timesteps=1_250_000, progress_bar=True,
                            callback=WandbCallback(gradient_save_freq=100, verbose=2))
trained_model.save("trained_model")

logger.finish()
