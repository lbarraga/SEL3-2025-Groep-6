import typing

import jax.numpy as jnp
import numpy as np

import wandb


class WandbEvosaxLogger:

    def __init__(self, project_name: str, config: dict = None, **wandb_init_kwargs):
        self.run = wandb.init(project=project_name, config=config, **wandb_init_kwargs)

    @staticmethod
    def log_generation(
        generation: int,
        gen_time: float,
        es_state: typing.Any,
        fitness: jnp.ndarray,
        final_states: typing.Any
    ):
        wandb.log({
            "Generation": generation,
            "Time/Generation_Seconds": gen_time, # Note: Timing might be less accurate now
            "Overall_Best_Fitness": float(es_state.best_fitness),
            "Generation/Mean_Fitness": float(jnp.mean(fitness)),
            "Generation/Median_Fitness": float(jnp.median(fitness)),
            "Generation/Std_Fitness": float(jnp.std(fitness)),
            "Generation/Max_Fitness": float(jnp.max(fitness)),
            "Generation/Min_Fitness": float(jnp.min(fitness)),
            "Task/Mean_Final_Distance_m": float(jnp.mean(final_states.current_distance)),
            "Task/Min_Final_Distance_m": float(jnp.min(final_states.current_distance)),
            "Task/Mean_Steps": float(jnp.mean(final_states.steps_taken)),
            "Task/Termination_Rate": float(jnp.mean(final_states.terminated.astype(jnp.float32))),
            "Task/Truncation_Rate": float(jnp.mean(final_states.truncated.astype(jnp.float32))),
        })

    @staticmethod
    def log_summary(es_state: typing.Any):
        wandb.summary.update({
            "final_overall_best_fitness": float(es_state.best_fitness),
            "final_mean_params": np.array(es_state.mean),
            "final_best_params": np.array(es_state.best_member)
        })

    @staticmethod
    def finish():
        wandb.finish()

