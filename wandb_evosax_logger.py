import typing

import flax
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
        fitness: jnp.ndarray, # Shape: (POPULATION_SIZE,) - already the mean fitness per individual
        final_states: typing.Any # Shape: (POPULATION_SIZE, k) PyTree
    ):
        # === Fitness Stats (based on the mean fitness returned by evaluation fn) ===
        mean_fitness = float(jnp.mean(fitness))
        median_fitness = float(jnp.median(fitness))
        std_fitness = float(jnp.std(fitness))
        max_fitness = float(jnp.max(fitness))
        min_fitness = float(jnp.min(fitness))

        # === Statistics Derived from the k Trials Per Individual ===

        # --- Current Distance ---
        # Shape: (POPULATION_SIZE, k)
        distances = final_states.current_distance

        # Calculate stats per individual across their k trials (axis=1)
        mean_dist_per_ind = jnp.mean(distances, axis=1) # Shape: (POPULATION_SIZE,)
        min_dist_per_ind = jnp.min(distances, axis=1)   # Shape: (POPULATION_SIZE,)
        max_dist_per_ind = jnp.max(distances, axis=1)   # Shape: (POPULATION_SIZE,)
        std_dist_per_ind = jnp.std(distances, axis=1)   # Shape: (POPULATION_SIZE,)

        # Calculate the mean of these per-individual stats across the population (axis=0)
        mean_mean_dist = float(jnp.mean(mean_dist_per_ind))
        mean_min_dist = float(jnp.mean(min_dist_per_ind))
        mean_max_dist = float(jnp.mean(max_dist_per_ind))
        mean_std_dist = float(jnp.mean(std_dist_per_ind))

        # Overall min distance across all trials (as before, but explicit)
        overall_min_dist = float(jnp.min(distances)) # Operates on flattened (POP_SIZE * k)

        # --- Steps Taken ---
        # Shape: (POPULATION_SIZE, k)
        steps = final_states.steps_taken
        mean_steps_per_ind = jnp.mean(steps, axis=1) # Shape: (POPULATION_SIZE,)
        mean_mean_steps = float(jnp.mean(mean_steps_per_ind))

        # --- Termination/Truncation Rates ---
        # Calculate mean rate per individual across k trials (axis=1)
        term_rate_per_ind = jnp.mean(final_states.terminated.astype(jnp.float32), axis=1)
        trunc_rate_per_ind = jnp.mean(final_states.truncated.astype(jnp.float32), axis=1)

        # Calculate mean rate across the population (axis=0)
        mean_term_rate = float(jnp.mean(term_rate_per_ind))
        mean_trunc_rate = float(jnp.mean(trunc_rate_per_ind))


        wandb.log({
            # Overall ES State
            "Generation": generation,
            "Time/Generation_Seconds": gen_time,
            "Overall_Best_Fitness": float(es_state.best_fitness), # Fitness of the best member's mean performance

            # Population Fitness Stats (Mean performance per individual)
            "Generation/Mean_Fitness": mean_fitness,
            "Generation/Median_Fitness": median_fitness,
            "Generation/Std_Fitness": std_fitness,
            "Generation/Max_Fitness": max_fitness, # Max of the mean performances
            "Generation/Min_Fitness": min_fitness, # Min of the mean performances

            # Aggregated Task Stats from k-Trials
            "Task_Agg/Mean_Mean_Final_Distance_m": mean_mean_dist,
            "Task_Agg/Mean_Min_Final_Distance_m": mean_min_dist,
            "Task_Agg/Mean_Max_Final_Distance_m": mean_max_dist,
            "Task_Agg/Mean_Std_Final_Distance_m": mean_std_dist,
            "Task_Agg/Overall_Min_Final_Distance_m": overall_min_dist, # Absolute best distance achieved in any trial
            "Task_Agg/Mean_Mean_Steps": mean_mean_steps,
            "Task_Agg/Mean_Termination_Rate": mean_term_rate,
            "Task_Agg/Mean_Truncation_Rate": mean_trunc_rate,

        })

    @staticmethod
    def log_summary(es_state: typing.Any):
        wandb.summary.update({
            "final_overall_best_fitness": float(es_state.best_fitness),
            "final_mean_params": np.array(es_state.mean),
            "final_best_params": np.array(es_state.best_member)
        })

    @staticmethod
    def log_model_artifact(
        parameters: typing.Any,
        filename: str,
        artifact_name: str,
        artifact_type: str = 'model',
    ):
        param_bytes = flax.serialization.to_bytes(parameters)
        with open(filename, "wb") as f:
            f.write(param_bytes)

        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
        )
        artifact.add_file(filename)
        wandb.log_artifact(artifact)

    @staticmethod
    def finish():
        wandb.finish()

