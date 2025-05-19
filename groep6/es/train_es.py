import time

import jax
import jax.numpy as jnp
from evosax import OpenES
from jax.flatten_util import ravel_pytree
from tqdm import tqdm

from brittle_star_environment import create_evaluation_fn
from groep6.defaults import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, SEED, FIXED_OMEGA, TARGET_SAMPLING_RADIUS, \
    NUM_SEGMENTS_PER_ARM, NUM_EVALUATIONS_PER_INDIVIDUAL
from groep6.es.defaults import NUM_GENERATIONS, POPULATION_SIZE, SIGMA_INIT, WANDB_PROJECT_NAME, MODEL_FILE
from groep6.nn import CPGController
from groep6.wandb_evosax_logger import WandbEvosaxLogger

def train_es(
        num_generations = NUM_GENERATIONS, population_size: int = POPULATION_SIZE,
        seed: int = SEED, sigma_init: float = SIGMA_INIT, target_sampling_radius: float = TARGET_SAMPLING_RADIUS,
        num_evaluations: int = NUM_EVALUATIONS_PER_INDIVIDUAL, fixed_omega: float = FIXED_OMEGA,
        wandb_project_name: str = WANDB_PROJECT_NAME, model_file: str = MODEL_FILE
):
    """
    Trains a CPGController model using an evolutionary strategy (OpenES).

    This function initializes a CPGController, sets up an OpenES strategy,
    and then runs a training loop for a specified number of generations.
    In each generation, it asks the ES for a population of model parameters,
    evaluates these parameters using a simulated environment, and then tells
    the ES the fitness scores. Results are logged to Weights & Biases.

    Args:
        num_generations: The number of generations to run the evolutionary strategy.
        population_size: The number of individuals in the population for each generation.
        seed: The random seed for reproducibility.
        sigma_init: The initial standard deviation for the OpenES strategy.
        target_sampling_radius: The radius within which targets are sampled in the environment.
        num_evaluations: The number of times each individual is evaluated. The final fitness
                         is the average of these evaluations.
        fixed_omega: The fixed angular frequency for the CPG oscillators.
        wandb_project_name: The name of the Weights & Biases project for logging.
        model_file: The name of the file to save the trained model parameters.
    """
    master_key = jax.random.PRNGKey(seed)
    rng, model_init_rng = jax.random.split(master_key)

    num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
    model = CPGController(num_outputs=num_cpg_params_to_generate)

    num_joint_positions = NUM_OSCILLATORS_PER_ARM * NUM_ARMS * NUM_SEGMENTS_PER_ARM
    dummy_input = jnp.zeros((1, 1 + num_joint_positions))  # +1 for relative direction to target in radials
    initial_model_params_tree = model.init(model_init_rng, dummy_input)['params']
    flat_initial_model_params, unravel_fn = ravel_pytree(initial_model_params_tree)
    num_model_params = flat_initial_model_params.shape[0]

    logger = WandbEvosaxLogger(
        project_name=wandb_project_name,
        config={
            "population_size": population_size,
            "num_generations": num_generations,
            "sigma_init": sigma_init,
            "seed": seed,
            "num_model_params": num_model_params,
            "fixed_omega": fixed_omega,
            "target_sampling_radius": target_sampling_radius,
            "nn_hidden_dim": f"{model.hidden_dim1} x {model.hidden_dim2}",
            "num_evaluations_per_individual": num_evaluations,
        }
    )

    strategy = OpenES(popsize=population_size, num_dims=num_model_params, sigma_init=sigma_init, maximize=True)
    es_params = strategy.default_params
    es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

    evaluate_batch_fn = create_evaluation_fn(model_obj=model, unravel_fn=unravel_fn, num_evaluations=num_evaluations)

    for generation in tqdm(range(num_generations)):

        loop_start_time = time.time()
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)

        # Ask for new network parameters
        flat_model_params_pop, es_state = strategy.ask(rng_ask, es_state, es_params)

        rng_eval_batch = jax.random.split(rng_eval, population_size)
        fitness, final_states_batch = evaluate_batch_fn(rng_eval_batch, flat_model_params_pop)  # Pass network params

        # Tell the strategy the mean fitness results
        es_state = strategy.tell(flat_model_params_pop, fitness, es_state, es_params)  # Pass network params
        gen_time = time.time() - loop_start_time

        # Log results (final_states_batch now contains states from k evaluations per individual)
        logger.log_generation(
            generation=generation + 1,
            gen_time=gen_time,
            es_state=es_state,
            fitness=fitness,
            final_states=final_states_batch
        )

    logger.log_model_artifact(
        parameters=unravel_fn(es_state.mean),
        filename=model_file,
        artifact_name=f'{wandb_project_name}-model',
        artifact_type='model'
    )

    logger.log_summary(es_state)
    logger.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train the model using evolutionary strategy")
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS,
                        help="Number of generations to train for")
    parser.add_argument("--population_size", type=int, default=POPULATION_SIZE,
                        help="Population size for the evolutionary strategy")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--sigma_init", type=float, default=SIGMA_INIT,
                        help="Initial standard deviation for the evolutionary strategy")
    parser.add_argument("--fixed_omega", type=float, default=FIXED_OMEGA,
                        help="Fixed omega value for the CPG model")
    parser.add_argument("--target_sampling_radius", type=float, default=TARGET_SAMPLING_RADIUS,
                        help="Target sampling radius for the CPG model")
    parser.add_argument("--num_evaluations", type=int, default=NUM_EVALUATIONS_PER_INDIVIDUAL,
                        help="Number of evaluations per individual in the evolutionary strategy. "
                             "Reward is the average reward of these evaluations.")
    parser.add_argument("--wandb_project_name", type=str, default=WANDB_PROJECT_NAME,
                        help="Wandb project name for logging")
    parser.add_argument("--model_file", type=str, default=MODEL_FILE,
                        help="The name of the file to save the trained model parameters")

    args = parser.parse_args()

    train_es(**vars(args))
