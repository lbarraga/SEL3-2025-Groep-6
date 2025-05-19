import time

import jax
import jax.numpy as jnp
from evosax import OpenES
from jax.flatten_util import ravel_pytree

from brittle_star_environment import create_evaluation_fn, NUM_EVALUATIONS_PER_INDIVIDUAL
from groep6.config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, SEED, FIXED_OMEGA, TARGET_SAMPLING_RADIUS, \
    NUM_SEGMENTS_PER_ARM
from groep6.nn import CPGController
from groep6.wandb_evosax_logger import WandbEvosaxLogger

POPULATION_SIZE = 250
NUM_GENERATIONS = 250
SIGMA_INIT = 0.1
WANDB_PROJECT_NAME = "k_eval_multiple_inference_joint_positions"


def train_es(
        num_generations = NUM_GENERATIONS, population_size: int = POPULATION_SIZE,
        seed: int = SEED, sigma_init: float = SIGMA_INIT, target_sampling_radius: float = TARGET_SAMPLING_RADIUS,
        num_evaluations_per_individual: int = NUM_EVALUATIONS_PER_INDIVIDUAL, fixed_omega: float = FIXED_OMEGA,
        num_evaluations: int = NUM_EVALUATIONS_PER_INDIVIDUAL, wandb_project_name: str = WANDB_PROJECT_NAME
):
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
            "num_evaluations_per_individual": num_evaluations_per_individual,
        }
    )

    strategy = OpenES(popsize=population_size, num_dims=num_model_params, sigma_init=sigma_init, maximize=True)
    es_params = strategy.default_params
    es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

    evaluate_batch_fn = create_evaluation_fn(model_obj=model, unravel_fn=unravel_fn, num_evaluations=num_evaluations)

    for generation in range(num_generations):

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
        filename=f"final_model_gen{num_generations}.msgpack",
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
    parser.add_argument("--num_evaluations_per_individual", type=int, default=NUM_EVALUATIONS_PER_INDIVIDUAL,
                        help="Number of evaluations per individual in the evolutionary strategy")
    parser.add_argument("--wandb_project_name", type=str, default=WANDB_PROJECT_NAME,
                        help="Wandb project name for logging")

    args = parser.parse_args()

    train_es(
        num_generations=args.num_generations,
        population_size=args.population_size,
        seed=args.seed,
        sigma_init=args.sigma_init
    )
