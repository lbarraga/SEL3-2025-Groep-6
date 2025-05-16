import time

import jax
import jax.numpy as jnp
from evosax import OpenES
from jax.flatten_util import ravel_pytree

from brittle_star_environment import create_evaluation_fn, NUM_EVALUATIONS_PER_INDIVIDUAL
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, SEED, FIXED_OMEGA, TARGET_SAMPLING_RADIUS, NUM_SEGMENTS_PER_ARM
from nn import CPGController
from wandb_evosax_logger import WandbEvosaxLogger

POPULATION_SIZE = 500
NUM_GENERATIONS = 300
SIGMA_INIT = 0.1
WANDB_PROJECT_NAME = "k_eval_multiple_inference_joint_positions"  # Updated project name

master_key = jax.random.PRNGKey(SEED)
rng, model_init_rng = jax.random.split(master_key)

num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
model = CPGController(num_outputs=num_cpg_params_to_generate)

num_joint_positions = NUM_OSCILLATORS_PER_ARM * NUM_ARMS * NUM_SEGMENTS_PER_ARM
dummy_input = jnp.zeros((1, 1 + num_joint_positions))  # +1 for relative direction to target in radials
initial_model_params_tree = model.init(model_init_rng, dummy_input)['params']
flat_initial_model_params, unravel_fn = ravel_pytree(initial_model_params_tree)
num_model_params = flat_initial_model_params.shape[0]

logger = WandbEvosaxLogger(
    project_name=WANDB_PROJECT_NAME,
    config={
        "population_size": POPULATION_SIZE,
        "num_generations": NUM_GENERATIONS,
        "sigma_init": SIGMA_INIT,
        "seed": SEED,
        "num_model_params": num_model_params,
        "fixed_omega": FIXED_OMEGA,
        "target_sampling_radius": TARGET_SAMPLING_RADIUS,
        "nn_hidden_dim": f"{model.hidden_dim1} x {model.hidden_dim2}",
        "num_evaluations_per_individual": NUM_EVALUATIONS_PER_INDIVIDUAL,
    }
)

strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_model_params, sigma_init=SIGMA_INIT, maximize=True)
es_params = strategy.default_params
es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

evaluate_batch_fn = create_evaluation_fn(model_obj=model, unravel_fn=unravel_fn)

try:
    # generation = 0
    # while True:
    for generation in range(NUM_GENERATIONS):
        if generation % 100 == 0:
            logger.log_model_artifact(
                parameters=unravel_fn(es_state.mean),
                filename=f"final_model_gen{NUM_GENERATIONS}.msgpack",
                artifact_name=f'{WANDB_PROJECT_NAME}-model',
                artifact_type='model'
            )

        loop_start_time = time.time()
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)

        # Ask for new network parameters
        flat_model_params_pop, es_state = strategy.ask(rng_ask, es_state, es_params)  # Changed variable name for clarity

        rng_eval_batch = jax.random.split(rng_eval, POPULATION_SIZE)
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
        # generation += 1

except KeyboardInterrupt:
    print("Training interrupted. Saving model...")

logger.log_model_artifact(
    parameters=unravel_fn(es_state.mean),
    filename=f"final_model_gen{NUM_GENERATIONS}.msgpack",
    artifact_name=f'{WANDB_PROJECT_NAME}-model',
    artifact_type='model'
)

logger.log_summary(es_state)
logger.finish()
