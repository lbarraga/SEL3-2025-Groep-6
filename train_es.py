import time

import jax
import jax.numpy as jnp
from evosax import OpenES
from jax.flatten_util import ravel_pytree

from brittle_star_environment import create_evaluation_fn
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, SEED, TARGET_SAMPLING_RADIUS, FIXED_OMEGA
from nn import CPGController
from util import generate_cpg_for_eval
from wandb_evosax_logger import WandbEvosaxLogger

POPULATION_SIZE = 100
NUM_GENERATIONS = 500
SIGMA_INIT = 0.1
WANDB_PROJECT_NAME = "evosax_brittle_star_nn"

master_key = jax.random.PRNGKey(SEED)
rng, model_init_rng = jax.random.split(master_key)

num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
model = CPGController(num_outputs=num_cpg_params_to_generate)

# Initialize the model parameters
dummy_input = jnp.zeros((1, 2))
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
        "target_sampling_region": f"Circle perimeter radius {TARGET_SAMPLING_RADIUS}",
        "nn_hidden_dim": model.hidden_dim,
    }
)

strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_model_params, sigma_init=SIGMA_INIT, maximize=True)
es_params = strategy.default_params
es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

evaluate_batch_fn = create_evaluation_fn()

generate_batch_cpg_for_eval = jax.vmap(generate_cpg_for_eval, in_axes=(0, 0, None, None))

for generation in range(NUM_GENERATIONS):
    loop_start_time = time.time()
    rng, rng_ask, rng_gen, rng_eval = jax.random.split(rng, 4)

    flat_model_params, es_state = strategy.ask(rng_ask, es_state, es_params)

    rngs = jax.random.split(rng_gen, POPULATION_SIZE)
    cpg_params_pop, target_pos_pop = generate_batch_cpg_for_eval(rngs, flat_model_params, model, unravel_fn)

    rng_eval_batch = jax.random.split(rng_eval, POPULATION_SIZE)
    fitness, final_states = evaluate_batch_fn(rng_eval_batch, cpg_params_pop, target_pos_pop)

    es_state = strategy.tell(flat_model_params, fitness, es_state, es_params)
    gen_time = time.time() - loop_start_time

    logger.log_generation(
        generation=generation + 1,
        gen_time=gen_time,
        es_state=es_state,
        fitness=fitness,
        final_states=final_states
    )

logger.log_model_artifact(
    parameters=unravel_fn(es_state.mean),
    filename=f"final_model_gen{NUM_GENERATIONS}.msgpack",
    artifact_name=f'{WANDB_PROJECT_NAME}-model',
    artifact_type='model'
)

logger.log_summary(es_state)
logger.finish()
