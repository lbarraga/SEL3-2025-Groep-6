import time

import jax
import jax.numpy as jnp
from evosax import OpenES
from jax.flatten_util import ravel_pytree

from brittle_star_environment import create_evaluation_fn
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM
from nn import CPGController
from wandb_evosax_logger import WandbEvosaxLogger

POPULATION_SIZE = 100
NUM_GENERATIONS = 100
SIGMA_INIT = 0.1
SEED = 55
WANDB_PROJECT_NAME = "evosax_brittle_star_nn"
FIXED_OMEGA = 4.5
TARGET_SAMPLING_RADIUS = 1.8

master_key = jax.random.PRNGKey(SEED)
rng, model_init_rng = jax.random.split(master_key)

num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
model = CPGController(num_outputs=num_cpg_params_to_generate)

dummy_input = jnp.zeros((1, 2))
initial_model_params_tree = model.init(model_init_rng, dummy_input)['params']

flat_initial_model_params, unravel_fn = ravel_pytree(initial_model_params_tree)
num_model_params = flat_initial_model_params.shape[0]

hyperparameters = {
    "population_size": POPULATION_SIZE,
    "num_generations": NUM_GENERATIONS,
    "sigma_init": SIGMA_INIT,
    "seed": SEED,
    "num_model_params": num_model_params,
    "fixed_omega": FIXED_OMEGA,
    "target_sampling_region": f"Circle perimeter radius {TARGET_SAMPLING_RADIUS}",
    "nn_hidden_dim": model.hidden_dim,
}

logger = WandbEvosaxLogger(project_name=WANDB_PROJECT_NAME, config=hyperparameters)

strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_model_params, sigma_init=SIGMA_INIT, maximize=True)
es_params = strategy.default_params
es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

evaluate_batch_fn = create_evaluation_fn()

def generate_cpg_for_eval(rng_single, flat_model_params_single, model_obj, unravel_fn_single, fixed_omega_val):
    angle = jax.random.uniform(rng_single, minval=0, maxval=2 * jnp.pi)
    radius = TARGET_SAMPLING_RADIUS
    target_pos = jnp.array([radius * jnp.cos(angle), radius * jnp.sin(angle), 0.0])

    target_pos_2d = target_pos[:2]
    norm = TARGET_SAMPLING_RADIUS
    normalized_direction = target_pos_2d / norm

    model_params_single = unravel_fn_single(flat_model_params_single)

    cpg_params = model_obj.infer(model_params_single, normalized_direction, fixed_omega_val)

    return cpg_params, target_pos

generate_batch_cpg_for_eval = jax.vmap(generate_cpg_for_eval, in_axes=(0, 0, None, None, None))

start_time = time.time()

print("Starting ES training loop...")
for generation in range(NUM_GENERATIONS):
    loop_start_time = time.time()
    rng, rng_ask, rng_gen, rng_eval = jax.random.split(rng, 4)

    flat_model_params_pop, es_state = strategy.ask(rng_ask, es_state, es_params)

    rng_gen_batch = jax.random.split(rng_gen, POPULATION_SIZE)
    cpg_params_pop, target_pos_pop = generate_batch_cpg_for_eval(
        rng_gen_batch, flat_model_params_pop, model, unravel_fn, FIXED_OMEGA
    )

    rng_eval_batch = jax.random.split(rng_eval, POPULATION_SIZE)
    fitness, final_states = evaluate_batch_fn(rng_eval_batch, cpg_params_pop, target_pos_pop)

    es_state = strategy.tell(flat_model_params_pop, fitness, es_state, es_params)
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
    artifact_type='model',
    metadata=hyperparameters
)

logger.log_summary(es_state)
logger.finish()
