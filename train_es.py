import time
import jax
import jax.numpy as jnp

from evosax import OpenES

from brittle_star_environment import create_evaluation_fn
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM
from wandb_evosax_logger import WandbEvosaxLogger


POPULATION_SIZE = 100
NUM_GENERATIONS = 100
SIGMA_INIT = 0.1
SEED = 55
WANDB_PROJECT_NAME = "evosax_brittle_star"  # Define project name
num_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1

hyperparameters = {
    "population_size": POPULATION_SIZE,
    "num_generations": NUM_GENERATIONS,
    "sigma_init": SIGMA_INIT,
    "seed": SEED,
    "num_params": num_params,
}

logger = WandbEvosaxLogger(project_name=WANDB_PROJECT_NAME, config=hyperparameters)

evaluate_batch_fn = create_evaluation_fn()
master_key = jax.random.PRNGKey(SEED)
rng = master_key

strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_params, sigma_init=SIGMA_INIT, maximize=True)
es_params = strategy.default_params

initial_mean = jnp.zeros(num_params, dtype=jnp.float32)
initial_mean = initial_mean.at[-1].set(4.0)

es_state = strategy.initialize(rng, es_params, init_mean=initial_mean)

start_time = time.time()
for generation in range(NUM_GENERATIONS):
    loop_start_time = time.time()
    rng, rng_ask, rng_eval = jax.random.split(rng, 3)

    x, es_state = strategy.ask(rng_ask, es_state, es_params)
    rng_eval_batch = jax.random.split(rng_eval, POPULATION_SIZE)

    fitness, final_states = evaluate_batch_fn(rng_eval_batch, x)
    es_state = strategy.tell(x, fitness, es_state, es_params)
    gen_time = time.time() - loop_start_time

    logger.log_generation(
        generation=generation + 1,
        gen_time=gen_time,
        es_state=es_state,
        fitness=fitness,
        final_states=final_states
    )

logger.log_summary(es_state)
logger.finish()
