import jax

from ES_Model import ESModel
from evosax.algorithms import Open_ES as ES
from flax import nnx
from jax.flatten_util import ravel_pytree
import optax
from jax import numpy as jnp

inParams: int = 30
outParams: int = 21
num_generations: int = 512
population_size: int = 256

model = ESModel(inParams, outParams)


# lr_schedule = optax.exponential_decay(
#     init_value=0.01,
#     transition_steps=num_generations,
#     decay_rate=0.1,
# )
# std_schedule = optax.exponential_decay(
#     init_value=0.05,
#     transition_steps=num_generations,
#     decay_rate=0.2,
# )
params, state = nnx.split(model)

es = ES(
    population_size=population_size,
    solution=state
    # optimizer=optax.adam(learning_rate=lr_schedule),
    # std_schedule=std_schedule,
)

def step(carry, key):
    state, params, problem_state = carry
    key_ask, key_eval, key_tell = jax.random.split(key, 3)

    population, state = es.ask(key_ask, state, params)

    fitness, problem_state, _ = (1,1,1) # TODO: replace with actual fitness function jax.vmap(...)

    state, metrics = es.tell(
        key_tell, population, -fitness, state, params
    )  # Minimize fitness

    return (state, params, problem_state), metrics


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
state = es.init(subkey, state, params)

fitness_log = []
log_period = 32 # how often to print to stdout
for i in range(num_generations // log_period):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, log_period)
    (state, params, problem_state), metrics = jax.lax.scan(
        step,
        (state, params, problem_state),
        keys, # run step() len(keys) times, passing 1 key each time
    )

    mean = es.get_mean(state)

    key, subkey = jax.random.split(key)
    fitness, problem_state, info = (1,1,1) # TODO: replace with single run of fitness function (not to train)
    print(f"Generation {(i + 1) * log_period:3d} | Mean fitness: {fitness.mean():.2f}")
# params = es.default_params