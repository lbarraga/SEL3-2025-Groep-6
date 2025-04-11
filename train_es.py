import time

import jax
import jax.numpy as jnp
import numpy as np
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from evosax import OpenES
from tqdm import tqdm

from config import (
    SEED, morphology_spec, arena_config, env_config,
    DEFAULT_TARGET_POSITION
)
from new_env import get_simulation_components

POP_SIZE = 100
NUM_GENERATIONS = 100

PARAM_DIM = 21
PARAM_LOW = np.concatenate([np.full(10, -1.0), np.full(10, -1.0), np.array([1.0])])
PARAM_HIGH = np.concatenate([np.full(10, 1.0), np.full(10, 1.0), np.array([5.0])])
PARAM_LOW_JAX = jnp.array(PARAM_LOW)
PARAM_HIGH_JAX = jnp.array(PARAM_HIGH)

jax_env = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
    morphology=MJCFBrittleStarMorphology(specification=morphology_spec),
    arena=MJCFAquariumArena(configuration=arena_config),
    configuration=env_config, backend="MJX"
)

init_sim_state_fn, run_episode_fn = get_simulation_components(jax_env)
max_joint_limit = float(jax_env.action_space.high[0])
target_position = DEFAULT_TARGET_POSITION

def evaluate_parameters(rng_key, params):
    params_clipped = jnp.clip(params, PARAM_LOW_JAX, PARAM_HIGH_JAX)
    initial_state = init_sim_state_fn(rng_key, target_position)
    reward = run_episode_fn(initial_state, params_clipped, max_joint_limit)
    return reward

evaluate_fn = jax.jit(jax.vmap(evaluate_parameters, in_axes=(0, 0)))

strategy = OpenES(popsize=POP_SIZE, num_dims=PARAM_DIM, maximize=True)
es_params = strategy.default_params

rng = jax.random.PRNGKey(SEED)
rng, rng_init = jax.random.split(rng)
initial_mean = (PARAM_LOW_JAX + PARAM_HIGH_JAX) / 2.0
es_state = strategy.initialize(rng_init, es_params, init_mean=initial_mean)

print(f"Strategy initialized: OpenES. Population size: {POP_SIZE}, Parameter dim: {PARAM_DIM}")

# Removed wandb initialization block

print(f"Starting training for {NUM_GENERATIONS} generations...")
start_time = time.time()
log_interval = 10 # Keep simple print logging interval

for generation in tqdm(range(NUM_GENERATIONS), desc="Generation"):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, es_state = strategy.ask(rng_gen, es_state, es_params)
    eval_keys = jax.random.split(key=rng_eval, num=POP_SIZE)
    fitness = evaluate_fn(eval_keys, x)
    es_state = strategy.tell(x, fitness, es_state, es_params)

    # Simple print logging instead of wandb
    if generation % log_interval == 0 or generation == NUM_GENERATIONS -1 :
         best_fitness = es_state.best_fitness
         print(f"Generation: {generation+1}/{NUM_GENERATIONS}, Best Fitness (Max Reward): {best_fitness:.4f}")


end_time = time.time()
# Removed wandb.finish()

print("\n" + "="*30)
print("Optimization Finished!")
print(f"Total time: {end_time - start_time:.2f} seconds")

best_params = es_state.best_member
best_fitness = es_state.best_fitness

print(f"Best fitness found (Max Reward): {best_fitness:.6f}")
print(f"Best CPG parameters found (best individual):")
np.set_printoptions(precision=6, suppress=True)
print(np.array(best_params))

print("Script finished.")
