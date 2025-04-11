import time
import jax
import jax.numpy as jnp

from evosax import OpenES

from new_env import create_evaluation_fn
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM

if __name__ == "__main__":

    POPULATION_SIZE = 100
    NUM_GENERATIONS = 100
    SIGMA_INIT = 0.1
    SEED = 55

    num_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1
    evaluate_batch_fn = create_evaluation_fn()

    master_key = jax.random.PRNGKey(SEED)
    rng = master_key

    strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_params, sigma_init=SIGMA_INIT, maximize=True)
    es_params = strategy.default_params
    initial_mean = jnp.zeros(num_params, dtype=jnp.float32)
    es_state = strategy.initialize(rng, es_params, init_mean=initial_mean)

    start_time = time.time()
    print(f"Running {NUM_GENERATIONS} generations with population size {POPULATION_SIZE}...")
    for generation in range(NUM_GENERATIONS):
        loop_start_time = time.time()
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)

        x, es_state = strategy.ask(rng_ask, es_state, es_params)
        rng_eval_batch = jax.random.split(rng_eval, POPULATION_SIZE)
        fitness = evaluate_batch_fn(rng_eval_batch, x)
        es_state = strategy.tell(x, fitness, es_state, es_params)

        gen_time = time.time() - loop_start_time
        es_state.best_fitness.block_until_ready()
        print(f"Generation {generation+1}/{NUM_GENERATIONS} | Best Fitness: {es_state.best_fitness:.4f} | Time: {gen_time:.2f}s")



    total_time = time.time() - start_time
    print(f"--- Optimization Finished ---")
    print(f"Total time: {total_time:.2f} seconds ({total_time / NUM_GENERATIONS:.2f}s/gen avg)")
    final_best_fitness = es_state.best_fitness.block_until_ready()
    print(f"Final Best Fitness: {final_best_fitness:.4f}")

    final_params = es_state.mean
    params_list = final_params.tolist()
    params_str = ", ".join(f"{p:.6f}" for p in params_list)
    print(f"Final Parameters (Mean): [{params_str}]")
