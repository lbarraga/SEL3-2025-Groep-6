import time
import jax
import jax.numpy as jnp
from functools import partial

from evosax import OpenES

from new_env import SimulationState, run_episode_logic
from config import (
    create_environment, DEFAULT_TARGET_POSITION, CONTROL_TIMESTEP,
    NUM_ARMS, NUM_OSCILLATORS_PER_ARM
)
from cpg import CPG

def assemble_initial_state(env_state, cpg_state):
    current_pos = env_state.observations["disk_position"]
    target_pos_3d = jnp.concatenate([env_state.info["xy_target_position"], jnp.array([0.0])])
    initial_distance = jnp.linalg.norm(current_pos - target_pos_3d)
    return SimulationState(
        env_state=env_state, cpg_state=cpg_state,
        initial_distance=initial_distance, best_distance=initial_distance,
        current_distance=initial_distance, steps_taken=jnp.array(0, dtype=jnp.int32),
        last_progress_step=jnp.array(0, dtype=jnp.int32), terminated=jnp.array(False),
        truncated=jnp.array(False), reward=jnp.array(0.0, dtype=jnp.float32)
    )

@partial(jax.jit, static_argnames=(
    'jit_reset', 'cpg_reset_fn', 'assemble_fn', 'run_episode_fn'
))
def evaluate_single(rng, params, jit_reset, cpg_reset_fn, assemble_fn, run_episode_fn, target_pos):
    rng_env, rng_cpg = jax.random.split(rng)
    initial_env_state = jit_reset(rng=rng_env, target_position=target_pos)
    initial_cpg_state = cpg_reset_fn(rng=rng_cpg)
    initial_sim_state = assemble_fn(initial_env_state, initial_cpg_state)
    fitness = run_episode_fn(initial_sim_state, params)
    return fitness

if __name__ == "__main__":
    print("--- Starting EvoSax Optimization (OpenES) - Clipping Removed ---")

    POPULATION_SIZE = 100
    NUM_GENERATIONS = 100
    SIGMA_INIT = 0.1
    SEED = 55

    env = create_environment()
    cpg_instance = CPG(dt=CONTROL_TIMESTEP)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    max_joint_limit = float(env.action_space.high[0])
    num_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1
    print(f"CPG Parameters per individual: {num_params}")
    print(f"Max Joint Limit (from action space): {max_joint_limit:.4f}")

    cpg_reset_fn = cpg_instance.reset
    run_episode_partial = partial(
        run_episode_logic,
        step_fn=jit_step,
        cpg=cpg_instance,
        max_joint_limit=max_joint_limit
    )
    evaluate_single_partial = partial(
        evaluate_single,
        jit_reset=jit_reset,
        cpg_reset_fn=cpg_reset_fn,
        assemble_fn=assemble_initial_state,
        run_episode_fn=run_episode_partial,
        target_pos=DEFAULT_TARGET_POSITION
    )

    evaluate_batch_fn = jax.jit(jax.vmap(evaluate_single_partial, in_axes=(0, 0)))
    print("Batched evaluation function prepared.")

    master_key = jax.random.PRNGKey(SEED)
    rng = master_key

    strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_params, sigma_init=SIGMA_INIT, maximize=True)
    es_params = strategy.default_params

    initial_mean = jnp.zeros(num_params, dtype=jnp.float32)

    es_state = strategy.initialize(rng, es_params, init_mean=initial_mean)
    print("EvoSax OpenES strategy initialized with zero initial mean.")

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

