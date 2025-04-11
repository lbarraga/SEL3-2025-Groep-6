# from evosax.problems import GymnaxProblem as Problem
from evosax.problems.networks import MLP, categorical_output_fn

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from es_try.BrittleStarProblem import BrittleStarProblem

seed = 0
key = jax.random.key(seed)

policy = MLP(
    layer_sizes=(64, 64, 21),
    output_fn=categorical_output_fn,
)

problem = BrittleStarProblem(
    env_name="BrittleStar-v0",
    policy=policy,
    episode_length=200,
    num_rollouts=16,
)

key, subkey = jax.random.split(key)
problem_state = problem.init(key)

key, subkey = jax.random.split(key)
solution = problem.sample(subkey)

print(f"Number of pararmeters: {sum(leaf.size for leaf in jax.tree.leaves(solution))}")

from evosax.algorithms import Open_ES as ES

es = ES(
    population_size=16,
    solution=solution,
    optimizer=optax.adam(learning_rate=0.01),
    std_schedule=optax.constant_schedule(0.1),
)

params = es.default_params

def fitness_std_cond(population, fitness, state, params):
    return jnp.std(fitness) < 0.001

num_generations = 64

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

metrics_log = []
restart_generations = []
for i in range(num_generations):
    key, subkey = jax.random.split(key)
    key_ask, key_eval, key_tell = jax.random.split(subkey, 3)

    population, state = es.ask(key_ask, state, params)
    fitness, problem_state, info = problem.eval(key_eval, population, problem_state)
    state, metrics = es.tell(key_tell, population, -fitness, state, params)

    if fitness_std_cond(population, fitness, state, params):
        mean = es.get_mean(state)

        key, subkey = jax.random.split(key)
        state = es.init(subkey, mean, params)

        restart_generations.append(i)

    # Log metrics
    metrics_log.append(metrics)

print(f"Number of pararmeters: {sum(leaf.size for leaf in jax.tree.leaves(solution))}")

# Extract best fitness values from metrics_log
generations = list(range(len(metrics_log)))
best_fitness = [-metrics["best_fitness"] for metrics in metrics_log]

# Create the plot
plt.figure(figsize=(6, 3))
plt.plot(generations, best_fitness)

# Add vertical lines for restart generations
for gen in restart_generations:
    plt.axvline(x=gen, color="r", linestyle="--", alpha=0.7)

plt.xlabel("Generations")
plt.ylabel("Best Fitness")
plt.title("Best Fitness Across Generations")
plt.grid(True)
plt.tight_layout()
plt.show()
