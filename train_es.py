import time

import cma
import numpy as np

from brittlestar_gym_environment import BrittleStarEnv

SEED = 42
MAX_GENERATIONS = 100
INITIAL_SIGMA = 0.5

print("Initializing environment...")
env = BrittleStarEnv(seed=SEED)
param_dim = env.action_space.shape[0]
lower_bounds = env.action_space.low
upper_bounds = env.action_space.high
print(f"Parameter dimension: {param_dim}")
print(f"Lower bounds: {lower_bounds}")
print(f"Upper bounds: {upper_bounds}")

def evaluate_cpg_params(params, env_instance):
    params = np.clip(params, lower_bounds, upper_bounds)
    _observation, _info = env_instance.reset(seed=SEED)
    _observation, reward, terminated, truncated, info = env_instance.step(params)
    fitness = -reward
    return fitness

initial_params = (lower_bounds + upper_bounds) / 2.0
options = {
    'bounds': [lower_bounds, upper_bounds],
    'seed': SEED,
    'maxiter': MAX_GENERATIONS,
    'verbose': -3
}

print("Initializing CMA-ES...")
es = cma.CMAEvolutionStrategy(initial_params, INITIAL_SIGMA, options)
print(f"Population size: {es.popsize}")

print(f"Starting optimization for {MAX_GENERATIONS} generations (single-threaded)...")
start_time = time.time()
generation = 0

while not es.stop():
    generation += 1
    params_population = es.ask()

    fitness_scores = []
    for params in params_population:
        score = evaluate_cpg_params(params, env)
        fitness_scores.append(score)

    es.tell(params_population, fitness_scores)

    print(f"Generation: {generation}/{MAX_GENERATIONS}, Best Fitness (-Reward): {es.best.f:.4f}")

end_time = time.time()
print("\n" + "="*30)
print("Optimization Finished!")
print(f"Total time: {end_time - start_time:.2f} seconds")
if es.result.fbest is not None:
    print(f"Best fitness found (-Reward): {es.result.fbest:.6f}")
    print(f"Best CPG parameters found:")
    np.set_printoptions(precision=6, suppress=True)
    print(es.result.xbest)
    # np.save("best_cpg_params_es_minimal.npy", es.result.xbest)
else:
    print("Optimization did not produce a valid result (or was stopped early).")

env.close()
print("Environment closed.")
print("Script finished.")