
import pickle

import jax
import jax.numpy as jnp
import optuna
from evosax import OpenES
from jax.flatten_util import ravel_pytree

from brittle_star_environment import create_evaluation_fn
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, SEED
from nn import CPGController
from util import generate_cpg_for_eval, print_optuna_results

# reduced number of generations for optuna
NUM_GENERATIONS = 100

master_key = jax.random.PRNGKey(SEED)

num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
# here the model will change the hidden dimensions, so don't make in advance, just make the dummy input
dummy_input = jnp.zeros((1, 2))

evaluate_batch_fn = create_evaluation_fn()
generate_batch_cpg_for_eval = jax.vmap(generate_cpg_for_eval, in_axes=(0, 0, None, None, None))

def objective(trial):
    # Initialize the model parameters
    sigma_init = trial.suggest_categorical("sigma_init", [0.05, 0.1, 0.15, 0.2, 0.25])     # exploration vs exploitation
    population_size = trial.suggest_categorical("population_size", [100, 200, 300, 400])   # population size
    fixed_omega = trial.suggest_categorical("fixed_omega", [4.0, 4.5, 5.0])                # frequency of oscillators
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64])                  # hidden dimension of the nn

    print(f"Running generation {trial.number} with sigma_init={sigma_init}, population_size={population_size}, fixed_omega={fixed_omega}")

    # Initialize the model
    model = CPGController(num_outputs=num_cpg_params_to_generate, hidden_dim=hidden_dim)

    rng, model_init_rng = jax.random.split(master_key)

    initial_model_params_tree = model.init(model_init_rng, dummy_input)['params']
    flat_initial_model_params, unravel_fn = ravel_pytree(initial_model_params_tree)
    num_model_params = flat_initial_model_params.shape[0]

    strategy = OpenES(popsize=population_size, num_dims=num_model_params, sigma_init=sigma_init, maximize=True)
    es_params = strategy.default_params
    es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

    fitness = jnp.zeros(population_size)
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation}")
        rng, rng_ask, rng_gen, rng_eval = jax.random.split(rng, 4)

        flat_model_params, es_state = strategy.ask(rng_ask, es_state, es_params)

        rngs = jax.random.split(rng_gen, population_size)
        cpg_params_pop, target_pos_pop = generate_batch_cpg_for_eval(
            rngs, flat_model_params, model, unravel_fn, fixed_omega
        )

        rng_eval_batch = jax.random.split(rng_eval, population_size)
        fitness, final_states = evaluate_batch_fn(rng_eval_batch, cpg_params_pop, target_pos_pop)

        es_state = strategy.tell(flat_model_params, fitness, es_state, es_params)

    return jnp.mean(fitness).item()

def get_study(study_name: str):
    try:
        with open(study_name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return optuna.create_study(
            study_name="evosax_brittle_star_nn",
            direction="maximize"
        )

def run_optuna(n_trials: int = 100):
    # Initialize the study
    study_name = "evosax_brittle_star_nn.pkl"
    study = get_study(study_name)

    # Run the optimization
    study.optimize(objective, n_trials=n_trials)

    # Save the study to a pickle file
    with open(study_name, "wb") as f:
        pickle.dump(study, f)
        print(f"Study saved to {study_name}")

    # Print and save the results
    print("Number of finished trials: ", len(study.trials))

    params = study.best_trial.params
    values = study.best_trial.values

    print("Best parameters:", params)
    print("Best values:", values)


if __name__ == "__main__":
    run_optuna(n_trials=240)
    print_optuna_results("evosax_brittle_star_nn.pkl")
