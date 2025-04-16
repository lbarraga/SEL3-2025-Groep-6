import pickle
import time

import jax
import jax.numpy as jnp
import optuna
from evosax import OpenES
from jax.flatten_util import ravel_pytree

from brittle_star_environment import create_evaluation_fn
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, SEED, FIXED_OMEGA, TARGET_SAMPLING_RADIUS
from nn import CPGController
from util import generate_cpg_for_eval, print_optuna_results
from wandb_evosax_logger import WandbEvosaxLogger

# reduced number of generations for optuna
NUM_GENERATIONS = 1
POPULATION_SIZE = 200

WANDB_PROJECT_NAME = "evosax_brittle_star_nn_optuna2"

master_key = jax.random.PRNGKey(SEED)

num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
# here the model will change the hidden dimensions, so don't make in advance, just make the dummy input
dummy_input = jnp.zeros((1, 2))

evaluate_batch_fn = create_evaluation_fn()
generate_batch_cpg_for_eval = jax.vmap(generate_cpg_for_eval, in_axes=(0, 0, None, None))

search_space = {
    "sigma_init": [0.1, 0.15, 0.2, 0.25],        # exploration vs exploitation
    "hidden_dim": [16, 32, 64],                  # hidden dimension of the nn
}

def objective(trial):
    # Initialize the model parameters
    sigma_init = trial.suggest_categorical("sigma_init", search_space["sigma_init"])
    hidden_dim = trial.suggest_categorical("hidden_dim", search_space["hidden_dim"])

    print(f"Running generation {trial.number} with sigma_init={sigma_init} and hidden_dim={hidden_dim}")

    # Initialize the model
    model = CPGController(num_outputs=num_cpg_params_to_generate, hidden_dim=hidden_dim)

    rng, model_init_rng = jax.random.split(master_key)

    initial_model_params_tree = model.init(model_init_rng, dummy_input)['params']
    flat_initial_model_params, unravel_fn = ravel_pytree(initial_model_params_tree)
    num_model_params = flat_initial_model_params.shape[0]

    logger = WandbEvosaxLogger(
        project_name=WANDB_PROJECT_NAME,
        config={
            "population_size": POPULATION_SIZE,
            "num_generations": NUM_GENERATIONS,
            "sigma_init": sigma_init,
            "seed": SEED,
            "num_model_params": num_model_params,
            "fixed_omega": FIXED_OMEGA,
            "target_sampling_region": f"Circle perimeter radius {TARGET_SAMPLING_RADIUS}",
            "nn_hidden_dim": hidden_dim,
        }
    )

    strategy = OpenES(popsize=POPULATION_SIZE, num_dims=num_model_params, sigma_init=sigma_init, maximize=True)
    es_params = strategy.default_params
    es_state = strategy.initialize(rng, es_params, init_mean=flat_initial_model_params)

    fitness = jnp.zeros(POPULATION_SIZE)
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

    logger.log_summary(es_state)
    logger.finish()

    return jnp.mean(fitness).item()

def get_study(study_name: str):
    try:
        with open(study_name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        sampler = optuna.samplers.GridSampler(search_space)
        return optuna.create_study(
            study_name="evosax_brittle_star_nn",
            direction="maximize",
            sampler=sampler
        )

def run_optuna(n_trials: int = 10):
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
    run_optuna(n_trials=12)
    print_optuna_results("evosax_brittle_star_nn.pkl")
