import optax
import jax

from brittlestar_gym_environment import BrittleStarEnv
from evosax.algorithms import Open_ES as ES
from es_problem import BSCPGProblem as Problem
from evosax.problems.networks import MLP, tanh_output_fn

seed = 0
key = jax.random.PRNGKey(seed)

policy = MLP(
    layer_sizes=(6, 32, 32, 32, 21),
    output_fn=tanh_output_fn,
)

problem = Problem(
    env_name="brittle_star",
    policy=policy,
    episode_length=1000,
    num_rollouts=16,
    use_normalize_obs=True,
)

key, subkey = jax.random.split(key)
problem_state = problem.init(key)

key, subkey = jax.random.split(key)
solution = problem.sample(subkey)

env = BrittleStarEnv()

num_generations = 512
lr_schedule = optax.exponential_decay(
    init_value=0.01,
    transition_steps=num_generations,
    decay_rate=0.1,
)
std_schedule = optax.exponential_decay(
    init_value=0.05,
    transition_steps=num_generations,
    decay_rate=0.2,
)

es = ES(
    population_size=256,
    solution=solution,
    optimizer=optax.adam(learning_rate=lr_schedule),
    std_schedule=std_schedule
)

params = es.default_params

def step(carry, key):
    state, params, problem_state = carry
    key_ask, key_eval, key_tell = jax.random.split(key, 3)

    population, state = es.ask(key_ask, state, params)

    fitness, problem_state, _ = problem.eval(key_eval, population, problem_state)

    state, metrics = es.tell(
        key_tell, population, -fitness, state, params
    )  # Minimize fitness

    return (state, params, problem_state), metrics

key, subkey = jax.random.split(key)
state = es.init(subkey, solution, params)

fitness_log = []
log_period = 32
for i in range(num_generations // log_period):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, log_period)
    (state, params, problem_state), metrics = jax.lax.scan(
        step,
        (state, params, problem_state),
        keys,
    )

    mean = es.get_mean(state)

    key, subkey = jax.random.split(key)
    fitness, problem_state, info = problem.eval(
        key, jax.tree.map(lambda x: x[None], mean), problem_state
    )
    print(f"Generation {(i + 1) * log_period:3d} | Mean fitness: {fitness.mean():.2f}")
    
    
# visualise policy
mean = es.get_mean(state)
mean = es._unravel_solution(state.best_solution)

key, subkey = jax.random.split(key)
fitness, problem_state, info = problem.eval(
    key, jax.tree.map(lambda x: x[None], mean), problem_state
)
print(fitness[0])