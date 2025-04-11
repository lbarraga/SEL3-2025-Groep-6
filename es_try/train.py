from evosax import OpenES
from tqdm import tqdm
from new_env import *
from render import post_render, show_video, save_video

rng = jax.random.PRNGKey(seed=0)
env = bio_env_instance

def evaluate_parameters(rng: jax.random.PRNGKey,
                        parameters: jnp.ndarray) -> float:
    rng, env_rng = jax.random.split(key=rng, num=2)
    env_state = bio_env_reset_fn(rng=env_rng, target_position=TARGET_POSITION)

    cpg = cpg_instance
    rng, cpg_rng = jax.random.split(key=rng, num=2)
    cpg_state = cpg.reset(rng=cpg_rng)

    cpg_state = modulate_cpg(
        cpg_state=cpg_state,
        new_R=parameters[:10],
        new_X=parameters[10:20],
        new_omega=parameters[20],
        max_joint_limit=max_joint_limit
    )

    def _step(carry, _):
        _env_state, _cpg_state = carry

        _cpg_state = cpg.step(state=_cpg_state)
        _actions = map_cpg_outputs_fn(cpg_state=_cpg_state)
        _env_state = bio_env_step_fn(state=_env_state, action=_actions)

        return (_env_state, _cpg_state), None

    (final_env_state, _), _ = jax.lax.scan(
        f=_step,
        init=(env_state, cpg_state),
        length=env.environment_configuration.total_num_control_steps
    )

    fitness = calculate_distance(final_env_state, TARGET_POSITION)
    return fitness


def evaluate_parameters_visual(
        rng: jax.random.PRNGKey,
        parameters: jnp.ndarray,
) -> float:
    rng, env_rng = jax.random.split(key=rng, num=2)
    env_state = bio_env_reset_fn(rng=env_rng, target_position=TARGET_POSITION)

    cpg = cpg_instance
    rng, cpg_rng = jax.random.split(key=rng, num=2)
    cpg_state = cpg.reset(rng=cpg_rng)

    cpg_state = modulate_cpg(
        cpg_state=cpg_state,
        new_R=parameters[:10],
        new_X=parameters[10:20],
        new_omega=parameters[20],
        max_joint_limit=max_joint_limit
    )

    frames = []

    while not (env_state.terminated | env_state.truncated):
        cpg_state = cpg.step(state=cpg_state)
        actions = map_cpg_outputs_fn(cpg_state=cpg_state)
        env_state = bio_env_step_fn(state=env_state, action=actions)
        frame = post_render(env.render(state=env_state), environment_configuration=environment_configuration)
        frames.append(frame)

    print(frames[0].shape)
    save_video(images=frames, path="simulation.mp4")
    # show_video(images=frames, sim_time=env.environment_configuration.simulation_time)

    fitness = calculate_distance(env_state, TARGET_POSITION)
    return fitness

NUM_GENERATIONS = 5
NUM_PARAMETERS = 36
POP_SIZE = 100

# Instantiate the search strategy
rng = jax.random.PRNGKey(0)
strategy = OpenES(popsize=POP_SIZE, num_dims=NUM_PARAMETERS, maximize=True)
es_params = strategy.default_params
es_state = strategy.initialize(rng, es_params)

# Important: We parallelise the evaluation using jax.vmap!
evaluate_fn = jax.jit(jax.vmap(evaluate_parameters))
# Run ask-eval-tell loop -
for generation in tqdm(range(NUM_GENERATIONS), desc="Generation: "):
    rng, rng_gen, rng_eval = jax.random.split(rng, 3)
    x, es_state = strategy.ask(rng_gen, es_state, es_params)


    rng_eval = jax.random.split(key=rng_eval, num=POP_SIZE)
    fitness = evaluate_fn(rng_eval, x)
    es_state = strategy.tell(x, fitness, es_state, es_params)

# Get best overall population member & its fitness
es_state.best_member, es_state.best_fitness

evaluate_parameters_visual(rng=rng, parameters=es_state.best_member)