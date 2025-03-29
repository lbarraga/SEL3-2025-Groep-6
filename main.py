from functools import partial

import config
from cpg import *
from environment import *
from neural_network import BrittleStarNN
from render import *

rng = jax.random.PRNGKey(seed=0)
env = create_directed_environment(
    config.morphology_specification,
    config.arena_configuration,
    config.environment_configuration,
    "MJX"
)

# Fix the target location
env_step_fn = jax.jit(env.step)
jit_reset = jax.jit(partial(env.reset, target_position=(-1.25, 0.75, 0.)))

cpg = create_cpg()
cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))

# init neural network
model = BrittleStarNN()
params = model.init(rng, jnp.zeros((6,)))

done = False
frames = []
env_state: MJXEnvState = jit_reset(rng=jax.random.PRNGKey(seed=0))
while not (env_state.terminated | env_state.truncated):

    current_position = get_brittle_star_position(env_state)
    target_position = get_target_position(env_state)
    nn_input = jnp.concatenate([current_position, target_position])

    output = model.apply(params, nn_input)
    new_R, new_X, new_omega = output[:10], output[10:20], output[20]

    cpg_state = modulate_cpg(
        cpg_state=cpg_state,
        new_R=new_R,
        new_X=new_X,
        new_omega=new_omega,
        max_joint_limit=env.action_space.high[0] * 0.25
    )

    cpg_state = cpg.step(state=cpg_state)
    actions = map_cpg_outputs_to_actions(cpg_state=cpg_state)
    env_state = env_step_fn(state=env_state, action=actions)
    frame = post_render(env.render(state=env_state), environment_configuration=config.environment_configuration)
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")