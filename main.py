from functools import partial

import config
from cpg import *
from environment import *
from render import *

rng = jax.random.PRNGKey(seed=0)
env = create_directed_environment(
    config.morphology_specification,
    config.arena_configuration,
    config.environment_configuration,
    "MJX"
)

# Fix the target location
env_fixed_target_reset_fn = jax.jit(partial(env.reset, target_position=(-1.25, 0.75, 0.)))
env_step_fn = jax.jit(env.step)
jit_reset = jax.jit(env.reset)

cpg = create_cpg()
cpg_state = cpg.reset(rng=jax.random.PRNGKey(0))
# We set the max_joint_limit to only 25% of the true joint range of motion (you can test yourself what happens if we don't by changing this value).
cpg_state = modulate_cpg(cpg_state=cpg_state, leading_arm_index=0, max_joint_limit=env.action_space.high[0] * 0.25)

done = False
frames = []
env_state = jit_reset(rng=jax.random.PRNGKey(seed=0))
while not (env_state.terminated | env_state.truncated):
    cpg_state = cpg.step(state=cpg_state)
    actions = map_cpg_outputs_to_actions(cpg_state=cpg_state)
    env_state = env_step_fn(state=env_state, action=actions)
    frame = post_render(env.render(state=env_state), environment_configuration=config.environment_configuration)
    frames.append(frame)

show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")