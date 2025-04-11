import time

import jax
import jax.numpy as jnp

from brittle_star_environment import create_initial_simulation_state, simulation_single_step_logic
from config import (
    create_environment,
    DEFAULT_TARGET_POSITION,
    CONTROL_TIMESTEP,
    NUM_ARMS,
    NUM_OSCILLATORS_PER_ARM
)
from cpg import CPG, modulate_cpg

from render import show_video, post_render

cpg_params = jnp.array([
    -0.778954, -0.277990, 0.094271, -0.578112, -1.130316,
    0.448246, 1.224521, -0.109453, 0.572109, 0.263016,
    -0.744902, -0.151820, 0.283899, -0.713379, -0.041819,
    -0.198762, -0.349436, 0.017592, 0.217051, 0.450029,
    5.064455
])
SEED = 42

print("Setting up environment and CPG...")
env = create_environment()
cpg = CPG(dt=CONTROL_TIMESTEP)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

max_joint_limit = float(env.action_space.high[0])

master_key = jax.random.PRNGKey(SEED)
rng_env_reset, rng_cpg_reset = jax.random.split(master_key)

print("Resetting environment and CPG...")
initial_env_state = jit_reset(rng=rng_env_reset, target_position=DEFAULT_TARGET_POSITION)
initial_cpg_state = cpg.reset(rng=rng_cpg_reset)

print("Modulating CPG state...")
num_r_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM
new_R = cpg_params[:num_r_params]
new_X = cpg_params[num_r_params:-1]
new_omega = cpg_params[-1]

modulated_cpg_state = modulate_cpg(
    cpg_state=initial_cpg_state,
    new_R=new_R,
    new_X=new_X,
    new_omega=new_omega,
    max_joint_limit=max_joint_limit
)

sim_state = create_initial_simulation_state(initial_env_state, modulated_cpg_state)

frames = []

step_count = 0
start_sim_time = time.time()
while not sim_state.terminated and not sim_state.truncated:
    sim_state = simulation_single_step_logic(sim_state, jit_step, cpg)
    frame = env.render(state=sim_state.env_state)
    processed_frame = post_render(frame, environment_configuration=env.environment_configuration)
    frames.append(jnp.array(processed_frame))
    step_count += 1

end_sim_time = time.time()
print(f"Simulation finished after {step_count} steps.")
print(f"Final distance: {sim_state.current_distance:.4f}")
print(f"Terminated: {sim_state.terminated}, Truncated: {sim_state.truncated}")
print(f"Simulation wall-clock time: {end_sim_time - start_sim_time:.2f}s")

print("Generating video...")
show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path="simulation.mp4")
print("Video saved to simulation_new_api.mp4")

