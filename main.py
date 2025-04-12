import time
import jax
import jax.numpy as jnp

from brittle_star_environment import create_initial_simulation_state, simulation_single_step_logic
from config import (
    create_environment,
    CONTROL_TIMESTEP,
    NUM_ARMS,
    NUM_OSCILLATORS_PER_ARM
)
from cpg import CPG, modulate_cpg
from render import show_video, post_render
from nn import CPGController, load_model_params

SEED = 42
FIXED_OMEGA = 4.5
MODEL_FILENAME = "final_model_gen100.msgpack" # Specify model file path here
TARGET_X = 2 # Specify target X here
TARGET_Y = 2 # Specify target Y here

env = create_environment()
cpg = CPG(dt=CONTROL_TIMESTEP)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

max_joint_limit = float(env.action_space.high[0])
master_key = jax.random.PRNGKey(SEED)
rng_init, rng_env_reset, rng_cpg_reset = jax.random.split(master_key, 3)

num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
model = CPGController(num_outputs=num_cpg_params_to_generate)
dummy_input = jnp.zeros((1, 2))
example_params_tree = model.init(rng_init, dummy_input)['params']
loaded_params_tree = load_model_params(MODEL_FILENAME, example_params_tree)

target_pos_3d = jnp.array([TARGET_X, TARGET_Y, 0.0])
target_pos_2d = target_pos_3d[:2]

norm = jnp.linalg.norm(target_pos_2d)
safe_norm = jnp.where(norm < 1e-6, 1.0, norm)
normalized_direction = target_pos_2d / safe_norm
normalized_direction = jnp.where(norm < 1e-6, jnp.array([1.0, 0.0]), normalized_direction)

generated_rx_params = model.apply({'params': loaded_params_tree}, normalized_direction)

cpg_params = jnp.concatenate([generated_rx_params, jnp.array([FIXED_OMEGA])])

initial_env_state = jit_reset(rng=rng_env_reset, target_position=target_pos_3d)
initial_cpg_state = cpg.reset(rng=rng_cpg_reset)

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
    print(sim_state.terminated, sim_state.truncated)

end_sim_time = time.time()
print(f"Simulation finished after {step_count} steps.")
print(f"Final distance: {sim_state.current_distance:.4f}")
print(f"Terminated: {sim_state.terminated}, Truncated: {sim_state.truncated}")
print(f"Simulation wall-clock time: {end_sim_time - start_sim_time:.2f}s")

if frames:
    video_filename = f"simulation_target_{TARGET_X}_{TARGET_Y}.mp4"
    show_video(images=frames, sim_time=env.environment_configuration.simulation_time, path=video_filename)
    print(f"Video saved to {video_filename}")

