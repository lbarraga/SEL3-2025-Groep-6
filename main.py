import time

import jax
import jax.numpy as jnp

from brittle_star_environment import EpisodeEvaluator
from config import (
    NUM_ARMS,
    NUM_OSCILLATORS_PER_ARM, SEED, FIXED_OMEGA
)
from nn import CPGController, load_model_params
from render import show_video, post_render

FIXED_OMEGA = 4.5
MODEL_FILENAME = "final_model_gen2000.msgpack" # Specify model file path here
TARGET_POS = jnp.array([-1, -1])


def infer_model(path: str, rng: jnp.ndarray, direction_to_target: jnp.ndarray):

    # Load model parameters from file
    num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
    model = CPGController(num_outputs=num_cpg_params_to_generate)
    dummy_input = jnp.zeros((1, 2))
    example_params_tree = model.init(rng, dummy_input)['params']
    model_params = load_model_params(path, example_params_tree)

    return model.apply({'params': model_params}, direction_to_target)

if __name__ == '__main__':

    target_pos_3d = jnp.concatenate([TARGET_POS, jnp.array([0.0])])
    master_key = jax.random.PRNGKey(SEED)
    rng_init, rng_env_reset, rng_cpg_reset = jax.random.split(master_key, 3)

    # calculate direction to target
    direction = TARGET_POS / jnp.linalg.norm(TARGET_POS)

    # infer model
    generated_rx_params = infer_model(MODEL_FILENAME, rng_init, direction)
    cpg_params = jnp.concatenate([generated_rx_params, jnp.array([FIXED_OMEGA])])

    evaluator = EpisodeEvaluator()
    sim_state = evaluator.create_initial_state(rng=rng_env_reset, target_pos=target_pos_3d)
    modulated_cpg_params = evaluator.modulate_cpg(sim_state.cpg_state, cpg_params)
    sim_state = sim_state.replace(cpg_state=modulated_cpg_params)

    frames = []
    step_count = 0
    start_sim_time = time.time()
    while not sim_state.terminated and not sim_state.truncated:
        sim_state = evaluator.simulation_single_step_logic(sim_state)
        frame = evaluator.env.render(state=sim_state.env_state)
        processed_frame = post_render(frame, environment_configuration=evaluator.env.environment_configuration)
        frames.append(jnp.array(processed_frame))
        step_count += 1

    end_sim_time = time.time()
    print(f"Simulation finished after {step_count} steps.")
    print(f"Final distance: {sim_state.current_distance:.4f}")
    print(f"Terminated: {sim_state.terminated}, Truncated: {sim_state.truncated}")
    print(f"Simulation wall-clock time: {end_sim_time - start_sim_time:.2f}s")

    video_filename = f"simulation_target_{TARGET_POS[0]}_{TARGET_POS[1]}.mp4"
    show_video(images=frames, sim_time=evaluator.env.environment_configuration.simulation_time, path=video_filename)
    print(f"Video saved to {video_filename}")

