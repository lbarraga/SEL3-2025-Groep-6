import time

import jax
import jax.numpy as jnp

from groep6.SimulationState import SimulationState
from groep6.es.brittle_star_environment import EpisodeEvaluator, calculate_relative_direction, get_joint_positions
from groep6.config import (
    NUM_ARMS,
    NUM_OSCILLATORS_PER_ARM, SEED, MAX_STEPS_PER_EPISODE, NUM_INFERENCES_PER_TRIAL
)
from groep6.nn import CPGController, load_model_params
from groep6.render import show_video, post_render

FIXED_OMEGA = 4.5
MODEL_FILENAME = "final_model_gen300.msgpack"  # Specify model file path here
TARGET_POS = jnp.array([3, -1])


def infer_model(path: str, rng: jnp.ndarray, state: SimulationState):

    # Load model parameters from file
    num_cpg_params_to_generate = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2
    model = CPGController(num_outputs=num_cpg_params_to_generate)
    dummy_input = jnp.zeros((1, 31))
    example_params_tree = model.init(rng, dummy_input)['params']
    model_params = load_model_params(path, example_params_tree)

    direction_to_target = jnp.array([calculate_relative_direction(state)])
    joint_positions = get_joint_positions(state)
    nn_input = jnp.concatenate([direction_to_target, joint_positions])
    return model.apply({'params': model_params}, nn_input)

if __name__ == '__main__':

    target_pos_3d = jnp.concatenate([TARGET_POS, jnp.array([0.0])])
    master_key = jax.random.PRNGKey(SEED)
    rng_init, rng_env_reset, rng_cpg_reset = jax.random.split(master_key, 3)

    evaluator = EpisodeEvaluator()
    sim_state = evaluator.create_initial_state(rng=rng_env_reset, target_pos=target_pos_3d)

    # calculate direction to target
    direction = calculate_relative_direction(sim_state)

    frames = []
    step_count = 0
    start_sim_time = time.time()
    while not sim_state.terminated and not sim_state.truncated:
        if step_count % (MAX_STEPS_PER_EPISODE / NUM_INFERENCES_PER_TRIAL) == 0:
            # modulate CPG parameters
            generated_rx_params = infer_model(MODEL_FILENAME, rng_init, sim_state)
            cpg_params = jnp.concatenate([generated_rx_params, jnp.array([FIXED_OMEGA])])

            modulated_cpg_params = evaluator.modulate_cpg(sim_state.cpg_state, cpg_params)
            sim_state = sim_state.replace(cpg_state=modulated_cpg_params)

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

