import time

import jax
import jax.numpy as jnp

from groep6.SimulationState import SimulationState
from groep6.es.brittle_star_environment import EpisodeEvaluator, calculate_relative_direction, get_joint_positions
from groep6.defaults import (
    NUM_ARMS,
    NUM_OSCILLATORS_PER_ARM, SEED, MAX_STEPS_PER_EPISODE, NUM_INFERENCES_PER_TRIAL, FIXED_OMEGA, VIDEO_TARGET_POSITION
)
from groep6.es.defaults import MODEL_FILE
from groep6.nn import CPGController, load_model_params
from groep6.render import show_video, post_render


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


def create_video(model_path: str = MODEL_FILE, target_pos: jnp.ndarray = VIDEO_TARGET_POSITION):
    target_pos_3d = jnp.concatenate([target_pos, jnp.array([0.0])])
    master_key = jax.random.PRNGKey(SEED)
    rng_init, rng_env_reset, rng_cpg_reset = jax.random.split(master_key, 3)

    evaluator = EpisodeEvaluator()
    sim_state = evaluator.create_initial_state(rng=rng_env_reset, target_pos=target_pos_3d)

    frames = []
    step_count = 0
    start_sim_time = time.time()
    while not sim_state.terminated and not sim_state.truncated:
        if step_count % (MAX_STEPS_PER_EPISODE / NUM_INFERENCES_PER_TRIAL) == 0:
            # modulate CPG parameters
            generated_rx_params = infer_model(model_path, rng_init, sim_state)
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

    video_filename = f"simulation_target_{target_pos[0]}_{target_pos[1]}.mp4"
    show_video(images=frames, sim_time=evaluator.env.environment_configuration.simulation_time, path=video_filename)
    print(f"Video saved to {video_filename}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Create a video using a model")
    parser.add_argument("model_file", help="The msgpack model of the brittle star")
    parser.add_argument("--target_position", type=float, nargs=2, default=VIDEO_TARGET_POSITION, )
    args = parser.parse_args()

    create_video(args.model_file, jnp.array(args.target_position))
