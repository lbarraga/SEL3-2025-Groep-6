from stable_baselines3 import PPO

from groep6.defaults import MAX_STEPS_PER_PPO_EPISODE, VIDEO_TARGET_POSITION
from groep6.ppo.brittle_star_gym_environment import BrittleStarGymEnv
from groep6.ppo.defaults import DEVICE
from groep6.render import show_video

import jax.numpy as jnp


def create_video(model_ppo, output_file, env):
    observation, info = env.reset()
    frames = []

    terminated = False
    truncated = False
    step = 0

    action, _states = model_ppo.predict(observation, deterministic=True)
    env.sim_state = env.sim_state.replace(cpg_state=env.modulate_cpg(env.sim_state.cpg_state, action))

    while step < 500 and not terminated and not truncated:
        if step % MAX_STEPS_PER_PPO_EPISODE == 0:
            observation = env.get_observation()
            action, _states = model_ppo.predict(observation, deterministic=True)
            env.sim_state = env.sim_state.replace(cpg_state=env.modulate_cpg(env.sim_state.cpg_state, action))

        env.sim_state = env.simulation_single_step_logic(env.sim_state)
        frames.append(env.render())
        step += 1

    # Save video of trained agent
    show_video(images=frames, sim_time=100, path=output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a video using a model, all the arguments should be the same as during the training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model_path",
                        help="The stable baseline 3 model (.zip) of the brittle star")
    parser.add_argument("--output_file", type=str, default="../trained_agent.mp4",
                        help="The output file to save the video")
    parser.add_argument("--device", type=str, default=DEVICE,
                        help="Which device to run on: cpu, gpu, auto")
    parser.add_argument("--target_position", default=VIDEO_TARGET_POSITION, type=float, nargs=2,
                        help="The target position of the brittle star")

    args = parser.parse_args()

    # Create the environment
    target_pos_3d = jnp.concatenate([jnp.array(args.target_position), jnp.array([0.0])])
    env = BrittleStarGymEnv(target_position=target_pos_3d)

    model = PPO("MlpPolicy", env, verbose=0, device=args.device)
    trained_model = model.load(args.model_path)

    create_video(trained_model, args.output_file, env)
