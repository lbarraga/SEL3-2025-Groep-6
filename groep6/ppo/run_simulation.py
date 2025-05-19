from groep6.config import MAX_STEPS_PER_PPO_EPISODE
from groep6.ppo.brittle_star_gym_environment import BrittleStarGymEnv
from groep6.ppo.config import make_model
from groep6.render import show_video

def model_load(model_path: str):
    """Load the PPO model from the specified path."""
    model = make_model()
    trained_model = model.load(model_path)
    trained_model.device = "cpu"
    return trained_model

def create_video(model_ppo):
    # Create the environment
    env = BrittleStarGymEnv()

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
    show_video(images=frames, sim_time=100, path="../trained_agent.mp4")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a video using a model")
    parser.add_argument("model_file", help="The msgpack model of the brittle star")
    args = parser.parse_args()

    trained_model = model_load(args.model_file)
    create_video(trained_model)
