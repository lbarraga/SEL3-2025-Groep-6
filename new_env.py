from functools import partial

import jax
import jax.numpy as jnp
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment

# Required imports - script will fail if these modules are not found
from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM,
    MAX_STEPS_PER_EPISODE, NO_PROGRESS_THRESHOLD,
    CONTROL_TIMESTEP, DEFAULT_TARGET_POSITION, create_environment
)
from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions


@jax.jit
def init_simulation_state_logic(rng, env: BrittleStarDirectedLocomotionEnvironment, cpg: CPG, target_position: jnp.ndarray):
    """
    Initializes the simulation state including environment and CPG states.
    """
    rng_env, rng_cpg = jax.random.split(rng)
    initial_env_state = env.reset(rng=rng_env, target_position=target_position)
    initial_cpg_state = cpg.reset(rng=rng_cpg)

    current_pos = initial_env_state.observations["disk_position"]
    target_pos = jnp.concatenate([initial_env_state.info["xy_target_position"], jnp.array([0.0])])
    initial_distance = jnp.linalg.norm(current_pos - target_pos)

    state = {
        "env_state": initial_env_state,
        "cpg_state": initial_cpg_state,
        "rng": rng,
        "initial_distance": initial_distance,
        "best_distance": initial_distance,
        "current_distance": initial_distance,
        "steps_taken": jnp.array(0),
        "last_progress_step": jnp.array(0),
        "terminated": jnp.array(False),
        "truncated": jnp.array(False),
        "reward": jnp.array(0.0)
    }
    return state

# Jitting with static arguments for env and cpg instances.
@partial(jax.jit, static_argnames=['env', 'cpg'])
def simulation_single_step_logic(state, env: BrittleStarDirectedLocomotionEnvironment, cpg: CPG):
    """
    Executes a single step in the simulation using the provided env and cpg.
    """
    new_cpg_state = cpg.step(state=state["cpg_state"])
    motor_actions = map_cpg_outputs_to_actions(
        cpg_state=new_cpg_state,
        num_arms=NUM_ARMS, # Assumes global config constants are appropriate
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        num_oscillators_per_arm=NUM_OSCILLATORS_PER_ARM
    )
    new_env_state = env.step(state=state["env_state"], action=motor_actions)

    current_position = new_env_state.observations["disk_position"]
    target_position = jnp.concatenate([new_env_state.info["xy_target_position"], jnp.array([0.0])])
    current_distance = jnp.linalg.norm(current_position - target_position)

    progress_made = current_distance < state["best_distance"]
    current_step_index = state["steps_taken"] + 1
    last_progress_step = jnp.where(progress_made, current_step_index, state["last_progress_step"])
    best_distance = jnp.minimum(state["best_distance"], current_distance)

    # Termination and truncation conditions
    terminated = (current_distance < 0.1) | new_env_state.terminated
    truncated = new_env_state.truncated | ((current_step_index - last_progress_step) > NO_PROGRESS_THRESHOLD)

    new_state = state.copy()
    new_state.update({
        "env_state": new_env_state,
        "cpg_state": new_cpg_state,
        "steps_taken": current_step_index,
        "best_distance": best_distance,
        "current_distance": current_distance,
        "last_progress_step": last_progress_step,
        "terminated": terminated,
        "truncated": truncated,
    })
    return new_state

# Jitting with static arguments for env, cpg, and max_joint_limit.
@partial(jax.jit, static_argnames=['env', 'cpg', 'max_joint_limit'])
def run_episode_logic(initial_state, cpg_params: jnp.ndarray, env: BrittleStarDirectedLocomotionEnvironment, cpg: CPG, max_joint_limit: float):
    """
    Runs a full episode simulation loop, returning the total reward.
    """
    cpg_params = jnp.asarray(cpg_params)
    # Adapt parameter slicing based on your actual CPG structure
    num_cpg_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1 # Example: R, X per oscillator pair + 1 global omega
    if len(cpg_params) != num_cpg_params:
         raise ValueError(f"Expected {num_cpg_params} CPG parameters, got {len(cpg_params)}")

    new_R = cpg_params[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
    new_X = cpg_params[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:-1]
    new_omega = cpg_params[-1]

    modulated_cpg_state = modulate_cpg(
        cpg_state=initial_state["cpg_state"],
        new_R=new_R,
        new_X=new_X,
        new_omega=new_omega,
        max_joint_limit=max_joint_limit
    )

    loop_state = initial_state.copy()
    loop_state["cpg_state"] = modulated_cpg_state
    loop_state["distance_before_loop"] = initial_state["initial_distance"]

    # Define the step function for the loop body, passing static args
    def body_step(current_loop_state):
        return simulation_single_step_logic(current_loop_state, env, cpg)

    # Define the loop condition
    def loop_cond(current_loop_state):
        return (~current_loop_state["terminated"]
                & ~current_loop_state["truncated"]
                & (current_loop_state["steps_taken"] < MAX_STEPS_PER_EPISODE))

    # Run the simulation loop
    final_state = jax.lax.while_loop(loop_cond, body_step, loop_state)

    # Calculate final reward
    distance_after = final_state["current_distance"]
    distance_before = final_state["distance_before_loop"]
    improvement = distance_before - distance_after
    target_reached_bonus = jnp.where(distance_after < 0.1, 10.0, 0.0) # Example bonus
    reward = improvement + target_reached_bonus

    return reward


# --- Main Test Function ---
if __name__ == "__main__":
    print("Running basic test...")
    key = jax.random.PRNGKey(42)
    key_init, key_episode = jax.random.split(key)

    # Create environment and CPG instance
    env = create_environment()
    cpg_instance = CPG(dt=CONTROL_TIMESTEP)
    print("Environment and CPG created.")

    # Initialize state
    target_pos = DEFAULT_TARGET_POSITION
    initial_sim_state = init_simulation_state_logic(key_init, env, cpg_instance, target_pos)
    print(f"Initial state created. Initial distance: {initial_sim_state['initial_distance']:.4f}")

    # Define CPG parameters (adjust size/values based on your CPG)
    num_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1
    dummy_cpg_params = jnp.ones(num_params) * 0.5
    dummy_cpg_params = dummy_cpg_params.at[-1].set(jnp.pi) # Set omega
    print(f"Using {len(dummy_cpg_params)} dummy CPG parameters.")

    # Get max joint limit from environment spec
    max_joint_limit = env.morphology.specification.actuators[0].limits_config.pos[1]
    print(f"Using max joint limit: {max_joint_limit:.4f}")

    # Run episode
    print("Running episode...")
    final_reward = run_episode_logic(
        initial_state=initial_sim_state,
        cpg_params=dummy_cpg_params,
        env=env,
        cpg=cpg_instance,
        max_joint_limit=max_joint_limit
    )

    print(f"Episode finished. Final Reward: {final_reward:.4f}")

