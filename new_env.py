from functools import partial
import typing

import jax
import jax.numpy as jnp
from flax import struct

from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM,
    MAX_STEPS_PER_EPISODE, NO_PROGRESS_THRESHOLD,
    CONTROL_TIMESTEP, DEFAULT_TARGET_POSITION, create_environment,
)
from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions


@struct.dataclass
class SimulationState:
    env_state: typing.Any  # Use actual Env State type if known
    cpg_state: typing.Any  # Use actual CPG State type if known
    initial_distance: jnp.ndarray
    best_distance: jnp.ndarray
    current_distance: jnp.ndarray
    steps_taken: jnp.ndarray
    last_progress_step: jnp.ndarray
    terminated: jnp.ndarray  # Should be bool array
    truncated: jnp.ndarray  # Should be bool array
    reward: jnp.ndarray  # Keep for consistency, though calculated at end


@partial(jax.jit, static_argnames=['step_fn', 'cpg'])
def simulation_single_step_logic(state: SimulationState, step_fn, cpg: CPG) -> SimulationState:
    new_cpg_state = cpg.step(state=state.cpg_state)

    motor_actions = map_cpg_outputs_to_actions(
        cpg_state=new_cpg_state,
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        num_oscillators_per_arm=NUM_OSCILLATORS_PER_ARM
    )

    new_env_state = step_fn(state=state.env_state, action=motor_actions)

    current_position = new_env_state.observations["disk_position"]
    target_position = jnp.concatenate([new_env_state.info["xy_target_position"], jnp.array([0.0])])
    current_distance = jnp.linalg.norm(current_position - target_position)

    progress_made = current_distance < state.best_distance
    current_step_index = state.steps_taken + 1
    last_progress_step = jnp.where(progress_made, current_step_index, state.last_progress_step)
    best_distance = jnp.minimum(state.best_distance, current_distance)

    terminated = (current_distance < 0.1) | new_env_state.terminated
    truncated = new_env_state.truncated | ((current_step_index - last_progress_step) > NO_PROGRESS_THRESHOLD)

    new_state = state.replace(
        env_state=new_env_state,
        cpg_state=new_cpg_state,
        steps_taken=current_step_index,
        best_distance=best_distance,
        current_distance=current_distance,
        last_progress_step=last_progress_step,
        terminated=terminated,
        truncated=truncated,
    )
    return new_state


@partial(jax.jit, static_argnames=['step_fn', 'cpg', 'max_joint_limit'])
def run_episode_logic(initial_state: SimulationState, cpg_params: jnp.ndarray, step_fn, cpg: CPG,
                      max_joint_limit: float) -> jnp.ndarray:
    cpg_params = jnp.asarray(cpg_params)
    num_cpg_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1  # Example
    if len(cpg_params) != num_cpg_params:
        # Consider validating parameter shapes outside JIT if possible
        raise ValueError(f"Expected {num_cpg_params} CPG parameters, got {len(cpg_params)}")

    new_R = cpg_params[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
    new_X = cpg_params[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:-1]
    new_omega = cpg_params[-1]

    modulated_cpg_state = modulate_cpg(
        cpg_state=initial_state.cpg_state,
        new_R=new_R,
        new_X=new_X,
        new_omega=new_omega,
        max_joint_limit=max_joint_limit
    )

    loop_state = initial_state.replace(cpg_state=modulated_cpg_state)
    distance_before_loop = initial_state.initial_distance  # Access directly

    def body_step(current_loop_state: SimulationState) -> SimulationState:
        return simulation_single_step_logic(current_loop_state, step_fn, cpg)

    def loop_cond(current_loop_state: SimulationState) -> bool:
        return (~current_loop_state.terminated
                & ~current_loop_state.truncated
                & (current_loop_state.steps_taken < MAX_STEPS_PER_EPISODE))

    final_state = jax.lax.while_loop(loop_cond, body_step, loop_state)

    distance_after = final_state.current_distance
    improvement = distance_before_loop - distance_after  # Use stored value
    target_reached_bonus = jnp.where(distance_after < 0.1, 10.0, 0.0)  # Example bonus
    reward = improvement + target_reached_bonus

    return reward


# --- Main Test Function ---
if __name__ == "__main__":
    print("Running basic test...")
    key = jax.random.PRNGKey(42)
    key_init_env, key_init_cpg, key_episode = jax.random.split(key, 3)

    # --- Setup ---
    env = create_environment()
    cpg_instance = CPG(dt=CONTROL_TIMESTEP)
    print("Environment and CPG created.")

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    print("JITted env.reset and env.step created.")

    # --- Initialization ---
    initial_env_state = jit_reset(rng=key_init_env, target_position=DEFAULT_TARGET_POSITION)
    print("Environment reset.")

    initial_cpg_state = cpg_instance.reset(rng=key_init_cpg)
    print("CPG reset.")

    current_pos = initial_env_state.observations["disk_position"]
    target_pos_3d = jnp.concatenate([initial_env_state.info["xy_target_position"], jnp.array([0.0])])
    initial_distance = jnp.linalg.norm(current_pos - target_pos_3d)
    print(f"Initial distance calculated: {initial_distance:.4f}")

    # Instantiate the SimulationState class instead of creating a dictionary
    initial_sim_state = SimulationState(
        env_state=initial_env_state,
        cpg_state=initial_cpg_state,
        initial_distance=initial_distance,
        best_distance=initial_distance,
        current_distance=initial_distance,
        steps_taken=jnp.array(0, dtype=jnp.int32),  # Ensure appropriate dtype
        last_progress_step=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),  # JAX bool array
        truncated=jnp.array(False),  # JAX bool array
        reward=jnp.array(0.0, dtype=jnp.float32)
    )
    print("Initial SimulationState created.")

    # --- Episode Execution ---
    num_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1
    dummy_cpg_params = jnp.ones(num_params) * 0.5
    dummy_cpg_params = dummy_cpg_params.at[-1].set(jnp.pi)  # Set omega
    print(f"Using {len(dummy_cpg_params)} dummy CPG parameters.")

    max_joint_limit = env.action_space.high[0] * 0.25
    max_joint_limit = float(max_joint_limit)  # Ensure float for static arg if needed
    print(f"Using max joint limit for CPG modulation: {max_joint_limit:.4f}")

    print("Running episode...")
    # Pass the SimulationState instance
    final_reward = run_episode_logic(
        initial_state=initial_sim_state,
        cpg_params=dummy_cpg_params,
        step_fn=jit_step,
        cpg=cpg_instance,
        max_joint_limit=max_joint_limit
    )

    print(f"Episode finished. Final Reward: {final_reward:.4f}")
