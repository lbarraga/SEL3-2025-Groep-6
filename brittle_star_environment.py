from functools import partial
import typing

import jax
import jax.numpy as jnp
from flax import struct

from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM,
    MAX_STEPS_PER_EPISODE, NO_PROGRESS_THRESHOLD,
    create_environment, CONTROL_TIMESTEP, CLOSE_ENOUGH_DISTANCE, MAXIMUM_TIME_BONUS, TARGET_REACHED_BONUS
)
from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions


@struct.dataclass
class SimulationState:
    env_state: typing.Any
    cpg_state: typing.Any
    initial_distance: jnp.ndarray
    best_distance: jnp.ndarray
    current_distance: jnp.ndarray
    steps_taken: jnp.ndarray
    last_progress_step: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray


def create_initial_simulation_state(env_state, cpg_state):
    target_pos_3d = jnp.concatenate([env_state.info["xy_target_position"], jnp.array([0.0])])
    current_pos = env_state.observations["disk_position"]
    initial_distance = jnp.linalg.norm(current_pos - target_pos_3d)

    return SimulationState(
        env_state=env_state,
        cpg_state=cpg_state,
        initial_distance=initial_distance,
        best_distance=initial_distance,
        current_distance=initial_distance,
        steps_taken=jnp.array(0, dtype=jnp.int32),
        last_progress_step=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),
        truncated=jnp.array(False),
    )


@partial(jax.jit, static_argnames=['step_fn', 'cpg'])
def simulation_single_step_logic(state: SimulationState, step_fn, cpg: CPG) -> SimulationState:
    """Performs a single step of the simulation logic."""
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

    # Terminate if internal env terminates or if the distance to the target is small enough
    terminated = (current_distance < CLOSE_ENOUGH_DISTANCE) | new_env_state.terminated

    # Truncate if no progress has been made for a certain number of steps
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


@jax.jit
def calculate_final_reward(initial_distance: jnp.ndarray, final_state: SimulationState) -> jnp.ndarray:
    """Calculates the final reward for an episode based on performance."""
    current_distance = final_state.current_distance
    steps = final_state.steps_taken

    # 1. Improvement in distance to target
    improvement = initial_distance - current_distance

    # 2. Target reached bonus (only reached target)
    target_reached_bonus = jnp.where(current_distance < CLOSE_ENOUGH_DISTANCE, TARGET_REACHED_BONUS, 0.0)

    # 3. Time bonus (only reached target)
    steps_clipped = jnp.minimum(steps, MAX_STEPS_PER_EPISODE)
    time_bonus_factor = 1.0 - (steps_clipped / MAX_STEPS_PER_EPISODE)
    time_bonus = MAXIMUM_TIME_BONUS * time_bonus_factor
    time_bonus = jnp.where(current_distance < CLOSE_ENOUGH_DISTANCE, time_bonus, 0.0)

    return improvement + target_reached_bonus + time_bonus


@partial(jax.jit, static_argnames=['step_fn', 'cpg', 'max_joint_limit'])
def run_episode_logic(initial_state: SimulationState, cpg_params: jnp.ndarray, step_fn, cpg: CPG,
                      max_joint_limit: float) -> typing.Tuple[jnp.ndarray, SimulationState]:
    """Runs a full episode simulation and calculates the final reward."""
    cpg_params = jnp.asarray(cpg_params)

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

    def body_step(current_loop_state: SimulationState) -> SimulationState:
        return simulation_single_step_logic(current_loop_state, step_fn, cpg)

    def loop_cond(current_loop_state: SimulationState) -> bool:
        return (~current_loop_state.terminated
                & ~current_loop_state.truncated
                & (current_loop_state.steps_taken < MAX_STEPS_PER_EPISODE))

    final_state = jax.lax.while_loop(loop_cond, body_step, loop_state)
    reward = calculate_final_reward(initial_state.initial_distance, final_state)

    return reward, final_state


def evaluate_single_solution(rng, cpg_params, target_pos, jit_reset, cpg_reset_fn, assemble_fn, run_episode_fn):
    rng_env, rng_cpg = jax.random.split(rng)

    initial_env_state = jit_reset(rng=rng_env, target_position=target_pos)
    initial_cpg_state = cpg_reset_fn(rng=rng_cpg)
    initial_sim_state = assemble_fn(initial_env_state, initial_cpg_state)

    reward, final_state = run_episode_fn(initial_sim_state, cpg_params)

    return reward, final_state


def create_evaluation_fn():
    """Creates the batched evaluation function for use with ES."""
    env = create_environment()
    cpg_instance = CPG(dt=CONTROL_TIMESTEP)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    max_joint_limit = float(env.action_space.high[0])
    cpg_reset_fn = cpg_instance.reset

    run_episode_partial = partial(
        run_episode_logic,
        step_fn=jit_step,
        cpg=cpg_instance,
        max_joint_limit=max_joint_limit
    )

    evaluate_single_partial = partial(
        evaluate_single_solution,
        jit_reset=jit_reset,
        cpg_reset_fn=cpg_reset_fn,
        assemble_fn=create_initial_simulation_state,
        run_episode_fn=run_episode_partial
    )

    evaluate_batch_fn = jax.jit(jax.vmap(evaluate_single_partial, in_axes=(0, 0, 0)))

    return evaluate_batch_fn
