from functools import partial
import typing

import jax
import jax.numpy as jnp
from flax import struct

from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM,
    MAX_STEPS_PER_EPISODE, NO_PROGRESS_THRESHOLD,
    create_environment, DEFAULT_TARGET_POSITION, CONTROL_TIMESTEP
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
    reward: jnp.ndarray # Note: This field in the state might become redundant


def create_initial_simulation_state(env_state, cpg_state):
    current_pos = env_state.observations["disk_position"]
    target_pos_3d = jnp.concatenate([env_state.info["xy_target_position"], jnp.array([0.0])])
    initial_distance = jnp.linalg.norm(current_pos - target_pos_3d)
    return SimulationState(
        env_state=env_state, cpg_state=cpg_state,
        initial_distance=initial_distance, best_distance=initial_distance,
        current_distance=initial_distance, steps_taken=jnp.array(0, dtype=jnp.int32),
        last_progress_step=jnp.array(0, dtype=jnp.int32), terminated=jnp.array(False),
        truncated=jnp.array(False), reward=jnp.array(0.0, dtype=jnp.float32)
    )


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

    # Update reward field in state just before returning if desired, or remove it
    final_reward_calc = state.initial_distance - current_distance + jnp.where(current_distance < 0.1, 10.0, 0.0)

    new_state = state.replace(
        env_state=new_env_state,
        cpg_state=new_cpg_state,
        steps_taken=current_step_index,
        best_distance=best_distance,
        current_distance=current_distance,
        last_progress_step=last_progress_step,
        terminated=terminated,
        truncated=truncated,
        reward=final_reward_calc # Update reward in state
    )
    return new_state


@partial(jax.jit, static_argnames=['step_fn', 'cpg', 'max_joint_limit'])
def run_episode_logic(initial_state: SimulationState, cpg_params: jnp.ndarray, step_fn, cpg: CPG,
                      max_joint_limit: float) -> typing.Tuple[jnp.ndarray, SimulationState]: # Return type changed
    cpg_params = jnp.asarray(cpg_params)
    num_cpg_params = NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1

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
    distance_before_loop = initial_state.initial_distance

    def body_step(current_loop_state: SimulationState) -> SimulationState:
        return simulation_single_step_logic(current_loop_state, step_fn, cpg)

    def loop_cond(current_loop_state: SimulationState) -> bool:
        return (~current_loop_state.terminated
                & ~current_loop_state.truncated
                & (current_loop_state.steps_taken < MAX_STEPS_PER_EPISODE))

    final_state = jax.lax.while_loop(loop_cond, body_step, loop_state)

    distance_after = final_state.current_distance
    improvement = distance_before_loop - distance_after
    target_reached_bonus = jnp.where(distance_after < 0.1, 10.0, 0.0)
    reward = improvement + target_reached_bonus

    return reward, final_state


@partial(jax.jit, static_argnames=(
    'jit_reset', 'cpg_reset_fn', 'assemble_fn', 'run_episode_fn'
))
def evaluate_single_solution(rng, params, jit_reset, cpg_reset_fn, assemble_fn, run_episode_fn, target_pos):
    rng_env, rng_cpg = jax.random.split(rng)
    initial_env_state = jit_reset(rng=rng_env, target_position=target_pos)
    initial_cpg_state = cpg_reset_fn(rng=rng_cpg)
    initial_sim_state = assemble_fn(initial_env_state, initial_cpg_state)

    reward, final_state = run_episode_fn(initial_sim_state, params)
    return reward, final_state


def create_evaluation_fn():
    env = create_environment()
    cpg_instance = CPG(dt=CONTROL_TIMESTEP)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    max_joint_limit = float(env.action_space.high[0])
    cpg_reset_fn = cpg_instance.reset
    target_pos = DEFAULT_TARGET_POSITION

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
        run_episode_fn=run_episode_partial,
        target_pos=target_pos
    )

    # vmap will now map over the function returning (reward, final_state)
    # The result will be a tuple: (array_of_rewards, pytree_of_states)
    evaluate_batch_fn = jax.jit(jax.vmap(evaluate_single_partial, in_axes=(0, 0)))
    return evaluate_batch_fn

