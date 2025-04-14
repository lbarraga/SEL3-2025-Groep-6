import typing
from functools import partial

import jax
import jax.numpy as jnp

from SimulationState import SimulationState, create_initial_simulation_state
from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM,
    MAX_STEPS_PER_EPISODE, NO_PROGRESS_THRESHOLD,
    create_environment, CONTROL_TIMESTEP, CLOSE_ENOUGH_DISTANCE, MAXIMUM_TIME_BONUS, TARGET_REACHED_BONUS
)
from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions

class EpisodeEvaluator:
    """Encapsulates environment, CPG, and logic for evaluating episodes."""

    def __init__(self):
        """Initializes the evaluator, environment, CPG, and JITted functions."""
        self.env = create_environment()
        self.cpg = CPG(dt=CONTROL_TIMESTEP)
        self.max_joint_limit = float(self.env.action_space.high[0] * 0.5)

        # Store JIT-compiled versions of env/cpg methods
        self._jit_env_reset = jax.jit(self.env.reset)
        self._jit_env_step = jax.jit(self.env.step)
        self._jit_cpg_reset = jax.jit(self.cpg.reset)
        self._jit_cpg_step = jax.jit(self.cpg.step) # Step is now also JITted here

    @partial(jax.jit, static_argnames=['self'])
    def simulation_single_step_logic(self, state: SimulationState) -> SimulationState:
        """Performs a single step of the simulation logic."""
        new_cpg_state = self._jit_cpg_step(state=state.cpg_state)

        motor_actions = map_cpg_outputs_to_actions(
            cpg_state=new_cpg_state,
            num_arms=NUM_ARMS,
            num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
            num_oscillators_per_arm=NUM_OSCILLATORS_PER_ARM
        )

        new_env_state = self._jit_env_step(state=state.env_state, action=motor_actions)
        steps_taken = state.steps_taken + 1

        current_position = new_env_state.observations["disk_position"]
        target_position = jnp.concatenate([new_env_state.info["xy_target_position"], jnp.array([0.0])])
        current_distance = jnp.linalg.norm(current_position - target_position)

        progress_made = current_distance < state.best_distance
        last_progress_step = jnp.where(progress_made, steps_taken, state.last_progress_step)

        best_distance = jnp.minimum(state.best_distance, current_distance)

        # Terminate if internal env terminates or if the distance to the target is small enough
        terminated = (current_distance < CLOSE_ENOUGH_DISTANCE) | new_env_state.terminated

        # Truncate if no progress has been made for a certain number of steps
        truncated = new_env_state.truncated | ((steps_taken - last_progress_step) > NO_PROGRESS_THRESHOLD)

        new_state = state.replace(
            env_state=new_env_state,
            cpg_state=new_cpg_state,
            steps_taken=steps_taken,
            best_distance=best_distance,
            current_distance=current_distance,
            last_progress_step=last_progress_step,
            terminated=terminated,
            truncated=truncated,
        )
        return new_state


    @partial(jax.jit, static_argnames=['self'])
    def run_episode_logic(self, rng: jnp.ndarray, cpg_params: jnp.ndarray, target_pos: jnp.ndarray) -> typing.Tuple[jnp.ndarray, SimulationState]:
        """Runs a full episode simulation and calculates the final reward."""

        rng_env, rng_cpg = jax.random.split(rng)
        initial_env_state = self._jit_env_reset(rng=rng_env, target_position=target_pos)
        initial_cpg_state = self._jit_cpg_reset(rng=rng_cpg)

        initial_state = create_initial_simulation_state(initial_env_state, initial_cpg_state)

        cpg_params = jnp.asarray(cpg_params)

        new_R = cpg_params[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
        new_X = cpg_params[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:-1]
        new_omega = cpg_params[-1]

        modulated_cpg_state = modulate_cpg(
            cpg_state=initial_state.cpg_state,
            new_R=new_R,
            new_X=new_X,
            new_omega=new_omega,
            max_joint_limit=self.max_joint_limit
        )

        loop_state = initial_state.replace(cpg_state=modulated_cpg_state)

        def body_step(current_loop_state: SimulationState) -> SimulationState:
            return self.simulation_single_step_logic(current_loop_state)

        def loop_cond(current_loop_state: SimulationState) -> bool:
            return (~current_loop_state.terminated
                    & ~current_loop_state.truncated
                    & (current_loop_state.steps_taken < MAX_STEPS_PER_EPISODE))

        final_state = jax.lax.while_loop(loop_cond, body_step, loop_state)
        reward = calculate_final_reward(initial_state, final_state)

        return reward, final_state


def calculate_final_reward(initial_state: SimulationState, final_state: SimulationState) -> jnp.ndarray:
    """Calculates the final reward for an episode based on performance."""
    initial_distance = initial_state.current_distance
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

EvaluationFn = typing.Callable[
    [jax.Array, jax.Array, jax.Array], # input: rng_batch, cpg_params, target_pos
    typing.Tuple[jax.Array, jax.Array] # output: fitness, final_states
]

def create_evaluation_fn() -> EvaluationFn:
    """
    Creates the JIT-compiled, vmapped evaluation function using the Evaluator class.
    This non-vmapped function takes a random key, CPG parameters, and target position,
    and returns the fitness and final states.
    """
    evaluator = EpisodeEvaluator()
    # JIT + Vmap the method for batch evaluation
    evaluate_batch_fn = jax.jit(jax.vmap(evaluator.run_episode_logic, in_axes=(0, 0, 0)))
    return evaluate_batch_fn
