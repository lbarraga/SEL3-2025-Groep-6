import typing
from functools import partial

import jax
import jax.numpy as jnp

from groep6.SimulationState import SimulationState, create_initial_simulation_state
from groep6.config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM, MAX_STEPS_PER_EPISODE, create_environment,
    CONTROL_TIMESTEP, CLOSE_ENOUGH_DISTANCE, MAXIMUM_TIME_BONUS, TARGET_REACHED_BONUS, TARGET_SAMPLING_RADIUS,
    FIXED_OMEGA, NUM_EVALUATIONS_PER_INDIVIDUAL, NUM_INFERENCES_PER_TRIAL# Add new imports
)
from groep6.cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions, CPGState
from groep6.nn import CPGController


def sample_random_target_pos(rng_single):
    """Samples a random target position on the circle perimeter."""
    angle = jax.random.uniform(rng_single, minval=0, maxval=2 * jnp.pi)
    radius = TARGET_SAMPLING_RADIUS
    target_pos = jnp.array([radius * jnp.cos(angle), radius * jnp.sin(angle), 0.0])
    return target_pos


def calculate_relative_direction(state: SimulationState) -> jnp.ndarray:
    """Calculates the normalized direction vector from the origin to the target position."""
    dir_to_target = state.env_state.observations["unit_xy_direction_to_target"]
    angle_to_target = jnp.atan2(dir_to_target[1], dir_to_target[0])
    disk_rotation = state.env_state.observations["disk_rotation"][2]  # rotation around z-axis
    return disk_rotation - angle_to_target


def get_joint_positions(env: SimulationState):
    """Extracts joint positions from the environment state."""
    return env.env_state.observations["joint_position"]


class EpisodeEvaluator:
    """Encapsulates environment, CPG, and logic for evaluating episodes."""

    def __init__(self):
        """Initializes the evaluator, environment, CPG, and JITted functions."""
        self.env = create_environment()
        self.cpg = CPG(dt=CONTROL_TIMESTEP)
        self.max_joint_limit = float(self.env.action_space.high[0]) * 0.5

        # Store JIT-compiled versions of env/cpg methods
        self._jit_env_reset = jax.jit(self.env.reset)
        self._jit_env_step = jax.jit(self.env.step)
        self._jit_cpg_reset = jax.jit(self.cpg.reset)
        self._jit_cpg_step = jax.jit(self.cpg.step)

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

        new_state = state.replace(
            env_state=new_env_state,
            cpg_state=new_cpg_state,
            steps_taken=steps_taken,
            best_distance=best_distance,
            current_distance=current_distance,
            last_progress_step=last_progress_step,
            terminated=terminated,
            truncated=new_env_state.truncated,
        )
        return new_state

    @partial(jax.jit, static_argnames=['self', 'model_obj'])
    def run_single_trial(self,
                         rng: jnp.ndarray,
                         initial_target_pos: jnp.ndarray,
                         model_obj: CPGController,
                         model_params_single: typing.Any
                         ) -> typing.Tuple[jnp.ndarray, SimulationState]:
        """Runs a single trial with multiple CPG inferences, using global constants."""
        initial_state = self.create_initial_state(rng=rng, target_pos=initial_target_pos)

        def inference_loop_cond_fn(loop_vars):
            current_sim_state, num_inferences_done = loop_vars
            return ((num_inferences_done < NUM_INFERENCES_PER_TRIAL) &
                    ~current_sim_state.terminated &
                    ~current_sim_state.truncated)

        def inference_loop_body_fn(loop_vars):
            initial_state_at_inference, num_inferences_done = loop_vars

            # inputs to neural network
            direction = jnp.array([calculate_relative_direction(initial_state_at_inference)])
            joint_positions = get_joint_positions(initial_state_at_inference)
            nn_input = jnp.concatenate([direction, joint_positions])

            generated_rx_params = model_obj.apply({'params': model_params_single}, nn_input)
            cpg_params = jnp.concatenate([generated_rx_params, jnp.array([FIXED_OMEGA])])

            modulated_cpg_state = self.modulate_cpg(initial_state_at_inference.cpg_state, cpg_params)
            sim_state_after_modulation = initial_state_at_inference.replace(cpg_state=modulated_cpg_state)

            initial_inner_loop_state = sim_state_after_modulation

            target_steps_for_this_inference_period = sim_state_after_modulation.steps_taken + (MAX_STEPS_PER_EPISODE / NUM_INFERENCES_PER_TRIAL)

            def inner_sim_loop_cond_fn(inner_loop_sim_state: SimulationState) -> bool:
                return ((~inner_loop_sim_state.terminated &
                         ~inner_loop_sim_state.truncated) &
                        (inner_loop_sim_state.steps_taken < target_steps_for_this_inference_period))

            sim_state_after_inference_period = jax.lax.while_loop(
                inner_sim_loop_cond_fn,
                self.simulation_single_step_logic,
                initial_inner_loop_state
            )

            return sim_state_after_inference_period, num_inferences_done + 1

        loop_vars_init = (initial_state, 0)

        final_loop_vars = jax.lax.while_loop(inference_loop_cond_fn, inference_loop_body_fn, loop_vars_init)
        final_state_overall, num_inferences_done = final_loop_vars
        final_state_overall = final_state_overall.replace(n_inferences=num_inferences_done)

        reward = calculate_final_reward(initial_state, final_state_overall)
        return reward, final_state_overall

    @partial(jax.jit, static_argnames=['self', 'model_obj', 'unravel_fn_single', 'num_evaluations'])
    def evaluate_network_multiple_trials(
            self,
            rng: jnp.ndarray,
            flat_model_params: jnp.ndarray,
            model_obj: CPGController,
            unravel_fn_single: typing.Callable,
            num_evaluations: int
    ) -> typing.Tuple[jnp.ndarray, typing.Any]:
        """Evaluates a network over k trials with random targets and multiple inferences per trial."""
        model_params_single = unravel_fn_single(flat_model_params)

        def run_one_evaluation(carry, rng_single_eval):
            rng_target, rng_trial = jax.random.split(rng_single_eval)
            target_pos = sample_random_target_pos(rng_target)

            reward, final_state = self.run_single_trial(rng_trial, target_pos, model_obj, model_params_single)
            return carry, (reward, final_state)

        rng_eval_split = jax.random.split(rng, num_evaluations)
        _, (rewards, final_states) = jax.lax.scan(run_one_evaluation, None, rng_eval_split)
        mean_reward = jnp.mean(rewards)
        return mean_reward, final_states

    @partial(jax.jit, static_argnames=['self'])
    def create_initial_state(self, rng: jnp.ndarray, target_pos: jnp.ndarray) -> SimulationState:
        rng_env, rng_cpg = jax.random.split(rng)
        initial_env_state = self._jit_env_reset(rng=rng_env, target_position=target_pos)
        initial_cpg_state = self._jit_cpg_reset(rng=rng_cpg)

        return create_initial_simulation_state(initial_env_state, initial_cpg_state)

    @partial(jax.jit, static_argnames=['self'])
    def modulate_cpg(self, cpg_state: CPGState, parameters: jnp.ndarray) -> CPGState:
        """Modulates the CPG state with given parameters."""
        new_R = parameters[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
        new_X = parameters[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:-1]
        new_omega = parameters[-1]

        return modulate_cpg(cpg_state, new_R, new_X, new_omega, self.max_joint_limit)


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
    [jax.Array, jax.Array, CPGController],  # input: rng_batch, flat_model_params_batch, model_obj
    typing.Tuple[jax.Array, typing.Any]  # output: fitness (mean rewards), final_states_batch
]


# Update create_evaluation_fn
def create_evaluation_fn(model_obj: CPGController, unravel_fn: typing.Callable, num_evaluations: int) -> EvaluationFn:
    """
    Creates the JIT-compiled, vmapped evaluation function using the Evaluator class.
    Now takes the NN model object and unravel function directly.
    """
    evaluator = EpisodeEvaluator()
    # JIT + Vmap the new evaluation method
    # Pass model_obj and unravel_fn as static arguments to vmap/jit indirectly via partial
    # Pass num_evaluations as static argument
    evaluate_single_network_fn = partial(
        evaluator.evaluate_network_multiple_trials,
        model_obj=model_obj,
        unravel_fn_single=unravel_fn,
        num_evaluations=num_evaluations
    )
    # Vmap over rng and flat_model_params
    evaluate_batch_fn = jax.jit(jax.vmap(evaluate_single_network_fn, in_axes=(0, 0)))
    return evaluate_batch_fn
