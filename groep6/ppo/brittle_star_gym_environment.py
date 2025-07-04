import math
from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from jax import Array

from groep6.defaults import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM, CONTROL_TIMESTEP,
    create_environment, CLOSE_ENOUGH_DISTANCE, MAX_STEPS_PER_PPO_EPISODE,
    TARGET_REACHED_BONUS, FIXED_OMEGA
)
from groep6.cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions, CPGState
from groep6.SimulationState import SimulationState, create_initial_simulation_state
from groep6.util import calculate_direction, sample_random_target_pos, normalize_corner


class BrittleStarGymEnv(gym.Env):
    """
    Gymnasium-compatible environment for the Brittle Star CPG-based robot.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, seed: int = 0, target_position: jnp.ndarray = None):
        self.env = create_environment()
        self.cpg = CPG(dt=CONTROL_TIMESTEP)
        self.max_joint_limit = float(self.env.action_space.high[0] * 0.25)

        # Define action space with correct bounds (flattened version)
        # R values (10 values): -1 to 1
        # X values (10 values): 0 to max_joint_limit
        self.action_space = Box(
            low=np.concatenate([
                np.full(10, -1),
                np.full(10, 0)
            ]),
            high=np.concatenate([
                np.full(10, 1),
                np.full(10, self.max_joint_limit)
            ]),
            shape=(20,),
            dtype=jnp.float64
        )

        # Store JIT-compiled versions of env/cpg methods
        self._jit_env_reset = jax.jit(self.env.reset)
        self._jit_env_step = jax.jit(self.env.step)
        self._jit_cpg_reset = jax.jit(self.cpg.reset)
        self._jit_cpg_step = jax.jit(self.cpg.step)

        # Initialize the environment
        self.seed = seed
        self._rng = jax.random.PRNGKey(seed=seed)

        self.sim_state = None

        self._initialize(target_position)

        # Define observation space
        # disk rotation z, direction (x, y), joint positions, amplitudes
        joint_positions = self.get_joint_positions()
        len_jpos = len(joint_positions)

        amplitudes = self.sim_state.cpg_state.amplitudes
        len_amplitudes = len(amplitudes)

        # joint positions: -1.0 to 1.0
        # amplitudes: 0 to max_joint_limit
        self.observation_space = Box(
            low=np.concatenate([
                np.array([0.0, -math.inf, -math.inf]),
                np.full(len_jpos, -1),
                np.full(len_amplitudes, 0)
            ]),
            high=np.concatenate([
                np.array([2 * math.pi, math.inf, math.inf]),
                np.full(len_jpos, 1),
                np.full(len_amplitudes, self.max_joint_limit)
            ]),
            shape=(3 + len_jpos + len_amplitudes,),
            dtype=np.float64
        )

    def _initialize(self, target_position=None):
        """Initialize the simulation state"""
        self._rng, rng = jax.random.split(self._rng)

        if target_position is None:
            target_position = sample_random_target_pos(rng)

        env_state = self._jit_env_reset(rng=rng, target_position=target_position)
        cpg_state = self._jit_cpg_reset(rng=rng)

        self.sim_state = create_initial_simulation_state(env_state, cpg_state)

    @partial(jax.jit, static_argnames=['self'])
    def modulate_cpg(self, cpg_state: CPGState, parameters: jnp.ndarray, omega=FIXED_OMEGA) -> CPGState:
        """Modulates the CPG state with given parameters."""
        new_R = parameters[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
        new_X = parameters[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:]

        return modulate_cpg(cpg_state, new_R, new_X, omega, self.max_joint_limit)

    def get_brittle_star_position(self):
        return self.sim_state.env_state.observations["disk_position"]

    def get_target_position(self):
        return jnp.concatenate([self.sim_state.env_state.info["xy_target_position"], jnp.array([0.0])])

    def get_disk_rotation(self) -> float:
        """Get the current disk rotation on the z-axis."""
        return self.sim_state.env_state.observations["disk_rotation"][-1]

    def get_joint_positions(self) -> jnp.array:
        return self.sim_state.env_state.observations["joint_position"]

    def get_direction_to_target(self) -> jnp.array:
        return self.sim_state.env_state.observations["unit_xy_direction_to_target"]

    @partial(jax.jit, static_argnames=['self'])
    def _get_reward(self, initial_state: SimulationState, final_state: SimulationState) -> float:
        """Get the reward for the current state."""
        initial_distance = initial_state.current_distance
        current_distance = final_state.current_distance

        # 1. Improvement in distance to target
        improvement = initial_distance - current_distance

        # 2. Target reached bonus (only reached target)
        target_reached_bonus = jnp.where(current_distance < CLOSE_ENOUGH_DISTANCE, TARGET_REACHED_BONUS, 0.0)

        return improvement + target_reached_bonus

    def get_observation(self):
        """Get the current observations."""
        x, y = calculate_direction(self.get_target_position() - self.get_brittle_star_position())
        observation = [
            normalize_corner(self.get_disk_rotation()),
            x, y,
            *self.get_joint_positions(),
            *self.sim_state.cpg_state.amplitudes
        ]
        return observation

    def _get_info(self):
        return {
            "distance_to_target": float(jnp.linalg.norm(
                self.get_brittle_star_position() - self.get_target_position()
            )),
        }

    def reset(self, seed=None, options=None) -> tuple[Array, dict[str, float]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed = seed
            self._rng = jax.random.PRNGKey(seed=seed)

        self._initialize()

        observation = self.get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Step the environment with given action."""
        self.sim_state, reward = self._pure_step(self.sim_state, action)

        observation = self.get_observation()
        terminated = self.sim_state.terminated
        truncated = self.sim_state.truncated
        info = {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

        return observation, reward, terminated, truncated, info

    @partial(jax.jit, static_argnames=['self'])
    def _pure_step(self, sim_state: SimulationState, action):
        """Pure step function for JIT compilation."""
        cpg_state = self.modulate_cpg(sim_state.cpg_state, jnp.asarray(action))
        loop_state = sim_state.replace(cpg_state=cpg_state)

        def loop_cond(current_loop_state: SimulationState) -> bool:
            return (~current_loop_state.terminated
                    & ~current_loop_state.truncated
                    & (current_loop_state.steps_taken < MAX_STEPS_PER_PPO_EPISODE))

        prev_state = sim_state
        sim_state = jax.lax.while_loop(loop_cond, self.simulation_single_step_logic, loop_state)
        reward = self._get_reward(prev_state, self.sim_state)

        return sim_state, reward

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
        truncated = new_env_state.truncated

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

    def render(self):
        """Renders a single frame of the environment."""
        from groep6.render import post_render

        frame = self.env.render(state=self.sim_state.env_state)
        processed_frame = post_render(frame, self.env.environment_configuration)

        return jnp.array(processed_frame)

    def close(self):
        pass
