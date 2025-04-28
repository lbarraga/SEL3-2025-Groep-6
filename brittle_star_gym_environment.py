from functools import partial
from typing import Tuple, Dict

import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces
from jax import Array

from brittle_star_environment import EpisodeEvaluator
from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM, CONTROL_TIMESTEP,
    create_environment, CLOSE_ENOUGH_DISTANCE, MAX_STEPS_PER_PPO_EPISODE, NO_PROGRESS_THRESHOLD, MAXIMUM_TIME_BONUS,
    TARGET_REACHED_BONUS
)
from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions, CPGState
from SimulationState import SimulationState, create_initial_simulation_state
from util import calculate_direction, sample_random_target_pos


class BrittleStarGymEnv(gym.Env):
    """
    Gymnasium-compatible environment for the Brittle Star CPG-based robot.
    Action: CPG parameters (flat array for modulate_cpg)
    Observation: Environment observations (target position, direction (disk position?, joint positions?))
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_TIMESTEP)}

    def __init__(self, seed=0):
        self.env = create_environment()
        self.cpg = CPG(dt=CONTROL_TIMESTEP)
        self.max_joint_limit = float(self.env.action_space.high[0]) * 0.5

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1,),
            dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(2,),
            dtype=np.float64
        )

        # Store JIT-compiled versions of env/cpg methods
        self._jit_env_reset = jax.jit(self.env.reset)
        self._jit_env_step = jax.jit(self.env.step)
        self._jit_cpg_reset = jax.jit(self.cpg.reset)
        self._jit_cpg_step = jax.jit(self.cpg.step) # Step is now also JITted here

        # Initialize the environment
        self.seed = seed
        self._rng = jax.random.PRNGKey(seed=seed)
        self._env_state = None
        self._cpg_state = None
        self._sim_state = None

        self.brittle_star_env = EpisodeEvaluator()

        self._initialize()

    # @partial(jax.jit, static_argnames=['self'])
    def _initialize(self):
        self._rng, rng = jax.random.split(self._rng)
        target_pos = sample_random_target_pos(rng)
        self._env_state = self._jit_env_reset(rng=rng, target_position=target_pos)
        self._cpg_state = self._jit_cpg_reset(rng=rng)

        self._sim_state = create_initial_simulation_state(self._env_state, self._cpg_state)

    @partial(jax.jit, static_argnames=['self'])
    def modulate_cpg(self, cpg_state: CPGState, parameters: jnp.ndarray):
        """Modulates the CPG state with given parameters."""
        new_R = parameters[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
        new_X = parameters[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:-1]
        new_omega = parameters[-1]

        return modulate_cpg(cpg_state, new_R, new_X, new_omega, self.max_joint_limit)

    def get_brittle_star_position(self):
        return self._env_state.observations["disk_position"]

    def get_target_position(self):
        return jnp.concatenate([self._env_state.info["xy_target_position"], jnp.array([0.0])])

    def get_disk_rotation(self) -> float:
        return self._env_state.observations["disk_rotation"][-1]

    def get_joint_positions(self) -> jnp.array:
        return self._env_state.observations["joint_positions"]

    def get_direction_to_target(self) -> jnp.array:
        return self._env_state.observations["unit_xy_direction_to_target"]

    @partial(jax.jit, static_argnames=['self'])
    def _get_reward(self, initial_state: SimulationState, final_state: SimulationState) -> float:
        initial_distance = initial_state.current_distance
        current_distance = final_state.current_distance

        # 1. Improvement in distance to target
        improvement = initial_distance - current_distance

        # 2. Target reached bonus (only reached target)
        target_reached_bonus = jnp.where(current_distance < CLOSE_ENOUGH_DISTANCE, TARGET_REACHED_BONUS, 0.0)

        return improvement + target_reached_bonus


    def _get_observation(self):
        return calculate_direction(self.get_target_position() - self.get_brittle_star_position())

    def _get_info(self):
        return {
            "distance_to_target": float(jnp.linalg.norm(
                self.get_brittle_star_position() - self.get_target_position()
            )),
        }

    def reset(self, seed=None, options=None) -> tuple[Array, dict[str, float]]:
        if seed is not None:
            self.seed = seed
            self._rng = jax.random.PRNGKey(seed=seed)

        self._initialize()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._cpg_state, self._sim_state, reward = self._pure_step(self._cpg_state, self._sim_state, action)
        self._env_state = self._sim_state.env_state
        observation = self._get_observation()
        terminated = self._sim_state.terminated
        truncated = self._sim_state.truncated
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    @partial(jax.jit, static_argnames=['self'])
    def _pure_step(self, cpg_state, sim_state, action):
        """Pure step function for JIT compilation."""
        cpg_state = self.modulate_cpg(cpg_state, jnp.asarray(action))
        loop_state = sim_state.replace(cpg_state=cpg_state)

        def loop_cond(current_loop_state: SimulationState) -> bool:
            return (~current_loop_state.terminated
                    & ~current_loop_state.truncated
                    & (current_loop_state.steps_taken < MAX_STEPS_PER_PPO_EPISODE))

        prev_state = sim_state
        sim_state = jax.lax.while_loop(loop_cond, self.brittle_star_env.simulation_single_step_logic, loop_state)
        reward = self._get_reward(prev_state, self._sim_state)

        return cpg_state, sim_state, reward


    def render(self):
        from render import post_render

        frame = self.env.render(state=self._env_state)
        processed_frame = post_render(frame, self.env.environment_configuration)

        return np.array(processed_frame)

    def close(self):
        pass