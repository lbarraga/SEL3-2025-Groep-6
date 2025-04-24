from functools import partial
from typing import Tuple, Dict

import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
from gymnasium import spaces
from jax import Array

from config import (
    NUM_ARMS, NUM_SEGMENTS_PER_ARM, NUM_OSCILLATORS_PER_ARM, CONTROL_TIMESTEP,
    create_environment, CLOSE_ENOUGH_DISTANCE, MAX_STEPS_PER_EPISODE, NO_PROGRESS_THRESHOLD, MAXIMUM_TIME_BONUS,
    TARGET_REACHED_BONUS
)
from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions, CPGState
from SimulationState import SimulationState, create_initial_simulation_state

class BrittleStarGymEnv(gym.Env):
    """
    Gymnasium-compatible environment for the Brittle Star CPG-based robot.
    Action: CPG parameters (flat array for modulate_cpg)
    Observation: Environment observations (disk position, etc)
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": int(1 / CONTROL_TIMESTEP)}

    def __init__(self, seed=0):
        super().__init__()
        self.env = create_environment()
        self.cpg = CPG(dt=CONTROL_TIMESTEP)
        self.max_joint_limit = float(self.env.action_space.high[0]) * 0.5

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(NUM_ARMS * NUM_OSCILLATORS_PER_ARM * 2 + 1,),
            dtype=np.float64
        )

        # TODO
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(6,),
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

        self._current_obs = None
        self._target_pos = None
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._info = {}

        self._initialize()

    @partial(jax.jit, static_argnames=['self'])
    def _initialize(self, target_pos: jnp.ndarray):
        self._rng, rng = jax.random.split(self._rng)

        self._env_state = self._jit_env_reset(rng=rng, target_position=target_pos)
        self._cpg_state = self._jit_cpg_reset(rng=rng)

        self._sim_state = create_initial_simulation_state(self._env_state, self._cpg_state)

    @partial(jax.jit, static_argnames=['self'])
    def modulate_cpg(self, cpg_state: CPGState, parameters: jnp.ndarray):
        """Modulates the CPG state with given parameters."""
        new_R = parameters[:NUM_ARMS * NUM_OSCILLATORS_PER_ARM]
        new_X = parameters[NUM_ARMS * NUM_OSCILLATORS_PER_ARM:-1]
        new_omega = parameters[-1]

        self._cpg_state = modulate_cpg(cpg_state, new_R, new_X, new_omega, self.max_joint_limit)

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

    def _get_reward(self, initial_state: SimulationState, final_state: SimulationState) -> float:
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


    def _get_observation(self):
        # TODO
        return jnp.ndarray(0)

    def reset(self, seed=None, options=None) -> tuple[Array, dict[str, float]]:
        if seed is not None:
            self.seed = seed
            self._rng = jax.random.PRNGKey(seed=seed)

        self._initialize()

        observation = self._get_observation()
        info = {
            "distance_to_target": float(jnp.linalg.norm(
                self.get_brittle_star_position() - self.get_target_position()
            ))
        }

        return observation, info

    def step(self, action):
        pass

    def render(self):
        from render import post_render

        frame = self.env.render(state=self._env_state)
        processed_frame = post_render(frame, self.env.environment_configuration)

        return np.array(processed_frame)

    def close(self):
        pass