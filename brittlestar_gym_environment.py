from functools import partial
from typing import Dict, Tuple, Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

import config
from cpg import create_cpg, modulate_cpg, map_cpg_outputs_to_actions
from environment import create_directed_environment


class BrittleStarEnv(gym.Env):
    """
    Gym environment wrapper for the brittle star simulation.

    This environment integrates the JAX-based brittle star simulation with
    Gym's interface, enabling compatibility with RL libraries.
    """

    metadata = {"render_fps": 30}

    def __init__(self, target_position=(1.25, 0.75, 0.0), seed=0):
        """
        Initialize the brittle star environment.

        Args:
            target_position: The target position for the brittle star to reach.
            seed: Random seed for reproducibility.
        """
        self.target_position = target_position
        self.seed_value = seed

        # Create the JAX environment
        self.env = create_directed_environment(
            config.morphology_specification,
            config.arena_configuration,
            config.environment_configuration,
            "MJX"
        )

        # Create the CPG
        self.cpg = create_cpg()

        # JIT compile the environment step function
        self.env_step_fn = jax.jit(self.env.step)
        self.jit_reset = jax.jit(partial(self.env.reset, target_position=target_position))

        # Define action space with correct bounds (flattened version)
        # R values (8 values): -1.0 to 1.0
        # X values (8 values): -1.0 to 1.0
        # omega value (1 value): 0.0 to 5.0
        self.action_space = spaces.Box(
            low=np.concatenate([np.full(8, -1.0), np.full(8, -1.0), np.array([1.0])]),
            high=np.concatenate([np.full(8, 1.0), np.full(8, 1.0), np.array([5.0])]),
            shape=(17,),
            dtype=np.float64
        )

        # Observation space: brittle star position (3) and target position (3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

        # Initialize states
        self.rng = jax.random.PRNGKey(seed=seed)
        self.env_state = None
        self.cpg_state = None
        self._initialize_states()

    def _initialize_states(self):
        """Initialize the environment and CPG states."""
        self.rng, subkey1, subkey2 = jax.random.split(self.rng, 3)
        self.cpg_state = self.cpg.reset(rng=subkey1)
        self.env_state = self.jit_reset(rng=subkey2)

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation from the environment.

        Returns:
            numpy.ndarray: Current brittle star position and target position.
        """
        current_position = self._get_brittle_star_position()
        target_position = self._get_target_position()

        # Convert JAX arrays to NumPy for gym compatibility
        return np.concatenate([
            np.array(current_position),
            np.array(target_position)
        ])

    def _get_brittle_star_position(self):
        """Get the current position of the brittle star."""
        return self.env_state.observations["disk_position"]

    def _get_target_position(self):
        """Get the target position."""
        return jnp.concatenate([
            self.env_state.info["xy_target_position"],
            jnp.array([0.0])
        ])

    def _get_reward(self) -> float:
        """
        Calculate the reward based on distance to target and other factors.

        Returns:
            float: The reward value.
        """
        current_position = self._get_brittle_star_position()
        target_position = self._get_target_position()

        # Calculate distance to target
        distance = jnp.linalg.norm(current_position - target_position)

        # Base reward is negative distance (closer = higher reward)
        base_reward = -distance

        # Add a bonus for reaching the target
        target_reached_bonus = jnp.where(distance < 0.1, 10.0, 0.0)

        # Combine rewards
        reward = base_reward + target_reached_bonus

        return float(reward)

    def _get_terminated(self) -> bool:
        """
        Check if the episode should terminate.

        Returns:
            bool: True if the episode should terminate, False otherwise.
        """
        # Check if the brittle star has reached the target
        current_position = self._get_brittle_star_position()
        target_position = self._get_target_position()
        distance = jnp.linalg.norm(current_position - target_position)

        # Terminate if target reached or from the environment's terminated signal
        target_reached = distance < 0.1
        return bool(target_reached | self.env_state.terminated)

    def _get_truncated(self) -> bool:
        """
        Check if the episode should be truncated (e.g., max steps reached).

        Returns:
            bool: True if the episode should be truncated, False otherwise.
        """
        return bool(self.env_state.truncated)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed.
            options: Additional options for resetting.

        Returns:
            Tuple containing the initial observation and info dict.
        """
        if seed is not None:
            self.seed_value = seed
            self.rng = jax.random.PRNGKey(seed=seed)

        self._initialize_states()

        observation = self._get_observation()
        info = {
            "distance_to_target": float(jnp.linalg.norm(
                self._get_brittle_star_position() - self._get_target_position()
            ))
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.

        Args:
            action: Array of shape (17,) containing R (8), X (8), and omega (1) values.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert numpy action to JAX array
        action = jnp.array(action)

        # Extract CPG parameters from action
        new_R, new_X, new_omega = action[:8], action[8:16], action[16]

        # Modulate the CPG with the action
        self.cpg_state = modulate_cpg(
            cpg_state=self.cpg_state,
            new_R=new_R,
            new_X=new_X,
            new_omega=new_omega,
            max_joint_limit=self.env.action_space.high[0] * 0.25
        )

        # Step the CPG
        self.cpg_state = self.cpg.step(state=self.cpg_state)

        # Map CPG outputs to actions
        motor_actions = map_cpg_outputs_to_actions(cpg_state=self.cpg_state)

        # Step the environment
        self.env_state = self.env_step_fn(state=self.env_state, action=motor_actions)

        # Get observation, reward, and done signals
        observation = self._get_observation()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()

        # Additional info
        info = {
            "distance_to_target": float(jnp.linalg.norm(
                self._get_brittle_star_position() - self._get_target_position()
            )),
            "target_position": np.array(self._get_target_position()),
            "brittle_star_position": np.array(self._get_brittle_star_position())
        }

        if terminated:
            info["final_frame"] = self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        """
        Render the environment.

        Returns:
            numpy.ndarray: RGB array of the rendered frame.
        """
        from render import post_render

        # Render the environment and post-process the frame
        frame = self.env.render(state=self.env_state)
        processed_frame = post_render(
            frame,
            environment_configuration=config.environment_configuration
        )

        # Convert JAX array to NumPy for gym compatibility
        return np.array(processed_frame)

    def close(self):
        """Clean up resources."""
        pass


# Utility function to create the environment
def make_brittle_star_env(target_position=(1.25, 0.75, 0.0), seed=0):
    """
    Create a brittle star gym environment.

    Args:
        target_position: The target position for the brittle star to reach.
        seed: Random seed for reproducibility.

    Returns:
        BrittleStarEnv: The created environment.
    """
    return BrittleStarEnv(
        target_position=target_position,
        seed=seed
    )