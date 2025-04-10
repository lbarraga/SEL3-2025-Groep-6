from functools import partial
from typing import Dict, Tuple, Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification
from gymnasium import spaces
from overrides import overrides

from cpg import modulate_cpg, map_cpg_outputs_to_actions, CPG


class BrittleStarEnv(gym.Env):
    """
    Gym environment wrapper for the brittle star simulation.

    This environment integrates the JAX-based brittle star simulation with
    Gym's interface, enabling compatibility with RL libraries.
    """

    NUM_ARMS = 5
    NUM_SEGMENTS_PER_ARM = 3
    ARENA_SIZE = (2, 2)

    morphology_specification = default_brittle_star_morphology_specification(
        num_arms=NUM_ARMS,
        num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
        use_p_control=True,
        use_torque_control=False
    )

    arena_configuration = AquariumArenaConfiguration(
        size=ARENA_SIZE,
        sand_ground_color=False,
        attach_target=True,
        wall_height=1.5,
        wall_thickness=0.1
    )

    environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
        target_distance=1.2,
        joint_randomization_noise_scale=0.0,
        render_mode="rgb_array",
        simulation_time=20,
        num_physics_steps_per_control_step=10,
        time_scale=2,
        camera_ids=[0, 1],
        render_size=(480, 640)
    )

    metadata = {"render_fps": 30}

    def __init__(self, target_position=(1.25, 0.75, 0.0), seed=0):
        self.target_position = target_position
        self.seed_value = seed
        self.cpg_steps_per_ppo_step = 30

        self.env = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
            morphology=MJCFBrittleStarMorphology(specification=self.morphology_specification),
            arena=MJCFAquariumArena(configuration=self.arena_configuration),
            configuration=self.environment_configuration,
            backend="MJX"
        )

        # Create the CPG
        self.cpg = CPG(dt=self.environment_configuration.control_timestep)

        # JIT compile the environment step function
        self.env_step_fn = jax.jit(self.env.step)
        self.jit_reset = jax.jit(partial(self.env.reset, target_position=target_position))

        # Define action space with correct bounds (flattened version)
        # R values (10 values): -1.0 to 1.0
        # X values (10 values): -1.0 to 1.0
        # omega value (1 value): 0.0 to 5.0
        self.action_space = spaces.Box(
            low=np.concatenate([np.full(10, -1.0), np.full(10, -1.0), np.array([1.0])]),
            high=np.concatenate([np.full(10, 1.0), np.full(10, 1.0), np.array([5.0])]),
            shape=(21,),
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
        current_position = self.get_brittle_star_position()
        target_position = self.get_target_position()

        # Convert JAX arrays to NumPy for gym compatibility
        return np.concatenate([
            np.array(current_position),
            np.array(target_position)
        ])

    def get_brittle_star_position(self):
        return self.env_state.observations["disk_position"]

    def get_target_position(self):
        return jnp.concatenate([self.env_state.info["xy_target_position"], jnp.array([0.0])])

    def get_disk_rotation(self) -> float:
        return self.env_state.observations["disk_rotation"][-1]

    def get_joint_positions(self) -> jnp.array:
        return self.env_state.observations["joint_positions"]

    def get_direction_to_target(self) -> jnp.array:
        return self.env_state.observations["unit_xy_direction_to_target"]

    def _get_reward(self) -> float:
        """Calculate the reward based on improvement in distance to target."""
        current_position = self.get_brittle_star_position()
        target_position = self.get_target_position()

        # Calculate current distance to target
        current_distance = jnp.linalg.norm(current_position - target_position)

        # Calculate the improvement (previous distance - current distance)
        distance_improvement = self.previous_distance - current_distance

        # Store current distance for next step
        self.previous_distance = current_distance

        # Add a bonus for reaching the target
        target_reached_bonus = jnp.where(current_distance < 0.1, 10.0, 0.0)

        # Combine rewards
        reward = distance_improvement + target_reached_bonus

        return float(reward)

    def is_terminated(self) -> bool:
        """Check if the episode should terminate."""
        # Check if the brittle star has reached the target
        current_position = self.get_brittle_star_position()
        target_position = self.get_target_position()
        distance = jnp.linalg.norm(current_position - target_position)

        # Terminate if target reached or from the environment's terminated signal
        target_reached = distance < 0.1
        return bool(target_reached | self.env_state.terminated)

    def is_truncated(self) -> bool:
        """Check if the episode should be truncated (e.g., max steps reached)."""
        return bool(self.env_state.truncated)

    @overrides()
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.seed_value = seed
            self.rng = jax.random.PRNGKey(seed=seed)

        self._initialize_states()

        # Get current position and target for initial distance
        current_position = self.get_brittle_star_position()
        target_position = self.get_target_position()
        self.previous_distance = float(jnp.linalg.norm(current_position - target_position))

        observation = self._get_observation()
        info = {}

        return observation, info

    def modulate_cpg(self, action: jnp.ndarray) -> None:
        # Convert numpy action to JAX array
        action = jnp.array(action)

        # Extract CPG parameters from action
        new_R, new_X, new_omega = action[:10], action[10:20], action[20]

        # Modulate the CPG with the action only once per PPO step
        self.cpg_state = modulate_cpg(
            cpg_state=self.cpg_state,
            new_R=new_R,
            new_X=new_X,
            new_omega=new_omega,
            max_joint_limit=self.env.action_space.high[0]
        )


    @overrides()
    def step(self, action: jnp.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.
        The action modulates the CPG once, then lets it run for multiple steps.
        """
        # Convert numpy action to JAX array
        self.modulate_cpg(action)

        distance_before = jnp.linalg.norm(self.get_brittle_star_position() - self.get_target_position())

        best_distance = distance_before
        steps_taken = 0
        last_progress_step = 0
        no_progress_threshold = 20  # Stop if no progress for this many steps
        best_improvement = 0

        while steps_taken < 300:
            self.single_step()
            steps_taken += 1

            current_position = self.get_brittle_star_position()
            target_position = self.get_target_position()
            current_distance = jnp.linalg.norm(current_position - target_position)

            if current_distance < best_distance:
                best_improvement = best_distance - current_distance
                best_distance = current_distance
                last_progress_step = steps_taken

            if self.is_terminated() or self.is_truncated():
                break

            if (steps_taken - last_progress_step) > no_progress_threshold:
                break

        distance_after = jnp.linalg.norm(self.get_brittle_star_position() - self.get_target_position())
        improvement = distance_before - distance_after
        target_reached_bonus = 10 if distance_after < 0.1 else 0
        reward = float(improvement + target_reached_bonus)

        info = {
            "steps_taken": steps_taken,
            "best_distance": best_distance,
            "reward": reward,
            "is_terminated": self.is_terminated(),
            "is_truncated": self.is_truncated(),
            "best_improvement": best_improvement,
            "target_reached": distance_after < 0.1,
        }

        return self._get_observation(), reward, self.is_terminated(), self.is_truncated(), info


    def single_step(self):
        self.cpg_state = self.cpg.step(state=self.cpg_state)

        # Map CPG outputs to actions
        motor_actions = map_cpg_outputs_to_actions(
            cpg_state=self.cpg_state,
            num_arms=self.NUM_ARMS,
            num_segments_per_arm=self.NUM_SEGMENTS_PER_ARM,
            num_oscillators_per_arm=2
        )

        # Step the environment
        self.env_state = self.env_step_fn(state=self.env_state, action=motor_actions)
        return self._get_observation()


    def render(self):
        from render import post_render

        # Render the environment and post-process the frame
        frame = self.env.render(state=self.env_state)
        processed_frame = post_render(
            frame,
            environment_configuration=self.environment_configuration
        )

        # Convert JAX array to NumPy for gym compatibility
        return np.array(processed_frame)

    def close(self):
        """Clean up resources."""
        pass