from functools import partial

import jax
import jax.numpy as jnp
from evosax.problems.problem import Problem, State
from evosax.types import Fitness, Metrics, PyTree, Solution, Population
from flax import linen as nn, struct

from brittlestar_gym_environment import BrittleStarEnv


@struct.dataclass
class State(State):
    direction: PyTree
    dist_to_target: PyTree
    cpg_state: PyTree


class BrittleStarProblem(Problem):
    """Brittle Star Problem for Evolutionary Strategies."""

    def __init__(
            self,
            env_name: str,
            policy: nn.Module,
            episode_length: int | None = None,
            num_rollouts: int = 1
    ):
        """Initialize the Brittle Star problem."""
        self.env_name = env_name
        self.policy = policy
        self.num_rollouts = num_rollouts

        self.env = BrittleStarEnv()

        # Test policy and env compatibility
        key = jax.random.key(0)
        obs, info = self.env.reset()

        policy_params = self.policy.init(key, obs, key)

        action = self.policy.apply(policy_params, obs, key)
        self.env.step(action)

        self.episode_length = episode_length

        self._rollouts = jax.vmap(self._rollout, in_axes=(0, None, None))
        self._eval = jax.vmap(self._rollouts, in_axes=(None, 0, None))

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @partial(jax.jit, static_argnames=("self",))
    def init(self, key: jax.Array) -> State:
        """Initialize state with empty normalization statistics."""
        # Create a dummy observation to get the observation structure
        return State(
            counter=0,
            direction=self.env.get_direction_to_target(),
            dist_to_target= jnp.linalg.norm(self.env.get_brittle_star_position() - self.env.get_target_position()),
            cpg_state=self.env.cpg_state,
        )

    @partial(jax.jit, static_argnames=("self",))
    def eval(
        self,
        key: jax.Array,
        solutions: Population,
        state: State,
    ) -> tuple[Fitness, State, Metrics]:
        """Evaluate a population of policies."""
        keys = jax.random.split(key, self.num_rollouts)
        fitness, env_states = self._eval(keys, solutions, state)

        return (
            jnp.mean(fitness, axis=-1),
            state.replace(counter=state.counter + 1),
            {"env_states": env_states},
        )

    def _rollout(self, key: jax.Array, policy_params: PyTree, state: State):
        key_reset, key_scan = jax.random.split(key)

        # Reset environment
        obs, env_state = self.env.reset(key_reset)

        def _step(carry, key):
            obs, env_state, cum_reward, valid = carry

            key_action, key_step = jax.random.split(key)

            # Sample action from policy
            action = self.policy.apply(policy_params, obs, key_action)

            # Step environment
            obs, env_state, reward, done, _ = self.env.step(action)

            # Update cumulative reward and valid mask
            cum_reward = cum_reward + reward * valid
            valid = valid * (1 - done)
            carry = (
                obs,
                env_state,
                cum_reward,
                valid,
            )
            return carry, (obs, env_state)

        # Rollout
        keys = jax.random.split(key_scan, self.episode_length)
        carry, env_states = jax.lax.scan(
            _step,
            (
                obs,
                env_state,
                jnp.array(0.0),
                jnp.array(1.0),
            ),
            xs=keys,
        )

        # Return the sum of rewards accumulated by agent in episode rollout and states
        return carry[2], env_states

    @partial(jax.jit, static_argnames=("self",))
    def sample(self, key: jax.Array) -> Solution:
        """Sample a solution in the search space."""
        key_init, key_sample, key_input = jax.random.split(key, 3)
        obs = self.observation_space.sample(key_sample)
        return self.policy.init(key_init, obs, key_input)
