import math

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv

from SimulationState import create_initial_simulation_state, SimulationState
from config import NUM_ARMS, NUM_OSCILLATORS_PER_ARM, create_environment
from cpg import CPGState


def reset_cpg(rng: chex.PRNGKey) -> CPGState:
    num_oscillators = NUM_ARMS * NUM_OSCILLATORS_PER_ARM
    phase_rng, amplitude_rng, offsets_rng = jax.random.split(rng, 3)
    # noinspection PyArgumentList
    return CPGState(
        phases=jax.random.uniform(
            key=phase_rng, shape=(num_oscillators,), dtype=jnp.float32, minval=-0.001, maxval=0.001
        ),
        amplitudes=jnp.zeros(num_oscillators),
        offsets=jnp.zeros(num_oscillators),
        dot_amplitudes=jnp.zeros(num_oscillators),
        dot_offsets=jnp.zeros(num_oscillators),
        outputs=jnp.zeros(num_oscillators),
        time=0.0,
        R=jnp.zeros(num_oscillators),
        X=jnp.zeros(num_oscillators),
        omegas=jnp.zeros(num_oscillators)
    )

def get_joint_positions(sim_state: SimulationState) -> jnp.array:
    return sim_state.env_state.observations["unit_xy_direction_to_targe"]

class JAXVecEnv(VecEnv):
    def __init__(self, env_constructor, num_envs):
        # self.env_constructor = env_constructor
        self.num_envs = num_envs
        self.seed = 0
        self._rng = jax.random.PRNGKey(self.seed)
        # Instantiate once and reuse
        self.envs: jnp.ndarray = jnp.array([create_initial_simulation_state(reset_cpg(self._rng), create_environment()) for _ in range(num_envs)])
        # self.envs = [env_constructor() for _ in range(num_envs)]

        joint_positions = get_joint_positions(self.envs[0])
        len_jpos = len(joint_positions)

        amplitudes = self.envs[0].cpg_state.amplitudes
        len_amplitudes = len(amplitudes)

        self.observation_space = Box(
            low=np.concatenate([
                np.array([0.0, -math.inf, -math.inf]),
                np.full(len_jpos, -1),
                np.full(len_amplitudes, -self.max_joint_limit)
            ]),
            high=np.concatenate([
                np.array([2 * math.pi, math.inf, math.inf]),
                np.full(len_jpos, 1),
                np.full(len_amplitudes, self.max_joint_limit)
            ]),
            shape=(3 + len_jpos + len_amplitudes,),
            dtype=np.float64
        )

        self.action_space = self.envs[0].action_space

        super().__init__(num_envs, self.observation_space, self.action_space)

        self._reset_fn = jax.jit(jax.vmap(self._env_reset))
        self._step_fn = jax.jit(jax.vmap(self._env_step))

        self.jax_states = self.reset()

        # self.jax_states = self._reset_fn(self.envs)

        ####################
        ##### NEW CODE #####
        ####################







        ####################
        ##### OLD CODE #####
        ####################

    def _env_reset(self, env):
        # Reset pre-instantiated envs[i] via pure functional wrap
        obs = env.reset()
        return obs

    def _env_step(self, action, env):
        obs, reward, terminated, truncated, info = env.step(np.array(action))
        done = jax.lax.bitwise_or(terminated, truncated)
        return obs, reward, done, info

    def reset(self):
        obs_list = []
        for env in self.envs:
            obs = env.reset()
            obs_list.append(obs)
        self.jax_states = jnp.array(obs_list)
        return np.array(self.jax_states)

    def step(self, actions: np.ndarray):
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = self.envs[i].step(action)
            done = terminated or truncated
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        self.jax_states = jnp.array(obs_list)
        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(done_list),
            info_list
        )

    # def reset(self):
    #     self.jax_states = self._reset_fn(self.envs)
    #     return np.array(self.jax_states)
    #
    # def step(self, actions: np.ndarray):
    #     # We use partial application to map over (action, index)
    #     observations, rewards, dones, infos = self._step_fn(actions, self.envs)
    #     self.jax_states = observations
    #     return (
    #         np.array(observations),
    #         np.array(rewards),
    #         np.array(dones),
    #         infos
    #     )
    #
    def close(self):
        for env in self.envs:
            env.close()

    def seed(self, seed=None):
        if seed is not None:
            for i, env in enumerate(self.envs):
                env.seed(seed + i)

    def step_async(self, actions: np.ndarray):
        self.current_actions = actions

    def step_wait(self):
        return self.step(self.current_actions)

    def get_attr(self, attr_name, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, value, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for i in indices:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if indices is None:
            indices = range(self.num_envs)
        return [
            getattr(self.envs[i], method_name)(*method_args, **method_kwargs)
            for i in indices
        ]

    def get_images(self):
        return [env.render() for env in self.envs]

    def env_is_wrapped(self, wrapper_class, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        return [isinstance(self.envs[i].unwrapped, wrapper_class) for i in indices]

