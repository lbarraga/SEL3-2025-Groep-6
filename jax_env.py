import jax
import jax.numpy as jnp
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

class JAXVecEnv(VecEnv):
    def __init__(self, env_constructor, num_envs):
        self.env_constructor = env_constructor
        self.num_envs = num_envs

        # Instantiate once and reuse
        self.envs = [env_constructor() for _ in range(num_envs)]

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        super().__init__(num_envs, self.observation_space, self.action_space)

        self._reset_fn = jax.jit(jax.vmap(self._env_reset))
        self._step_fn = jax.jit(jax.vmap(self._env_step))

        # self.jax_states = self.reset()

        self.jax_states = self._reset_fn(jnp.arange(self.num_envs))

    def _env_reset(self, i):
        # Reset pre-instantiated envs[i] via pure functional wrap
        obs, info = self.envs[i].reset()
        return obs

    def _env_step(self, action, i):
        obs, reward, terminated, truncated, info = self.envs[i].step(np.array(action))
        done = jax.lax.bitwise_or(terminated, truncated)
        return obs, reward, done, info

    # def reset(self):
    #     obs_list = []
    #     for env in self.envs:
    #         obs, _ = env.reset()
    #         obs_list.append(obs)
    #     self.jax_states = jnp.array(obs_list)
    #     return np.array(self.jax_states)
    #
    # def step(self, actions: np.ndarray):
    #     obs_list, reward_list, done_list, info_list = [], [], [], []
    #     for i, action in enumerate(actions):
    #         obs, reward, terminated, truncated, info = self.envs[i].step(action)
    #         done = terminated or truncated
    #         obs_list.append(obs)
    #         reward_list.append(reward)
    #         done_list.append(done)
    #         info_list.append(info)
    #     self.jax_states = jnp.array(obs_list)
    #     return (
    #         np.array(obs_list),
    #         np.array(reward_list),
    #         np.array(done_list),
    #         info_list
    #     )

    def reset(self):
        self.jax_states = self._reset_fn(jnp.arange(self.num_envs))
        return np.array(self.jax_states)

    def step(self, actions: np.ndarray):
        indices = jnp.arange(self.num_envs)
        # We use partial application to map over (action, index)
        observations, rewards, dones, infos = jax.vmap(self._env_step)(actions, indices)
        self.jax_states = observations
        return (
            np.array(observations),
            np.array(rewards),
            np.array(dones),
            infos
        )

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

