from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import pickle

import config
from cpg import create_cpg, modulate_cpg, map_cpg_outputs_to_actions
from environment import create_directed_environment
from neural_network import BrittleStarNN


def train_step(train_state, batch, log_probs, advantages):
    # Placeholder training step function (to be implemented properly)
    loss = jnp.mean(advantages ** 2)  # Dummy loss function
    grads = jax.grad(lambda params: jnp.mean(advantages ** 2))(train_state.params)
    return train_state.apply_gradients(grads=grads)


def train_ppo(
        env,
        cpg,
        model,
        num_episodes: int = 100,
        max_steps_per_episode: int = 200,
        learning_rate: float = 1e-3
):
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((6,)))
    optimizer = optax.adam(learning_rate)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    env_step_fn = jax.jit(env.step)
    jit_reset = jax.jit(partial(env.reset, target_position=(-1.25, 0.75, 0.)))

    for episode in range(num_episodes):
        print(f"Episode {episode}")
        env_state = jit_reset(rng=jax.random.PRNGKey(episode))
        cpg_state = cpg.reset(rng=jax.random.PRNGKey(episode))

        episode_rewards = []
        trajectory = {"observations": [], "actions": [], "log_probs": [], "rewards": []}

        step = 0
        while step < max_steps_per_episode and not env_state.terminated and not env_state.truncated:
            current_position = env_state.observations["disk_position"]
            target_position = jnp.concatenate([
                env_state.info["xy_target_position"],
                jnp.array([0.0])
            ])
            nn_input = jnp.concatenate([current_position, target_position])

            output = model.apply(train_state.params, nn_input)
            new_R, new_X, new_omega = output[:10], output[10:20], output[20]

            cpg_state = modulate_cpg(
                cpg_state=cpg_state,
                new_R=new_R,
                new_X=new_X,
                new_omega=new_omega,
                max_joint_limit=env.action_space.high[0] * 0.25
            )

            cpg_state = cpg.step(state=cpg_state)
            actions = map_cpg_outputs_to_actions(cpg_state=cpg_state)
            env_state = env_step_fn(state=env_state, action=actions)

            reward = -jnp.linalg.norm(current_position - target_position)
            episode_rewards.append(reward)

            trajectory["observations"].append(nn_input)
            trajectory["actions"].append(output)
            trajectory["log_probs"].append(0.0)  # Placeholder for PPO log_probs
            trajectory["rewards"].append(reward)

            step += 1

        rewards = jnp.array(trajectory["rewards"])
        advantages = rewards - jnp.mean(rewards)
        batch = {k: jnp.stack(v) for k, v in trajectory.items()}
        train_state = train_step(train_state, batch, jnp.stack(trajectory["log_probs"]), advantages)

        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {jnp.sum(jnp.array(episode_rewards))}")

    return train_state.params


if __name__ == "__main__":
    env = create_directed_environment(
        config.morphology_specification,
        config.arena_configuration,
        config.environment_configuration,
        "MJX"
    )
    model = BrittleStarNN()
    cpg = create_cpg()

    trained_params = train_ppo(env, cpg, model)

    with open("trained_model.pkl", "wb") as f:
        pickle.dump(trained_params, f)
    print("Model training complete and saved.")
