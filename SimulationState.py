import typing

from flax import struct
import jax.numpy as jnp


@struct.dataclass
class SimulationState:
    env_state: typing.Any
    cpg_state: typing.Any
    initial_distance: jnp.ndarray
    best_distance: jnp.ndarray
    current_distance: jnp.ndarray
    steps_taken: jnp.ndarray
    last_progress_step: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    reward: jnp.ndarray # Note: Intermediate reward stored here


def create_initial_simulation_state(env_state, cpg_state):
    target_pos_xy = env_state.info["xy_target_position"]
    target_pos_3d = jnp.concatenate([target_pos_xy, jnp.array([0.0])])

    current_pos = env_state.observations["disk_position"]
    initial_distance = jnp.linalg.norm(current_pos - target_pos_3d)
    return SimulationState(
        env_state=env_state,
        cpg_state=cpg_state,
        initial_distance=initial_distance,
        best_distance=initial_distance,
        current_distance=initial_distance,
        steps_taken=jnp.array(0, dtype=jnp.int32),
        last_progress_step=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),
        truncated=jnp.array(False),
        reward=jnp.array(0.0, dtype=jnp.float32)
    )