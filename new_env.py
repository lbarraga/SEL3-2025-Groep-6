# functional_brittle_star_env.py
from functools import partial
from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
# Assume these imports work and the underlying objects/functions are JAX-compatible
from biorobot.brittle_star.environment.directed_locomotion.dual import BrittleStarDirectedLocomotionEnvironment
from biorobot.brittle_star.environment.directed_locomotion.shared import \
    BrittleStarDirectedLocomotionEnvironmentConfiguration
from biorobot.brittle_star.mjcf.arena.aquarium import MJCFAquariumArena, AquariumArenaConfiguration
from biorobot.brittle_star.mjcf.morphology.morphology import MJCFBrittleStarMorphology
from biorobot.brittle_star.mjcf.morphology.specification.default import default_brittle_star_morphology_specification

from cpg import CPG, modulate_cpg, map_cpg_outputs_to_actions

# --- Configuration ---
NUM_ARMS = 5
NUM_SEGMENTS_PER_ARM = 3
ARENA_SIZE = (2, 2)
TARGET_POSITION_XY = jnp.array([1.25, 0.75])
TARGET_POSITION = jnp.concatenate([TARGET_POSITION_XY, jnp.array([0.0])])
MAX_SIM_STEPS = 150
NO_PROGRESS_THRESHOLD = 20
LOWER_BOUNDS = jnp.concatenate([jnp.full(10, -1.0), jnp.full(10, -1.0), jnp.array([1.0])])
UPPER_BOUNDS = jnp.concatenate([jnp.full(10, 1.0), jnp.full(10, 1.0), jnp.array([5.0])])


# --- Static Environment Configuration ---
morphology_specification = default_brittle_star_morphology_specification(
    num_arms=NUM_ARMS, num_segments_per_arm=NUM_SEGMENTS_PER_ARM, use_p_control=True, use_torque_control=False
)
arena_configuration = AquariumArenaConfiguration(
    size=ARENA_SIZE, sand_ground_color=False, attach_target=True, wall_height=1.5, wall_thickness=0.1
)
environment_configuration = BrittleStarDirectedLocomotionEnvironmentConfiguration(
    target_distance=1.2,
    joint_randomization_noise_scale=0.0,
    render_mode="rgb_array",
    simulation_time=20,
    num_physics_steps_per_control_step=10,
    time_scale=2,
    camera_ids=[0, 1],
    render_size=(480, 640),
)

# --- Underlying BioRobot Env (Static Instance for Methods/Info) ---
bio_env_instance = BrittleStarDirectedLocomotionEnvironment.from_morphology_and_arena(
    morphology=MJCFBrittleStarMorphology(specification=morphology_specification),
    arena=MJCFAquariumArena(configuration=arena_configuration),
    configuration=environment_configuration,
    backend="MJX"
)
_control_timestep = environment_configuration.control_timestep
max_joint_limit = UPPER_BOUNDS[0]
# Get JITted step/reset functions - Pass target_pos as runtime arg now
bio_env_step_fn = jax.jit(bio_env_instance.step)
# Reset needs the target_position passed to it
bio_env_reset_fn = jax.jit(partial(bio_env_instance.reset))


# --- CPG Setup ---
cpg_instance = CPG(dt=_control_timestep)
cpg_reset_fn = jax.jit(cpg_instance.reset)
cpg_step_fn = jax.jit(cpg_instance.step)
modulate_cpg_fn = jax.jit(partial(modulate_cpg, max_joint_limit=max_joint_limit))
map_cpg_outputs_fn = jax.jit(partial(map_cpg_outputs_to_actions,
                                       num_arms=NUM_ARMS,
                                       num_segments_per_arm=NUM_SEGMENTS_PER_ARM,
                                       num_oscillators_per_arm=2))


# --- State Definition ---
class EnvState(NamedTuple):
    mjx_state: Any
    cpg_state: Any
    key: jax.random.PRNGKey
    steps_taken: int
    previous_distance: float
    best_distance: float
    last_progress_step: int
    terminated: bool
    truncated: bool


# --- Core JAX Helper Functions ---
# No longer need partial(jit, static_argnames...) for target_pos

@jax.jit
def get_brittle_star_position(mjx_state: Any) -> jnp.ndarray:
    return mjx_state.observations["disk_position"]

@jax.jit
def calculate_distance(mjx_state: Any, target_pos: jnp.ndarray) -> float:
    current_position = get_brittle_star_position(mjx_state)
    return jnp.linalg.norm(current_position - target_pos)

@jax.jit
def get_observation(mjx_state: Any, target_pos: jnp.ndarray) -> jnp.ndarray:
    current_position = get_brittle_star_position(mjx_state)
    return jnp.concatenate([current_position, target_pos])

@jax.jit
def check_termination_conditions(mjx_state: Any, target_pos: jnp.ndarray) -> tuple[bool, bool]:
    distance = calculate_distance(mjx_state, target_pos)
    target_reached = distance < 0.1
    terminated = target_reached | mjx_state.terminated
    truncated = mjx_state.truncated
    return terminated, truncated

# --- Functional Simulation Step Logic ---

@jax.jit
def _functional_single_step(mjx_state: Any, cpg_state: Any) -> tuple[Any, Any]:
    new_cpg_state = cpg_step_fn(state=cpg_state)
    motor_actions = map_cpg_outputs_fn(cpg_state=new_cpg_state)
    new_mjx_state = bio_env_step_fn(state=mjx_state, action=motor_actions)
    return new_mjx_state, new_cpg_state

# Pass target_pos as regular arg now
@partial(jax.jit, static_argnames=("no_progress_thresh",)) # Only no_progress_thresh is static
def _inner_simulation_step(loop_carry: EnvState, _: Any, target_pos: jnp.ndarray, no_progress_thresh: int) -> tuple[EnvState, None]:
    current_mjx_state, current_cpg_state, key, steps, prev_dist, best_dist, last_prog, term, trunc = loop_carry

    new_mjx_state, new_cpg_state = _functional_single_step(current_mjx_state, current_cpg_state)
    steps = steps + 1

    current_dist = calculate_distance(new_mjx_state, target_pos)
    new_best_dist = jnp.minimum(best_dist, current_dist)
    made_progress = current_dist < best_dist
    new_last_prog_step = jax.lax.select(made_progress, steps, last_prog)

    env_term, env_trunc = check_termination_conditions(new_mjx_state, target_pos)
    no_progress = (steps - new_last_prog_step) > no_progress_thresh
    new_term = term | env_term
    new_trunc = trunc | env_trunc | no_progress

    next_carry = EnvState(
        mjx_state=new_mjx_state,
        cpg_state=new_cpg_state,
        key=key,
        steps_taken=steps,
        previous_distance=prev_dist,
        best_distance=new_best_dist,
        last_progress_step=new_last_prog_step,
        terminated=new_term,
        truncated=new_trunc
    )
    return next_carry, None


# --- Main Environment Interface Functions ---

# Pass target_pos as regular arg
@jax.jit
def init_state(key: jax.random.PRNGKey, target_pos: jnp.ndarray) -> EnvState:
    key, subkey1, subkey2 = jax.random.split(key, 3)
    cpg_state = cpg_reset_fn(rng=subkey1)
    # Pass target_pos XY to the underlying reset function
    mjx_state = bio_env_reset_fn(rng=subkey2, target_position=target_pos[:3])
    initial_distance = calculate_distance(mjx_state, target_pos)

    return EnvState(
        mjx_state=mjx_state,
        cpg_state=cpg_state,
        key=key,
        steps_taken=0,
        previous_distance=initial_distance,
        best_distance=initial_distance,
        last_progress_step=0,
        terminated=False,
        truncated=False
    )

# Pass target_pos as regular arg, only max_steps and no_progress_thresh are static
@partial(jax.jit, static_argnames=("max_steps", "no_progress_thresh"))
def functional_env_step(initial_state: EnvState, cpg_params: jnp.ndarray, max_steps: int, target_pos: jnp.ndarray, no_progress_thresh: int) -> tuple[EnvState, float, jnp.ndarray]:
    clipped_params = jnp.clip(cpg_params, LOWER_BOUNDS, UPPER_BOUNDS)
    modulated_cpg_state = modulate_cpg_fn(
        cpg_state=initial_state.cpg_state,
        new_R=clipped_params[:10],
        new_X=clipped_params[10:20],
        new_omega=clipped_params[20]
    )
    start_loop_state = initial_state._replace(cpg_state=modulated_cpg_state)

    # Pass target_pos as a regular (non-static) argument to scan_body via partial
    scan_body = partial(_inner_simulation_step, target_pos=target_pos, no_progress_thresh=no_progress_thresh)

    final_loop_state, _ = jax.lax.scan(scan_body, start_loop_state, None, length=max_steps)

    final_dist = calculate_distance(final_loop_state.mjx_state, target_pos)
    improvement = initial_state.previous_distance - final_dist
    target_reached_bonus = jax.lax.select(final_dist < 0.1, 10.0, 0.0)
    reward = improvement + target_reached_bonus

    final_observation = get_observation(final_loop_state.mjx_state, target_pos)

    return final_loop_state, reward, final_observation


# --- Example Usage ---
if __name__ == '__main__':
    SEED = 42
    key = jax.random.PRNGKey(SEED)
    print("Initializing state...")
    key, init_key = jax.random.split(key)
    # Pass TARGET_POSITION as regular argument
    state = init_state(init_key, TARGET_POSITION)
    print(f"Initial distance: {state.previous_distance:.4f}")

    key, param_key = jax.random.split(key)
    mid_params = (LOWER_BOUNDS + UPPER_BOUNDS) / 2.0
    example_params = mid_params + jax.random.normal(param_key, shape=mid_params.shape) * 0.2

    print("\nRunning functional step...")
    # Pass TARGET_POSITION as regular argument
    final_state, reward, observation = functional_env_step(
        state, example_params, MAX_SIM_STEPS, TARGET_POSITION, NO_PROGRESS_THRESHOLD
    )
    print(f"Final Reward: {reward:.4f}")
    # print(f"Final Observation: {observation}")
    # print(f"Terminated: {final_state.terminated}, Truncated: {final_state.truncated}")

    # --- Conceptual VMAP Example (Optional) ---
    # print("\nRunning VMAP example...")
    # batch_size = 4
    # key, *init_keys = jax.random.split(key, batch_size + 1)
    # batch_init_keys = jnp.stack(init_keys)
    # vmapped_init = jax.vmap(init_state, in_axes=(0, None)) # Pass target_pos as None (broadcast)
    # batch_initial_states = vmapped_init(batch_init_keys, TARGET_POSITION)
    # key, *param_keys = jax.random.split(key, batch_size + 1)
    # batch_param_keys = jnp.stack(param_keys)
    # def generate_params(k): return mid_params + jax.random.normal(k, shape=mid_params.shape) * 0.2
    # batch_params = jax.vmap(generate_params)(batch_param_keys)
    # vmapped_step = jax.vmap(functional_env_step, in_axes=(0, 0, None, None, None), static_broadcasted_argnums=(2, 4)) # Pass target_pos as None (broadcast)
    # batch_final_states, batch_rewards, batch_obs = vmapped_step(batch_initial_states, batch_params, MAX_SIM_STEPS, TARGET_POSITION, NO_PROGRESS_THRESHOLD)
    # print(f"Batch rewards shape: {batch_rewards.shape}")
    # print(f"Example batch reward: {batch_rewards[0]:.4f}")

