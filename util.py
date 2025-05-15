import pickle
from typing import List

import jax
import jax.numpy as jnp

from config import TARGET_SAMPLING_RADIUS, FIXED_OMEGA


def sample_random_target_pos(rng_single):
    """Samples a random target position on the circle perimeter."""
    angle = jax.random.uniform(rng_single, minval=0, maxval=2 * jnp.pi)
    radius = TARGET_SAMPLING_RADIUS
    target_pos = jnp.array([radius * jnp.cos(angle), radius * jnp.sin(angle), 0.0])
    return target_pos

def calculate_direction(target_pos: List[float]) -> List[float]:
    """Calculates the normalized direction vector from the origin to the target position."""
    target_pos_2d = target_pos[:2]
    norm = TARGET_SAMPLING_RADIUS
    normalized_direction = target_pos_2d / norm
    return normalized_direction

def normalize_corner(omega: float) -> float:
    """Normalizes the direction vector to have a magnitude of 1."""
    while omega < 0:
        omega += 2 * jnp.pi
    while omega > 2 * jnp.pi:
        omega -= 2 * jnp.pi
    return omega

def generate_cpg_for_eval(
        rng_single,
        flat_model_params_single,
        model_obj,
        unravel_fn_single
):
    target_pos = sample_random_target_pos(rng_single)
    direction = calculate_direction(target_pos)

    # Pass direction to model to get CPG parameters
    model_params_single = unravel_fn_single(flat_model_params_single)
    generated_rx_params = model_obj.apply({'params': model_params_single}, direction)
    cpg_params = jnp.concatenate([generated_rx_params, jnp.array([FIXED_OMEGA])])

    return cpg_params, target_pos

def print_optuna_results(pickle_file: str):
    with open(pickle_file, "rb") as f:
        study = pickle.load(f)
        for trial in study.trials:
            print(f"Params: {trial.params}, Values: {trial.values}")
        print(f"Best params: {study.best_trial.params}, Best values: {study.best_trial.values}")
