import functools
from typing import Callable, List
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from flax import struct

from config import morphology_specification, environment_configuration


def euler_solver(
    current_time: float,
    y: float,
    derivative_fn: Callable[[float, float], float],
    delta_time: float
) -> float:
    slope = derivative_fn(current_time, y)
    next_y = y + delta_time * slope
    return next_y


@struct.dataclass
class CPGState:
    time: float
    phases: jnp.ndarray
    dot_amplitudes: jnp.ndarray  # first order derivative of the amplitude
    amplitudes: jnp.ndarray
    dot_offsets: jnp.ndarray  # first order derivative of the offset 
    offsets: jnp.ndarray
    outputs: jnp.ndarray

    # We'll make these modulatory parameters part of the state as they will change as well
    R: jnp.ndarray
    X: jnp.ndarray
    omegas: jnp.ndarray
    rhos: jnp.ndarray


class CPG:
    def __init__(
        self,
        weights: jnp.ndarray,
        amplitude_gain: float = 20,
        offset_gain: float = 20,
        dt: float = 0.01
    ) -> None:
        self._weights = weights
        self._amplitude_gain = amplitude_gain
        self._offset_gain = offset_gain
        self._dt = dt
        self._solver = euler_solver

    @property
    def num_oscillators(self) -> int:
        return self._weights.shape[0]

    @staticmethod
    def phase_de(
        weights: jnp.ndarray,
        amplitudes: jnp.ndarray,
        phases: jnp.ndarray,
        phase_biases: jnp.ndarray,
        omegas: jnp.ndarray
    ) -> jnp.ndarray:
        @jax.vmap  # vectorizes this function for us over an additional batch dimension (in this case over all oscillators)
        def sine_term(phase_i: float, phase_biases_i: float) -> jnp.ndarray:
            return jnp.sin(phases - phase_i - phase_biases_i)

        couplings = jnp.sum(weights * amplitudes * sine_term(phase_i=phases, phase_biases_i=phase_biases), axis=1)
        return omegas + couplings

    @staticmethod
    def second_order_de(
        gain: jnp.ndarray,
        modulator: jnp.ndarray,
        values: jnp.ndarray,
        dot_values: jnp.ndarray
    ) -> jnp.ndarray:
        return gain * ((gain / 4) * (modulator - values) - dot_values)

    @staticmethod
    def first_order_de(dot_values: jnp.ndarray) -> jnp.ndarray:
        return dot_values

    @staticmethod
    def output(
        offsets: jnp.ndarray,
        amplitudes: jnp.ndarray,
        phases: jnp.ndarray
    ) -> jnp.ndarray:
        return offsets + amplitudes * jnp.cos(phases)

    def reset(self, rng: chex.PRNGKey) -> CPGState:
        phase_rng, amplitude_rng, offsets_rng = jax.random.split(rng, 3)
        # noinspection PyArgumentList
        return CPGState(
            phases=jax.random.uniform(
                key=phase_rng, shape=(self.num_oscillators,), dtype=jnp.float32, minval=-0.001, maxval=0.001
            ),
            amplitudes=jnp.zeros(self.num_oscillators),
            offsets=jnp.zeros(self.num_oscillators),
            dot_amplitudes=jnp.zeros(self.num_oscillators),
            dot_offsets=jnp.zeros(self.num_oscillators),
            outputs=jnp.zeros(self.num_oscillators),
            time=0.0,
            R=jnp.zeros(self.num_oscillators),
            X=jnp.zeros(self.num_oscillators),
            omegas=jnp.zeros(self.num_oscillators),
            rhos=jnp.zeros_like(self._weights)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: CPGState) -> CPGState:
        # Update phase
        new_phases = self._solver(
            current_time=state.time,
            y=state.phases,
            derivative_fn=lambda t,y: self.phase_de(
                omegas=state.omegas,
                amplitudes=state.amplitudes,
                phases=y,
                phase_biases=state.rhos,
                weights=self._weights
            ),
            delta_time=self._dt
        )
        new_dot_amplitudes = self._solver(
            current_time=state.time,
            y=state.dot_amplitudes,
            derivative_fn=lambda t, y: self.second_order_de(
                gain=self._amplitude_gain, modulator=state.R, values=state.amplitudes, dot_values=y
            ),
            delta_time=self._dt
        )
        new_amplitudes = self._solver(
            current_time=state.time,
            y=state.amplitudes,
            derivative_fn=lambda t, y: self.first_order_de(
                dot_values=state.dot_amplitudes
            ),
            delta_time=self._dt
        )
        new_dot_offsets = self._solver(
            current_time=state.time,
            y=state.dot_offsets,
            derivative_fn=lambda t, y: self.second_order_de(
                gain=self._offset_gain, modulator=state.X, values=state.offsets, dot_values=y
            ),
            delta_time=self._dt
        )
        new_offsets = self._solver(
            current_time=0,
            y=state.offsets,
            derivative_fn=lambda t, y: self.first_order_de(
                dot_values=state.dot_offsets
            ),
            delta_time=self._dt
        )

        new_outputs = self.output(offsets=new_offsets, amplitudes=new_amplitudes, phases=new_phases)
        # noinspection PyUnresolvedReferences
        return state.replace(
            phases=new_phases,
            dot_amplitudes=new_dot_amplitudes,
            amplitudes=new_amplitudes,
            dot_offsets=new_dot_offsets,
            offsets=new_offsets,
            outputs=new_outputs,
            time=state.time + self._dt
        )


def create_cpg() -> CPG:
    ip_oscillator_indices = jnp.arange(0, 10, 2)
    oop_oscillator_indices = jnp.arange(1, 10, 2)

    adjacency_matrix = jnp.zeros((10, 10))
    # Connect oscillators within an arm
    adjacency_matrix = adjacency_matrix.at[ip_oscillator_indices, oop_oscillator_indices].set(1)
    # Connect IP oscillators of neighbouring arms
    adjacency_matrix = adjacency_matrix.at[
        ip_oscillator_indices, jnp.concatenate((ip_oscillator_indices[1:], jnp.array([ip_oscillator_indices[0]])))
    ].set(1)
    # Connect OOP oscillators of neighbouring arms
    adjacency_matrix = adjacency_matrix.at[
        oop_oscillator_indices, jnp.concatenate((oop_oscillator_indices[1:], jnp.array([oop_oscillator_indices[0]]))
    )].set(1)

    # Make adjacency matrix symmetric (i.e. make all connections bi-directional)
    adjacency_matrix = jnp.maximum(adjacency_matrix, adjacency_matrix.T)

    return CPG(
        weights=10 * adjacency_matrix,
        amplitude_gain=40,
        offset_gain=40,
        dt=environment_configuration.control_timestep
    )


def get_oscillator_indices_for_arm(arm_index: int) -> Tuple[int, int]:
    return arm_index * 2, arm_index * 2 + 1


@jax.jit
def modulate_cpg(
    cpg_state: CPGState,
    leading_arm_index: int,
    max_joint_limit: float
) -> CPGState:
    left_rower_arm_indices = [(leading_arm_index - 1) % 5, (leading_arm_index - 2) % 5]
    right_rower_arm_indices = [(leading_arm_index + 1) % 5, (leading_arm_index + 2) % 5]

    # TODO: unused, so remove?
    # leading_arm_ip_oscillator_index, leading_arm_oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=leading_arm_index)

    R = jnp.zeros_like(cpg_state.R)
    X = jnp.zeros_like(cpg_state.X)
    rhos = jnp.zeros_like(cpg_state.rhos)
    omegas = 2 * jnp.pi * jnp.ones_like(cpg_state.omegas)
    phases_bias_pairs = []

    def modulate_leading_arm(_X: jnp.ndarray, _arm_index: int) -> jnp.ndarray:
        ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
        return _X.at[oop_oscillator_index].set(max_joint_limit)

    def modulate_left_rower(_R: jnp.ndarray, _arm_index: int) -> Tuple[jnp.ndarray, List[Tuple[int, int, float]]]:
        ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
        _R = _R.at[ip_oscillator_index].set(max_joint_limit)
        _R = _R.at[oop_oscillator_index].set(max_joint_limit)
        _phase_bias_pairs = [(ip_oscillator_index, oop_oscillator_index, jnp.pi / 2)]
        return _R, _phase_bias_pairs

    def modulate_right_rower(_R: jnp.ndarray, _arm_index: int) -> Tuple[jnp.ndarray, List[Tuple[int, int, float]]]:
        ip_oscillator_index, oop_oscillator_index = get_oscillator_indices_for_arm(arm_index=_arm_index)
        _R = _R.at[ip_oscillator_index].set(max_joint_limit)
        _R = _R.at[oop_oscillator_index].set(max_joint_limit)
        _phase_bias_pairs = [(ip_oscillator_index, oop_oscillator_index, -jnp.pi / 2)]
        return _R, _phase_bias_pairs

    def phase_biases_second_rowers(_left_arm_index: int, _right_arm_index: int) -> List[Tuple[int, int, float]]:
        left_ip_oscillator_index, _ = get_oscillator_indices_for_arm(arm_index=_left_arm_index)
        right_ip_oscillator_index, _ = get_oscillator_indices_for_arm(arm_index=_right_arm_index)
        _phase_bias_pairs = [(left_ip_oscillator_index, right_ip_oscillator_index, jnp.pi)]
        return _phase_bias_pairs

    X = modulate_leading_arm(_X=X, _arm_index=leading_arm_index)

    R, phb = modulate_left_rower(_R=R, _arm_index=left_rower_arm_indices[0])
    phases_bias_pairs += phb

    R, phb = modulate_left_rower(_R=R, _arm_index=left_rower_arm_indices[1])
    phases_bias_pairs += phb

    R, phb = modulate_right_rower(_R=R, _arm_index=right_rower_arm_indices[0])
    phases_bias_pairs += phb

    R, phb = modulate_right_rower(_R=R, _arm_index=right_rower_arm_indices[1])
    phases_bias_pairs += phb

    phases_bias_pairs += phase_biases_second_rowers(_left_arm_index=left_rower_arm_indices[1], _right_arm_index=right_rower_arm_indices[1])

    for oscillator1, oscillator2, bias in phases_bias_pairs:
        rhos = rhos.at[oscillator1, oscillator2].set(bias)
        rhos = rhos.at[oscillator2, oscillator1].set(-bias)

    # noinspection PyUnresolvedReferences
    return cpg_state.replace(R=R, X=X, rhos=rhos, omegas=omegas)


@jax.jit
def map_cpg_outputs_to_actions(cpg_state: CPGState) -> jnp.ndarray:
    num_arms = morphology_specification.number_of_arms
    num_oscillators_per_arm = 2
    num_segments_per_arm = morphology_specification.number_of_segments_per_arm[0]

    cpg_outputs_per_arm = cpg_state.outputs.reshape((num_arms, num_oscillators_per_arm))
    cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments_per_arm, axis=0)

    return cpg_outputs_per_segment.flatten()
