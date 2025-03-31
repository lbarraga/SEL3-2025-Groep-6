import functools
from typing import Callable

import chex
import jax
import jax.numpy as jnp
from flax import struct

def euler_solver(
    current_time: float,
    y: jnp.ndarray,
    derivative_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
    delta_time: float
) -> jnp.ndarray:
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


class CPG:
    def __init__(
        self,
        amplitude_gain: float = 40,
        offset_gain: float = 40,
        dt: float = 0.01
    ) -> None:
        self._amplitude_gain = amplitude_gain
        self._offset_gain = offset_gain
        self._dt = dt
        self._solver = euler_solver

    @property
    def num_oscillators(self) -> int:
        return 10 # TODO dont hardcode

    @staticmethod
    def phase_de(omegas: jnp.ndarray) -> jnp.ndarray:
        return omegas # No coupling, so we return just the omegas

    @staticmethod
    def second_order_de(
        gain: float,
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
            omegas=jnp.zeros(self.num_oscillators)
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: CPGState) -> CPGState:
        # Update phase
        new_phases = self._solver(
            current_time=state.time,
            y=state.phases,
            derivative_fn=lambda t,y: self.phase_de(omegas=state.omegas),
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


@jax.jit
def modulate_cpg(
        cpg_state: CPGState,
        new_R: jnp.ndarray,
        new_X: jnp.ndarray,
        new_omega: float,
        max_joint_limit: float
) -> CPGState:

    X = jnp.clip(new_X, -max_joint_limit, max_joint_limit)
    R = jnp.clip(new_R, -max_joint_limit, max_joint_limit)
    omegas = jnp.broadcast_to(new_omega, R.shape)

    # Return the updated CPGState with modulated values
    return cpg_state.replace(R=R, X=X, omegas=omegas)


@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def map_cpg_outputs_to_actions(
        cpg_state: CPGState,
        num_arms: int,
        num_segments_per_arm: int,
        num_oscillators_per_arm: int
) -> jnp.ndarray:
    cpg_outputs_per_arm = cpg_state.outputs.reshape((num_arms, num_oscillators_per_arm))
    cpg_outputs_per_segment = cpg_outputs_per_arm.repeat(num_segments_per_arm, axis=0)

    return cpg_outputs_per_segment.flatten()
