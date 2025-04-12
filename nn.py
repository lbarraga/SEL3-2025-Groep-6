from typing import Any

import flax.linen as nn
import flax.serialization
import jax.numpy as jnp


class CPGController(nn.Module):
    num_outputs: int
    hidden_dim: int = 32

    @nn.compact
    def __call__(self, norm_direction: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(features=self.hidden_dim)(norm_direction)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_outputs)(x)
        return x

def save_model_params(params: Any, filename: str):
    param_bytes = flax.serialization.to_bytes(params)
    with open(filename, "wb") as f:
        f.write(param_bytes)

def load_model_params(filename: str, example_params: Any) -> Any:
    with open(filename, "rb") as f:
        param_bytes = f.read()
    loaded_params = flax.serialization.from_bytes(example_params, param_bytes)
    return loaded_params
