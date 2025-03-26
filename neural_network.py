import jax
import jax.numpy as jnp
from flax import linen as nn

class BrittleStarNN(nn.Module):
    def setup(self):
        self.dense1 = nn.Dense(64)  # First hidden layer with 64 units
        self.dense2 = nn.Dense(64)  # Second hidden layer with 64 units
        self.output_layer = nn.Dense(21)  # Output size 21 (R, X, omega)

    def __call__(self, inputs):
        x = self.dense1(inputs)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        outputs = self.output_layer(x)
        return outputs


if __name__ == '__main__':
    # Example usage
    rng = jax.random.PRNGKey(0)

    current_position = jax.random.uniform(rng, (3,), minval=-1.0, maxval=1.0)
    target_position = jax.random.uniform(rng, (3,), minval=-1.0, maxval=1.0)

    nn_input = jnp.concatenate([current_position, target_position])  # Shape (6,)

    model = BrittleStarNN()
    params = model.init(rng, nn_input) # init needs nn_input to infer the shape of the input

    output = model.apply(params, nn_input)
    R, X, omega = output[:10], output[10:20], output[20]

    print(f"R: {R}")
    print(f"X: {X}")
    print(f"omega: {omega}")
