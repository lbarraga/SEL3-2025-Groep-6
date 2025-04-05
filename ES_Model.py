from flax import nnx
import jax
import jax.numpy as jnp

class ESModel(nnx.Module):
    def __init__(self, din: int, dout: int, rngs: nnx.Rngs = nnx.Rngs(0)):
        key = rngs.params()
        self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
        self.b = nnx.Param(jnp.zeros((dout,)))
        self.din, self.dout = din, dout

    # def __call__(self, x):
    #     x = self.dense1(x)
    #     x = jax.nn.relu(x)
    #     x = self.dense2(x)
    #     return x