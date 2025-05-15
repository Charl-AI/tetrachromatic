"""JAX+NNX reimplementation of the EDM2 architecture by Karras et al."""

# Written by C Jones, 2025. MIT License.

import jax
import jax.numpy as jnp
from flax import nnx


def mp_silu(x: jax.Array) -> jax.Array:
    return jax.nn.silu(x) / 0.596


def mp_sum(a: jax.Array, b: jax.Array, t: float = 0.5):
    lerp = a * (1 - t) + t * b
    scale = jnp.sqrt((1 - t) ** 2 + t**2)
    return lerp / scale


def mp_cat(a: jax.Array, b: jax.Array, axis: int = 1, t: float = 0.5):
    Na = a.shape[axis]
    Nb = b.shape[axis]
    C = jnp.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / jnp.sqrt(Na) * (1 - t)
    wb = C / jnp.sqrt(Nb) * t
    return jnp.concat([wa * a, wb * b], axis=axis)


class MPFourier(nnx.Module):
    def __init__(self, num_channels: int, bandwidth: int = 1, *, rngs: nnx.Rngs):
        fkey = rngs.params()
        pkey = rngs.params()

        self.freqs = nnx.Variable(
            2 * jnp.pi * jax.random.normal(fkey, (num_channels,)) * bandwidth
        )
        self.phases = nnx.Variable(
            2 * jnp.pi * jax.random.uniform(pkey, (num_channels,))
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        y = x.astype(jnp.float32)
        y = jnp.outer(y, self.freqs.astype(jnp.float32))
        y = y + self.phases.astype(jnp.float32)
        y = jnp.cos(y) * jnp.sqrt(2)
        return y.astype(x.dtype)


def normalize(
    x: jax.Array,
    axis: int | tuple[int, ...] | None = None,
    eps: float = 1e-4,
):
    if axis is None:
        if x.ndim == 0:
            axis = None
        elif x.ndim == 1:
            axis = 0
        else:
            axis = tuple(range(1, x.ndim))

    x = x.astype(jnp.float32)
    norm = jnp.linalg.vector_norm(x, axis=axis, keepdims=True)

    scale = jnp.sqrt(norm.size / x.size)
    norm = eps + norm * scale
    return x / norm.astype(x.dtype)


class MPConv(nnx.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel: tuple[int], *, rngs: nnx.Rngs
    ):
        wkey = rngs.params()
        self.out_channels = out_channels
        self.weight = nnx.Param(
            jax.random.normal(wkey, shape=(out_channels, in_channels, *kernel))
        )

    def forward(self, x: jax.Array, gain: float = 1):
        w = self.weight.astype(jnp.float32)
