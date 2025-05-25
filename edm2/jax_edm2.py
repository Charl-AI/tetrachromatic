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
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        wkey = rngs.params()
        self.out_channels = out_channels
        self.weight = nnx.Param(
            jax.random.normal(wkey, shape=(out_channels, in_channels, *kernel))
        )

    def __call__(self, x: jax.Array, gain: float = 1):
        w = self.weight.astype(jnp.float32)
        w = normalize(w)  # forced weight normalisation
        w = w * (gain / jnp.sqrt(w[0].size))  # MP-scaling
        w = w.astype(x.dtype)
        if w.ndim == 2:
            return x @ w.T
        assert w.ndim == 4
        return jax.lax.conv(x, w, window_strides=(1, 1), padding="SAME")


def resample(x, f: tuple[int, ...] = (1, 1), mode: str = "keep"):
    if mode == "keep":
        return x

    f_arr = jnp.array(f, dtype=jnp.float32)
    assert f_arr.ndim == 1 and len(f_arr) % 2 == 0
    pad = (len(f_arr) - 1) // 2
    f_arr = f_arr / f_arr.sum()
    f_arr = jnp.outer(f_arr, f_arr)[jnp.newaxis, jnp.newaxis, :, :]
    f_arr = const_like(x, f_arr)
    c = x.shape[1]

    if mode == "down":
        return torch.nn.functional.conv2d(
            x, f_arr.tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
        )
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f_arr * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=(pad,)
    )


class Block(nnx.Module):
    def __init__(self,
        in_channels: int,                 # Number of input channels.
        out_channels: int,                # Number of output channels.
        emb_channels: int,                # Number of embedding channels.
        flavor: str = "enc",              # Flavor: 'enc' or 'dec'.
        resample_mode: str = "keep",      # Resampling: 'keep', 'up', or 'down'.
        resample_filter: tuple = (1, 1),  # Resampling filter.
        attention: bool = False,          # Include self-attention?
        channels_per_head: int = 64,      # Number of channels per attention head.
        dropout: float = 0.0,             # Dropout probability.
        res_balance: float = 0.3,         # Balance between main branch (0) and residual branch (1).
        attn_balance: float = 0.3,        # Balance between main branch (0) and self-attention (1).
        clip_act: float = 256.0,          # Clip output activations. None = do not clip.
        *,
        rngs: nnx.Rngs
    ):  # fmt: skip
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act

        self.emb_gain = nnx.Variable(jnp.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels,
            out_channels,
            kernel=(3, 3),
            rngs=rngs,
        )
        # self.emb_linear = MPConv(
        #     emb_channels,
        #     out_channels,
        #     kernel=[],
        # )
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=(3, 3), rngs=rngs)
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=(1, 1), rngs=rngs)
            if in_channels != out_channels
            else None
        )
        self.attn_qkv = (
            MPConv(out_channels, out_channels * 3, kernel=(1, 1), rngs=rngs)
            if self.num_heads != 0
            else None
        )
        self.attn_proj = (
            MPConv(out_channels, out_channels, kernel=(1, 1), rngs=rngs)
            if self.num_heads != 0
            else None
        )

    def __call__(self, x: jax.Array, emb: jax.Array) -> jax.Array:
        # Main branch
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, axis=1)  # pixel norm
