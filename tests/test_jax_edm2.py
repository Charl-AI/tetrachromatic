import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from flax import nnx

import ml_utils.jax_edm2 as jaxedm
import tests.edm2 as torchedm

torch.manual_seed(42)


@pytest.mark.skip("Not a test, just a helper function")
def j2t(x: jax.Array) -> torch.Tensor:
    """Convert a jax array to pytorch (assumes both are float32)."""
    np_array = np.asarray(x, dtype=np.float32)
    return torch.from_numpy(np_array)


@pytest.mark.skip("Not a test, just a helper function")
def t2j(x: torch.Tensor) -> jax.Array:
    """Convert a pytorch array to jax (assumes both are float32)."""
    np_array = x.to(torch.float32).detach().numpy()
    return jnp.array(np_array)


def test_mp_silu():
    x0 = torch.zeros(10)
    x1 = torch.ones((32, 32))
    x2 = torch.randn(16, 32)
    x3 = torch.randn(16, 3, 28, 28)

    inputs = [x0, x1, x2, x3]
    for input in inputs:
        j = jaxedm.mp_silu(t2j(input))
        t = torchedm.mp_silu(input)
        assert torch.allclose(t, j2t(j), rtol=1e-6)


def test_mp_sum():
    a = torch.randn(16, 32) * 2 + 4
    b = torch.randn(16, 32) * 3 + 1

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        j = jaxedm.mp_sum(t2j(a), t2j(b), t)
        t_ = torchedm.mp_sum(a, b, t)
        assert torch.allclose(t_, j2t(j), rtol=1e-4)


def test_mp_cat():
    a = torch.randn(16, 32) * 2 + 4
    b = torch.randn(16, 32) * 3 + 1

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        j = jaxedm.mp_cat(t2j(a), t2j(b), t=t)
        t_ = torchedm.mp_cat(a, b, t=t)
        assert torch.allclose(t_, j2t(j), rtol=1e-6)


def test_mp_fourier():
    jf = jaxedm.MPFourier(64, rngs=nnx.Rngs(params=0))
    tf = torchedm.MPFourier(64)

    assert jf.freqs.shape == tf.freqs.shape
    assert jf.phases.shape == tf.phases.shape

    # perform surgery on torch params
    tf.freqs = j2t(jf.freqs.value)
    tf.phases = j2t(jf.phases.value)

    t0 = torch.rand(128)
    t1 = torch.rand(32)
    t2 = torch.rand(64)

    for t in [t0, t1, t2]:
        j = jf(t2j(t))
        t_ = tf(t)
        assert torch.allclose(t_, j2t(j), rtol=1e-6)


def test_normalize():
    x0 = torch.zeros(32)
    x1 = torch.ones(32)
    x2 = torch.randn(64)
    x3 = torch.randn(32, 3, 28, 28)

    for x in [x0, x1, x2, x3]:
        j = jaxedm.normalize(t2j(x))
        t = torchedm.normalize(x)
        assert torch.allclose(t, j2t(j), rtol=1e-6)

    j = jaxedm.normalize(t2j(x3), axis=(1, 2, 3))
    t = torchedm.normalize(x3, dim=(1, 2, 3))
    assert torch.allclose(t, j2t(j), rtol=1e-6)

    j = jaxedm.normalize(t2j(x3), axis=(0, 1, 2, 3))
    t = torchedm.normalize(x3, dim=(0, 1, 2, 3))
    assert torch.allclose(t, j2t(j), rtol=1e-5)
