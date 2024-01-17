import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp

from bayes_jones.common import wrap, TEC_CONV, CLOCK_CONV

tfpd = tfp.distributions


@pytest.fixture(scope='package')
def basic_jones_data():
    key = random.PRNGKey(0)
    freqs = jnp.linspace(121, 166, 24)  # MHz
    tec = 90.  # mTECU
    const = 2.  # rad
    clock = 0.5  # ns
    uncert = 0.1
    phase = wrap(tec * (TEC_CONV / freqs) + clock * (CLOCK_CONV * freqs) + const)
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)
    phase_obs = jnp.arctan2(Y_obs[..., freqs.size:], Y_obs[..., :freqs.size])
    return phase, phase_obs, freqs, (tec, clock, const, uncert)
