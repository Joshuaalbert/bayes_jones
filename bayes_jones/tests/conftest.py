import pytest
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp
from jaxns import Prior, Model

from bayes_jones.common import wrap, TEC_CONV, CLOCK_CONV, wrapped_diff

tfpd = tfp.distributions


@pytest.fixture(scope='package')
def basic_jones_data():
    key = random.PRNGKey(0)
    freqs = jnp.linspace(121, 166, 24)  # MHz
    tec = 90.  # mTECU
    const = 2.  # rad
    clock = 0.5  # ns
    uncert = 0.05
    phase = wrap(tec * (TEC_CONV / freqs) + clock * (CLOCK_CONV * freqs) + const)
    Y = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    Y_obs = Y + uncert * random.normal(key, shape=Y.shape)
    phase_obs = jnp.arctan2(Y_obs[..., freqs.size:], Y_obs[..., :freqs.size])
    return phase, phase_obs, freqs, (tec, clock, const, uncert)


@pytest.fixture(scope='package')
def basic_jones_model(basic_jones_data):
    phase, phase_obs, freqs, (tec, clock, const, uncert) = basic_jones_data

    def log_likelihood(tec, clock, const, uncert):
        phase = tec * (TEC_CONV / freqs) + const + clock * (CLOCK_CONV * freqs)
        dist = tfpd.MultivariateNormalDiag(scale_diag=uncert)
        return dist.log_prob(wrapped_diff(phase, phase_obs))

    def prior_model():
        tec = yield Prior(tfpd.Cauchy(0., 100.), name='tec')  # mTECU
        const = yield Prior(tfpd.Uniform(-jnp.pi, jnp.pi), name='const')  # rad
        clock = yield Prior(tfpd.Uniform(-2., 2.), name='clock')  # ns
        uncert = yield Prior(tfpd.HalfNormal(0.5), name='uncert')  # rad
        return tec, clock, const, uncert

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    return model
