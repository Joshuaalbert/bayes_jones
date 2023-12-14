from typing import NamedTuple, Literal, Union

import jax
import tensorflow_probability.substrates.jax as tfp
from jax import numpy as jnp, tree_map
from jaxns import Prior, Model, DefaultNestedSampler, TerminationCondition, summary, plot_diagnostics, plot_cornerplot, \
    marginalise_static
from jaxns.internals.types import NestedSamplerResults

from bayes_jones.common import TEC_CONV, CLOCK_CONV, wrapped_diff
from bayes_jones.types import FloatArray, PRNGKey, float_type

tfpd = tfp.distributions


class GaussianParameters(NamedTuple):
    dtec: FloatArray  # mTECU
    clock: FloatArray  # ns
    const: FloatArray  # rad
    uncert: FloatArray  # rad


class StudentTParameters(NamedTuple):
    dtec: FloatArray  # mTECU
    clock: FloatArray  # ns
    const: FloatArray  # rad
    uncert: FloatArray  # rad
    df: FloatArray  #


class ParameterEstimator:
    """
    Estimates DTEC, clock and constant terms from a set of phase measurements over frequency.
    """

    def __init__(self, freqs: FloatArray, noise_model: Literal['gaussian', 'student-t'] = 'gaussian'):
        self.freqs = freqs
        self.noise_model = noise_model
        self._infer_compiled = jax.jit(self._infer).lower(jax.random.PRNGKey(0), jnp.zeros_like(freqs)).compile()

    def infer(self, key: PRNGKey, phase_obs: FloatArray) -> Union[GaussianParameters, StudentTParameters]:
        infer_key, marginalise_key = jax.random.split(key)
        results: NestedSamplerResults = self._infer_compiled(infer_key, phase_obs)

        def trim(x):
            if x.size > 1:
                return x[:results.total_num_samples]
            return x

        results = tree_map(trim, results)

        summary(results)
        plot_diagnostics(results)
        plot_cornerplot(results)

        if self.noise_model == 'gaussian':
            def to_params(dtec, clock, const, uncert) -> GaussianParameters:
                return GaussianParameters(dtec=dtec, clock=clock, const=const, uncert=uncert)
        elif self.noise_model == 'student-t':
            def to_params(dtec, clock, const, uncert, df) -> StudentTParameters:
                return StudentTParameters(dtec=dtec, clock=clock, const=const, uncert=uncert, df=df)
        else:
            raise ValueError(f"Unknown noise model {self.noise_model}")

        return marginalise_static(
            key=marginalise_key,
            samples=results.samples,
            log_weights=results.log_dp_mean,
            ESS=1000,
            fun=to_params
        )

    def _infer(self, key: PRNGKey, phase_obs: FloatArray) -> NestedSamplerResults:
        """
        Infer DTEC, clock and constant terms from a set of phase measurements over frequency.

        Args:
            phase_obs: phase measurements (rad)
            uncert: phase measurement uncertainties (rad)

        Returns:
            Parameters
        """
        if self.noise_model == 'gaussian':
            def log_likelihood(dtec, clock, const, uncert):
                phase = dtec * (TEC_CONV / self.freqs) + clock * (CLOCK_CONV * self.freqs) + const
                dist = tfpd.Normal(loc=jnp.asarray(0., float_type), scale=uncert)
                return jnp.sum(dist.log_prob(wrapped_diff(phase_obs, phase)))

            def prior_model():
                dtec = yield Prior(tfpd.Cauchy(0., 100.), name='dtec')  # mTECU
                const = yield Prior(tfpd.Uniform(-jnp.pi, jnp.pi), name='const')  # rad
                clock = yield Prior(tfpd.Uniform(-2., 2.), name='clock')  # ns
                uncert = yield Prior(tfpd.HalfNormal(0.5), name='uncert')  # rad
                return dtec, clock, const, uncert
        elif self.noise_model == 'student-t':
            def log_likelihood(dtec, clock, const, uncert, df):
                phase = dtec * (TEC_CONV / self.freqs) + clock * (CLOCK_CONV * self.freqs) + const
                dist = tfpd.StudentT(loc=jnp.asarray(0., float_type), scale=uncert, df=df)
                return jnp.sum(dist.log_prob(wrapped_diff(phase_obs, phase)))

            def prior_model():
                dtec = yield Prior(tfpd.Cauchy(0., 100.), name='dtec')  # mTECU
                const = yield Prior(tfpd.Uniform(-jnp.pi, jnp.pi), name='const')  # rad
                clock = yield Prior(tfpd.Uniform(-2., 2.), name='clock')  # ns
                uncert = yield Prior(tfpd.HalfNormal(0.5), name='uncert')  # rad
                df = yield Prior(tfpd.Uniform(1., 10.), name='df')  # rad
                return dtec, clock, const, uncert, df
        else:
            raise ValueError(f"Unknown noise model {self.noise_model}")

        model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
        ns = DefaultNestedSampler(model=model, num_live_points=model.U_ndims * 64 * 3, max_samples=1e5)

        termination_reason, state = ns(key, term_cond=TerminationCondition())
        result = ns.to_results(termination_reason=termination_reason, state=state, trim=False)
        return result
