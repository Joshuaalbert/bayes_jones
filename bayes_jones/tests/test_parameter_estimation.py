import time

import jax

from bayes_jones.parameter_estimation import ParameterEstimator


def test_parameter_estimator_gaussian(basic_jones_data):
    phase, phase_obs, freqs, (tec, clock, const, uncert) = basic_jones_data
    estimator = ParameterEstimator(freqs=freqs, noise_model='gaussian')
    t0 = time.time()
    parameters = estimator.infer(key=jax.random.PRNGKey(0), phase_obs=phase_obs)
    parameters.dtec.block_until_ready()
    dt = time.time() - t0
    print(f"Time taken: {dt}")
    print(parameters)

def test_parameter_estimator_student_t(basic_jones_data):
    phase, phase_obs, freqs, (tec, clock, const, uncert) = basic_jones_data
    estimator = ParameterEstimator(freqs=freqs, noise_model='student-t')
    t0 = time.time()
    parameters = estimator.infer(key=jax.random.PRNGKey(0), phase_obs=phase_obs)
    parameters.dtec.block_until_ready()
    dt = time.time() - t0
    print(f"Time taken: {dt}")
    print(parameters)