import jax.numpy as jnp

from bayes_jones.types import FloatArray

TEC_CONV = -8.4479745  # rad*MHz/mTECU
CLOCK_CONV = (2e-3 * jnp.pi)  # rad/MHz/ns


def wrap(phi: FloatArray) -> FloatArray:
    """
    Wraps phase to [-pi, pi).

    Args:
        phi: phase (rad)

    Returns:
        wrapped phase (rad)
    """
    return (phi + jnp.pi) % (2 * jnp.pi) - jnp.pi


def wrapped_diff(phi1: FloatArray, phi2: FloatArray) -> FloatArray:
    """
    Returns wrapped difference.

    Args:
        phi1: phase (rad)
        phi2: phase (rad)

    Returns:
        wrapped difference (rad)
    """
    return wrap(wrap(phi1) - wrap(phi2))
