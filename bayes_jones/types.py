import chex
import jax.numpy as jnp

float_type = jnp.result_type(float)
int_type = jnp.result_type(int)
complex_type = jnp.result_type(complex)

PRNGKey = chex.PRNGKey
FloatArray = chex.Array
IntArray = chex.Array
BoolArray = chex.Array
