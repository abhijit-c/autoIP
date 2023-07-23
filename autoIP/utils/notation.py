from typing import Any, Dict, Generic, Tuple, TypeVar, Union, Callable
from jax import Array
from jax.typing import ArrayLike

Operator = Callable[[ArrayLike], Array]
LinearOperator = Operator

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

MVN = tfd.MultivariateNormalLinearOperator
