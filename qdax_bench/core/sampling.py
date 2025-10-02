"""
Core components of the MAP-Elites-sampling algorithm.
From https://github.dev/adaptive-intelligent-robotics/Uncertain_Quality_Diversity
"""

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

@jax.jit
def average(quantities: jnp.ndarray) -> jnp.ndarray:
    """Default expectation extractor using average."""
    return jnp.nanmean(quantities, axis=1)


@jax.jit
def median(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative expectation extractor using median.
    More robust to outliers than average."""
    return jnp.nanmedian(quantities, axis=1)


@jax.jit
def mode(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative expectation extractor using mode.
    More robust to outliers than average.
    WARNING: for multidimensional objects such as descriptor, do
    dimension-wise selection.
    """

    def _mode(quantity: jnp.ndarray) -> jnp.ndarray:

        # Ensure correct dimensions for both single and multi-dimension
        quantity = jnp.reshape(quantity, (quantity.shape[0], -1))

        # Dimension-wise voting in case of multi-dimension
        def _dim_mode(dim_quantity: jnp.ndarray) -> jnp.ndarray:
            unique_vals, counts = jnp.unique(
                dim_quantity, return_counts=True, size=dim_quantity.size
            )
            return unique_vals[jnp.argmax(counts)]

        # vmap over dimensions
        return jnp.squeeze(jax.vmap(_dim_mode)(jnp.transpose(quantity)))

    # vmap over individuals
    return jax.vmap(_mode)(quantities)


@jax.jit
def closest(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative expectation extractor selecting individual
    that has the minimum distance to all other individuals. This
    is an approximation of the geometric median.
    More robust to outliers than average."""

    def _closest(values: jnp.ndarray) -> jnp.ndarray:
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.nansum(jnp.square(x - y)))

        distances = jax.vmap(
            jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
        )(values, values)
        return values[jnp.nanargmin(jnp.nanmean(distances, axis=0))]

    return jax.vmap(_closest)(quantities)


@jax.jit
def std(quantities: jnp.ndarray) -> jnp.ndarray:
    """Default reproducibility extractor using standard deviation."""
    return jnp.nanstd(quantities, axis=1)


@jax.jit
def mad(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative reproducibility extractor using Median Absolute Deviation.
    More robust to outliers than standard deviation."""
    num_samples = quantities.shape[1]
    median = jnp.repeat(
        jnp.nanmedian(quantities, axis=1, keepdims=True), num_samples, axis=1
    )
    return jnp.nanmedian(jnp.abs(quantities - median), axis=1)


@jax.jit
def iqr(quantities: jnp.ndarray) -> jnp.ndarray:
    """Alternative reproducibility extractor using Inter-Quartile Range.
    More robust to outliers than standard deviation."""
    q1 = jnp.nanquantile(quantities, 0.25, axis=1)
    q4 = jnp.nanquantile(quantities, 0.75, axis=1)
    return q4 - q1

def multi_sample_scoring_function(
    policies_params: Genotype,
    random_key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Wrap scoring_function to perform sampling.

    This function returns the fitnesses, descriptors, and extra_scores computed
    over num_samples evaluations with the scoring_fn.

    Args:
        policies_params: policies to evaluate
        random_key: JAX random key
        scoring_fn: scoring function used for evaluation
        num_samples: number of samples to generate for each individual

    Returns:
        (n, num_samples) array of fitnesses,
        (n, num_samples, num_descriptors) array of descriptors,
        dict with num_samples extra_scores per individual,
        JAX random key
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_samples)

    # evaluate
    sample_scoring_fn = jax.vmap(
        scoring_fn,
        # vectorizing over axis 0 vectorizes over the num_samples random keys
        in_axes=(None, 0),
        # indicates that the vectorized axis will become axis 1, i.e., the final
        # output is shape (batch_size, num_samples, ...)
        out_axes=1,
    )
    all_fitnesses, all_descriptors, all_extra_scores = sample_scoring_fn(
        policies_params, keys
    )

    return all_fitnesses, all_descriptors, all_extra_scores