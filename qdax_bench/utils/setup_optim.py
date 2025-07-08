import functools
import warnings
import jax
import jax.numpy as jnp

import hydra 

from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.core.containers.mapelites_repertoire import compute_cvt_centroids

def setup_qd(config):
    key = jax.random.PRNGKey(config["seed"])
    key, subkey = jax.random.split(key)

    scoring_fn = hydra.utils.instantiate(
        config["task"]["scoring_function"],
    )

    init_batch_size = config["algo"]["initial_batch"]
    num_param_dimensions = config["task"]["search_space"]["n_dimensions"]
    min_param = config["task"]["search_space"]["minval"]
    max_param = config["task"]["search_space"]["maxval"]

    init_variables_func = functools.partial(
        jax.random.uniform,
        shape=(init_batch_size, num_param_dimensions),
        minval=min_param,
        maxval=max_param,
    )

    metrics_function = functools.partial(
        default_qd_metrics,
        qd_offset=config["task"]["qd_offset"],
    )

    # Compute the centroids
    key, subkey = jax.random.split(key)
    min_bd = config["task"]["descriptors"]["minval"]
    max_bd = config["task"]["descriptors"]["maxval"]
    centroids = compute_cvt_centroids(
        num_descriptors=config["task"]["descriptors"]["n_dimensions"],
        num_init_cvt_samples=config["algo"]["archive"]["num_init_cvt_samples"],
        num_centroids=config["algo"]["archive"]["num_centroids"],
        minval=min_bd,
        maxval=max_bd,
        key=subkey,
    )

    return centroids, min_bd, max_bd, scoring_fn, metrics_function, init_variables_func, key
