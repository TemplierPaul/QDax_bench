"""Blackbox Optimization Benchmarking Task."""
from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from qdax_bench.tasks.bbob.bbob_fns import bbob_fns
from qdax_bench.tasks.bbob.bbob_noise import NoiseModel, NoiseParams


def compute_descriptor_bounds(
    x_range: Tuple[float, float],
    num_dims: int,
    descriptor_size: int,
    n_sigma: float = 3.0,
    projection: str = "gaussian_random_projection",
) -> Tuple[float, float]:
    """Compute probabilistic bounds on a Gaussian-projected descriptor value.

    Args:
        x_range: Tuple (a, b), the min and max possible values for each input dimension.
        num_dims: Number of input dimensions.
        descriptor_size: Size of the projected descriptor (number of output dims).
        n_sigma: Number of standard deviations to include in the bound (default: 3 for ~99%).

    Returns:
        A tuple (lower_bound, upper_bound) on the descriptor value.
    """
    if projection == "gaussian_random_projection": 
        a, b = x_range
        max_x_sq = jnp.maximum(a ** 2, b ** 2)
        variance = (num_dims * max_x_sq) / descriptor_size
        std_dev = jnp.sqrt(variance)
        bound = n_sigma * std_dev
        return -bound, bound
    elif projection == "random_index":
        # For random index, the descriptor is a one-hot vector, so bounds are fixed
        return x_range[0], x_range[1]
    else:
        raise ValueError(f"Unsupported projection type: {projection}. "
                         "Supported types: 'gaussian_random_projection', 'random_index'.")

@dataclass
class BBOBParams:
    num_dims: jax.Array
    x_opt: jax.Array
    f_opt: jax.Array
    descriptor_params: jax.Array
    noise_params: NoiseParams
    R: jax.Array
    Q: jax.Array


class BBOBTask:
    """Blackbox Optimization Benchmarking Task class."""

    def __init__(
        self,
        num_dims: int = 2,
        descriptor: str = "gaussian_random_projection",
        descriptor_size: int = 2,
        fn_name: str = "sphere",
        task_seed: int = 0,
        x_range: list[float] = [-5.0, 5.0],
        x_opt_range: list[float] = [-4.0, 4.0],
        f_opt_range: list[float] = [0.0, 0.0],
        clip_x: bool = False,
        sample_rotation: bool = False,
        noise_model: str = "noiseless", # "noiseless", "gaussian", "uniform", "cauchy", "additive"
        noise_stabilization: bool = True, 
    ):
        self.num_dims = num_dims
        self.x_range = x_range
        self.x_opt_range = x_opt_range
        self.f_opt_range = f_opt_range
        self.clip_x = clip_x
        self.sample_rotation = sample_rotation
        self.task_seed = task_seed

        fn = bbob_fns.get(fn_name, None)
        if fn is None:
            raise ValueError(
                f"BBOB function '{fn_name}' is not supported. "
                "Available functions: " + ", ".join(bbob_fns.keys())
            )
        self.fn = jax.vmap(fn, in_axes=(0, None, None, None, None))
    
        # Descriptor
        self.descriptor = descriptor
        self._descriptor_size = descriptor_size

        # Noise
        self.noise_model = NoiseModel(
            noise_model_name=noise_model,
            use_stabilization=noise_stabilization
        )

    def get_bbob_params(self, key=None) -> BBOBParams:
        """Sample BBOB task parameters."""
        if key is None:
            key = jax.random.PRNGKey(self.task_seed)
        key_x, key_f, key_noise, key_desc = jax.random.split(key, 4)

        x_opt = jax.random.uniform(
            key_x,
            shape=(self.num_dims,),
            minval=self.x_opt_range[0],
            maxval=self.x_opt_range[1],
        )
        f_opt = jax.random.uniform(
            key_f,
            minval=self.f_opt_range[0],
            maxval=self.f_opt_range[1],
        )

        jax.debug.print(
            "BBOB Task Parameters: x_opt={x_opt}, f_opt={f_opt}",
            x_opt=x_opt,
            f_opt=f_opt,
        )

        # Sample noise model parameters
        noise_params = self.noise_model.sample_params(key_noise)

        # Descriptor
        if self.descriptor == "gaussian_random_projection":
            descriptor_params = self.gaussian_random_projection(key_desc)
        elif self.descriptor == "random_index":
            descriptor_params = self.random_index(key_desc)
        else:
            raise NotImplementedError

        # Optimal descriptor value
        d_opt = descriptor_params @ x_opt
        jax.debug.print(
            "BBOB Task Descriptor: d_opt={d_opt}",
            d_opt=d_opt,
        )
        
        if self.sample_rotation:
            key_r, key_q = jax.random.split(key)
            R = self.generate_random_rotation(key_r, self.num_dims)
            Q = self.generate_random_rotation(key_q, self.num_dims)
        else:
            R = jnp.eye(self.num_dims)
            Q = jnp.eye(self.num_dims)

        return BBOBParams(self.num_dims, x_opt, f_opt, descriptor_params, noise_params, R, Q)

    def scoring_function(
        self,
        x: jax.Array,
        key: jax.Array,
        task_params: BBOBParams,
    ) -> tuple[Fitness, Descriptor, ExtraScores]:
        if self.clip_x:
            x = jnp.clip(x, self.x_range[0], self.x_range[1])

        fn_val, fn_pen = self.fn(
            x,
            task_params.x_opt,
            task_params.R,
            task_params.Q,
            task_params.num_dims,
        )

        # Apply noise
        fn_noise = self.noise_model.apply(key, fn_val, task_params.noise_params)

        # Add boundary handling penalty and optimal function value
        fn_val = fn_noise + fn_pen + task_params.f_opt

        # Descriptor
        descriptor = jax.vmap(lambda x_i: task_params.descriptor_params @ x_i)(x)

        return -fn_val, descriptor, {}

    def sample_x(self, key: jax.Array) -> jax.Array:
        return jax.random.uniform(
            key,
            shape=(self.num_dims,),
            minval=self.x_range[0],
            maxval=self.x_range[1],
        )

    @property
    def descriptor_size(self):
        return self._descriptor_size

    def generate_random_rotation(self, key: jax.Array, num_dims: int) -> jax.Array:
        """Generate a random (n, n) rotation matrix uniformly sampled from SO(n).

        This implementation follows the method described in:
        "How to generate a random unitary matrix" [Maris Ozols 2006]
        http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
        https://github.com/alecjacobson/gptoolbox/blob/master/matrix/rand_rotation.m

        Uses a fixed-size matrix of num_dims and masks the extra dimensions to handle
        variable num_dims while remaining jit-compatible.
        """
        # Generate fixed-size random normal matrix but mask based on num_dims
        random_matrix = jax.random.normal(key, (self.num_dims, self.num_dims))
        mask = (jnp.arange(self.num_dims)[:, None] < num_dims) & (jnp.arange(self.num_dims)[None, :] < num_dims)
        random_matrix = jnp.where(mask, random_matrix, 0.0)

        # Add identity matrix for masked region to ensure valid QR decomposition
        random_matrix = random_matrix + jnp.where(~mask, jnp.eye(self.num_dims), 0.0)

        # QR decomposition
        orthogonal_matrix, upper_triangular = jnp.linalg.qr(random_matrix)

        # Extract diagonal and create sign correction matrix
        diagonal = jnp.diag(upper_triangular)
        sign_correction = jnp.diag(diagonal / jnp.abs(diagonal))

        # Apply sign correction
        rotation = orthogonal_matrix @ sign_correction

        # Ensure determinant is 1 by possibly flipping first row
        determinant = jnp.linalg.det(rotation)
        rotation = rotation.at[0].multiply(determinant)

        return rotation

    def random_index(self, key: jax.Array) -> jax.Array:
        descriptor_ids = jax.random.choice(
            key,
            jnp.arange(self.num_dims),
            shape=(self.descriptor_size,),
            replace=False,
        )
        return jax.nn.one_hot(descriptor_ids, num_classes=self.num_dims)

    def gaussian_random_projection(self, key: jax.Array) -> jax.Array:
        return jax.random.normal(
            key,
            shape=(self.descriptor_size, self.num_dims),
        ) / jnp.sqrt(self.descriptor_size)

