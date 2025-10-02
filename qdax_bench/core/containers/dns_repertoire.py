# dns_repertoire.py

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)

class DNSRepertoire(MapElitesRepertoire):
    """
    A repertoire for Dominated Novelty Search (DNS).

    This repertoire implements the selection mechanism of Dominated Novelty
    Search, which is a Quality-Diversity algorithm that uses a dynamic
    fitness transformation to guide the search. The selection is based on the
    "dominated novelty score", which is the average distance to the k-nearest
    fitter solutions in the descriptor space.

    Args:
        k: The number of nearest-fitter neighbors to consider for the
           dominated novelty score.
    """

    k: int = struct.field(pytree_node=False)

    @classmethod
    def init_default(
        cls,
        genotype: Genotype,
        centroids: Centroid,
        one_extra_score: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> DNSRepertoire:
        """Initialize a Map-Elites repertoire with an initial population of
        genotypes. Requires the definition of centroids that can be computed
        with any method such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so
        it can be called easily called from other modules.

        Args:
            genotype: the typical genotype that will be stored.
            centroids: the centroids of the repertoire
            keys_extra_scores: keys of the extra scores to store in the repertoire

        Returns:
            A repertoire filled with default values.
        """
        if one_extra_score is None:
            one_extra_score = {}

        one_extra_score = {
            key: value
            for key, value in one_extra_score.items()
            if key in keys_extra_scores
        }

        # get number of centroids
        num_centroids = centroids.shape[0]

        # default fitness is -inf
        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_centroids, 1))

        # default genotypes is all 0
        default_genotypes = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            genotype,
        )

        # default descriptor is all zeros
        default_descriptors = jnp.zeros_like(centroids)

        # default extra scores is empty dict
        default_extra_scores = jax.tree.map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
            one_extra_score,
        )

        return cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=default_descriptors,
            extra_scores=default_extra_scores,
            keys_extra_scores=keys_extra_scores,
            k=-jnp.inf,  # Dummy value, will be set properly in init
        )
    
    @classmethod
    def init(  # type: ignore
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        *args,
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
        k: int = 10,
        **kwargs,
    ) -> DNSRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            extra_scores: extra scores of the initial genotypes
            keys_extra_scores: keys of the extra scores to store in the repertoire
            k: The number of nearest-fitter neighbors to consider for the
               dominated novelty score.

        Returns:
            an initialized MAP-Elite repertoire
        """
        repertoire = super().init(
            genotypes,
            fitnesses,
            descriptors,
            centroids,
            *args,
            extra_scores=extra_scores,
            keys_extra_scores=keys_extra_scores,
            **kwargs,
        )
        return repertoire.replace(
            k=k,
            centroids=repertoire.descriptors,  # Use descriptors as centroids for plotting
        )


    def sample(self, random_key: RNGKey, num_samples: int) -> tuple[Genotype, RNGKey]:
        """
        Sample a batch of genotypes from the repertoire.

        Args:
            random_key: A JAX random key.
            num_samples: The number of genotypes to sample.

        Returns:
            A tuple containing the sampled genotypes and a new random key.
        """
        repertoire_empty = self.fitnesses == -jnp.inf
        p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)

        random_key, subkey = jax.random.split(random_key)
        samples = jax.random.choice(
            subkey,
            self.genotypes,
            shape=(num_samples,),
            p=p,
        )
        return samples, random_key
    
    def get_survivors_indices(  
        self,
        candidate_fitnesses: Fitness,
        candidate_descriptors: Descriptor,
    ) -> jnp.ndarray:
        """
        Get the indices of the individuals that survive the DNS selection.

        Args:
            candidate_fitnesses: The fitnesses of the candidate individuals.
            candidate_descriptors: The descriptors of the candidate individuals.

        Returns:
            The indices of the individuals that survive the DNS selection.
        """
        population_size = self.fitnesses.shape[0]

        # 1. Calculate dominated novelty scores for the combined population
        competition_fitnesses = self._calculate_dominated_novelty_scores(
            candidate_fitnesses, candidate_descriptors
        )

        # 2. Select the top N individuals based on the competition score
        indices = jnp.argsort(competition_fitnesses)[::-1]
        survivor_indices = indices[:population_size]

        return survivor_indices

    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores | None = None,
    ) -> DNSRepertoire:
        """
        Add a batch of individuals to the repertoire and apply DNS selection.
        This implementation assumes the repertoire is always full.

        Args:
            batch_of_genotypes: The genotypes of the individuals to add.
            batch_of_descriptors: The descriptors of the individuals to add.
            batch_of_fitnesses: The fitnesses of the individuals to add.
            batch_of_extra_scores: Additional scores, not used in DNS.

        Returns:
            The updated repertoire.
        """
        # Ensure incoming fitnesses have the correct shape (batch_size, 1)
        batch_of_fitnesses = jnp.reshape(
            batch_of_fitnesses, (batch_of_fitnesses.shape[0], 1)
        )
    
        # 1. Combine the current repertoire with the new batch
        candidate_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            self.genotypes,
            batch_of_genotypes,
        )
        candidate_fitnesses = jnp.concatenate(
            [self.fitnesses, batch_of_fitnesses], axis=0
        )
        candidate_descriptors = jnp.concatenate(
            [self.descriptors, batch_of_descriptors], axis=0
        )

        # 2. Select the top N individuals based on the competition score
        survivor_indices = self.get_survivors_indices(
            candidate_fitnesses, candidate_descriptors
        )

        # 3. Create the new repertoire with the survivors
        new_genotypes = jax.tree_util.tree_map(
            lambda x: x[survivor_indices], candidate_genotypes
        )
        new_fitnesses = candidate_fitnesses[survivor_indices]
        new_descriptors = candidate_descriptors[survivor_indices]

        # 4. Update the repertoire state
        return DNSRepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=new_descriptors,  # Use descriptors as centroids for plotting
            k=self.k,
            extra_scores=self.extra_scores,  # Extra scores are not updated in DNS
            keys_extra_scores=self.keys_extra_scores,
        )

    def _calculate_dominated_novelty_scores(
        self, fitnesses: Fitness, descriptors: Descriptor
    ) -> jnp.ndarray:
        """
        Calculate the dominated novelty score for all individuals in a batch.
        This version is memory-efficient and JIT-compatible.
        """
        # Get the size of the candidate population
        num_candidates = descriptors.shape[0]

        # 1. Compute the full pairwise distance matrix (N, N)
        # Using broadcasting to stay efficient: (N, 1, D) - (1, N, D) -> (N, N, D)
        diffs = descriptors[:, None, :] - descriptors[None, :, :]
        distance_matrix = jnp.linalg.norm(diffs, axis=-1)
        # print("Distance matrix:", distance_matrix.shape)

        # 2. Compute the fitter mask (N, N)
        # fitter_mask[i, j] is True if individual j is fitter than individual i
        fitter_mask = fitnesses[None, :] > fitnesses[:, None]
        # print("Fitter mask:", fitter_mask.shape)
        # Remove 1 dimension 
        fitter_mask = jnp.squeeze(fitter_mask, axis=-1)
        # print("Fitter mask after squeeze:", fitter_mask.shape)

        # 3. Mask distances: keep only distances to fitter individuals, others are inf
        # This correctly handles the self-distance (i vs i) as fitness[i] > fitness[i] is False
        masked_distances = jnp.where(fitter_mask, distance_matrix, jnp.inf)

        # Mask self-distances to avoid zero distances
        masked_distances = masked_distances.at[jnp.arange(num_candidates), jnp.arange(num_candidates)].set(jnp.inf)

        # 4. Sort distances for each individual
        # jnp.sort is efficient and moves all jnp.inf values to the end
        sorted_distances = jnp.sort(masked_distances, axis=1)

        # 5. Calculate number of fitter individuals for each candidate
        num_fitter = jnp.sum(fitter_mask, axis=1)

        # Determine how many neighbors to average (k or num_fitter, whichever is smaller)
        k_nearest = jnp.minimum(self.k, num_fitter).astype(int)

        # 6. Create a mask to select the top k_nearest distances for each row
        # This avoids slicing with a dynamic index, which is not JIT-friendly
        range_matrix = jnp.arange(num_candidates)[None, :]  # Shape: (1, N)
        mean_mask = range_matrix < k_nearest[:, None]      # Shape: (N, N)

        # 7. Calculate the sum of the top k distances for each row
        # Apply the mask, zeroing out distances we don't want to include
        masked_sorted_distances = jnp.where(mean_mask, sorted_distances, 0)
        sum_of_distances = jnp.sum(masked_sorted_distances, axis=1)

        # 8. Compute the mean, avoiding division by zero
        # If k_nearest is 0, the score should be inf, which we handle next
        mean_scores = sum_of_distances / jnp.maximum(1, k_nearest)

        # 9. Final scores: if an individual has no fitter neighbors, its score is inf
        final_scores = jnp.where(num_fitter > 0, mean_scores, jnp.inf)

        # 10. Remove all points where fitness is -inf (empty slots in the repertoire)
        empty_mask = fitnesses == -jnp.inf
        empty_mask = jnp.squeeze(empty_mask, axis=-1)
        final_scores = jnp.where(empty_mask, -jnp.inf, final_scores)

        return final_scores