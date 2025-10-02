# dns_repertoire.py

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)

from qdax_bench.core.containers.dns_repertoire import DNSRepertoire

class ExtractDNSRepertoire(DNSRepertoire):
    """
    A repertoire for Extract Dominated Novelty Search (E-DNS).

    This repertoire implements the selection mechanism of Dominated Novelty
    Search, which is a Quality-Diversity algorithm that uses a dynamic
    fitness transformation to guide the search. The selection is based on the
    "dominated novelty score", which is the average distance to the k-nearest
    fitter solutions in the descriptor space.
    We combine it with Extract-ME for uncertainty handling.
    """
    fitnesses_all: Fitness
    descriptors_all: Descriptor

    def add_all_evals(
        self,
        batch_of_genotypes: Genotype,
        batch_of_all_descriptors: Descriptor,
        batch_of_all_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> ExtractDNSRepertoire:
        """
        Add a batch of elements to the repertoire.
        WARNING: This addition makes the hypothesis that batch_of_all_descriptors
        and batch_of_all_fitnesses are already dimensions num_evals.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes. Its shape is (batch_size,)

        Returns:
            The updated DNS repertoire.
        """

        # 1. Combine the current repertoire with the new batch
        candidate_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            self.genotypes,
            batch_of_genotypes,
        )

        candidate_descriptors_all = jnp.concatenate(
            [self.descriptors_all, batch_of_all_descriptors], axis=0
        )   # shape (N + B, num_evals, desc_dim)
        candidate_fitnesses_all = jnp.concatenate(
            [self.fitnesses_all, batch_of_all_fitnesses], axis=0
        )   # shape (N + B, num_evals)
        candidate_descriptors = descriptor_extractor(candidate_descriptors_all)  # shape (N + B, desc_dim)
        candidate_fitnesses = fitness_extractor(candidate_fitnesses_all)  # shape (N + B,)
        # Replace nan fitness by -inf to avoid selecting them as 
        candidate_fitnesses = jnp.where(jnp.isnan(candidate_fitnesses), -jnp.inf, candidate_fitnesses)
        candidate_fitnesses = jnp.reshape(
            candidate_fitnesses, (candidate_fitnesses.shape[0], 1)
        )

        # print("in add_all_evals, candidate_descriptors", candidate_descriptors.shape)
        # print("in add_all_evals, candidate_fitnesses", candidate_fitnesses.shape)

        # 3. Select the top N individuals based on the competition score
        survivor_indices = self.get_survivors_indices(
            candidate_fitnesses, candidate_descriptors
        )

        # 4. Create the new repertoire with the survivors
        new_genotypes = jax.tree_util.tree_map(
            lambda x: x[survivor_indices],
            candidate_genotypes,
        )
        new_fitnesses_all = candidate_fitnesses_all[survivor_indices]
        new_descriptors_all = candidate_descriptors_all[survivor_indices]
        new_fitnesses = candidate_fitnesses[survivor_indices]
        new_descriptors = candidate_descriptors[survivor_indices]

        print("in add_all_evals, new_descriptors", new_descriptors.shape)
        print("in add_all_evals, new_fitnesses", new_fitnesses.shape)
        print("in add_all_evals, new_descriptors_all", new_descriptors_all.shape)
        print("in add_all_evals, new_fitnesses_all", new_fitnesses_all.shape)

        return self.replace(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            fitnesses_all=new_fitnesses_all,
            descriptors_all=new_descriptors_all,
            centroids=new_descriptors,  # Using descriptors as centroids for plotting
        )
    
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        k: int,
        num_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> ExtractDNSRepertoire:
        """
        Initialize a DNS repertoire with an initial population of genotypes.
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
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            extra_scores: unused extra_scores of the initial genotypes
            k: number of neighbors to consider for the novelty score
            num_evals: number of evaluations per individual
            fitness_extractor: function that takes as input an array of shape
                (batch_size, num_evals) and returns an array of shape
                (batch_size,) containing the aggregated fitness of each individual.
            descriptor_extractor: function that takes as input an array of shape
                (batch_size, num_evals, num_descriptors) and returns an array of shape
                (batch_size, num_descriptors) containing the aggregated descriptor of each
                individual.

        Returns:
            an initialized DNS repertoire
        """
        # print(f"Init: fitness {fitnesses.shape}, descriptors {descriptors.shape}, centroids {centroids.shape}")
        if extra_scores is None:
            extra_scores = {}

        extra_scores = {
            key: value
            for key, value in extra_scores.items()
            if key in keys_extra_scores
        }

        num_centroids = centroids.shape[0]

        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )

        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_centroids, 1))
        default_fitnesses_all = jnp.nan * jnp.ones(
            shape=(num_centroids, num_evals)
        )

        default_descriptors = jnp.nan * jnp.zeros(shape=(num_centroids, descriptors.shape[-1]))
        default_descriptors_all = jnp.nan * jnp.ones(
            shape=(num_centroids, num_evals, descriptors.shape[-1])
        )

        repertoire = cls(
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            fitnesses_all=default_fitnesses_all,
            descriptors_all=default_descriptors_all,
            centroids=centroids,
            extra_scores=extra_scores,
            keys_extra_scores=keys_extra_scores,
            k=k,
        )

        # print("in init", repertoire.genotypes.shape)
        # print("in init", repertoire.descriptors.shape)
        # print("in init", repertoire.descriptors_all.shape)
        # print("in init", repertoire.fitnesses.shape)
        # print("in init", repertoire.fitnesses_all.shape)

        new_repertoire = repertoire.add_all_evals(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=fitness_extractor,
            descriptor_extractor=descriptor_extractor,
        )
        return new_repertoire # type: ignore
    
    def erase_extracted(
        self,
        indices: jnp.ndarray,
    ) -> ExtractDNSRepertoire:
        """
        Erase the individuals at the given indices from the repertoire.
        The erased individuals are replaced by default values.

        Args:
            indices: indices of the individuals to be erased
        Returns:
            The updated repertoire with the individuals at the given indices erased.
        """
        default_genotype = jax.tree_map(
            lambda x: jnp.zeros_like(x[0]), self.genotypes
        )
        default_fitness = -jnp.inf * jnp.ones((1,))
        default_fitness_all = jnp.nan * jnp.ones((self.fitnesses_all.shape[1],))
        default_descriptor = jnp.nan * jnp.zeros((self.descriptors.shape[1],))
        default_descriptor_all = jnp.nan * jnp.ones(
            (self.descriptors_all.shape[1], self.descriptors_all.shape[2])
        )

        new_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[indices].set(jnp.tile(default_genotype, (len(indices),) + (1,) * (x.ndim - 1))),
            self.genotypes,
        )
        new_fitnesses = self.fitnesses.at[indices].set(
            jnp.tile(default_fitness, (len(indices), 1))
        )
        new_fitnesses_all = self.fitnesses_all.at[indices].set(
            jnp.tile(default_fitness_all, (len(indices), 1))
        )
        new_descriptors = self.descriptors.at[indices].set(
            jnp.tile(default_descriptor, (len(indices), 1))
        )
        new_descriptors_all = self.descriptors_all.at[indices].set(
            jnp.tile(default_descriptor_all, (len(indices), 1, 1))
        )

        return self.replace(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            fitnesses_all=new_fitnesses_all,
            descriptors_all=new_descriptors_all,
        )

    def extract_uniform(
        self,
        key: RNGKey,
        num_samples: int,
    ) -> Tuple[ExtractDNSRepertoire, Genotype, Fitness, Descriptor]:
        """
        Extract num_samples random element from the grid.
        Extract means that they are removed from the grid when sampled.

        Args:
            key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            repertoire: the new repertoire
            extract_genotypes: extracted genotypes
            extract_fitnesses: all fitnesses of the extracted genotypes
            extract_descriptors: all descriptors of the extracted genotypes
            key: an updated jax PRNG random key
        """
        repertoire_size = self.fitnesses.shape[0]

        key, subkey = jax.random.split(key)
        indices = jax.random.choice(
            subkey,
            jnp.arange(repertoire_size),
            shape=(num_samples,),
            replace=False,
        )

        extract_genotypes = jax.tree_util.tree_map(
            lambda x: x[indices], self.genotypes
        )
        extract_fitnesses = self.fitnesses_all[indices]
        extract_descriptors = self.descriptors_all[indices]

        new_repertoire = self.erase_extracted(indices)

        return new_repertoire, extract_genotypes, extract_fitnesses, extract_descriptors