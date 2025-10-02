from __future__ import annotations

import warnings
from functools import partial
from typing import Callable, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from typing_extensions import TypeAlias

from brax.io import html
import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
    Metrics,
)

from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)

from qdax.utils.plotting import plot_2d_map_elites_repertoire
from qdax.core.emitters.repertoire_selectors.selector import (
    MapElitesRepertoireT,
    Selector,
)

Count: TypeAlias = jnp.ndarray

class ExtractMERepertoire(MapElitesRepertoire):
    """
    Class for a deep repertoire that also stores all past evaluations of each indiv.

    Adapted from https://github.dev/adaptive-intelligent-robotics/Uncertain_Quality_Diversity/core/containers/archive_sampling_repertoire.py
    Args:
        genotypes: a PyTree containing the genotypes of the best solutions ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple Jax array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        genotypes_depth: a PyTree containing all the genotypes ordered by the centroids.
            Each leaf has a shape (num_centroids, depth, num_features). The PyTree
            can be a simple Jax array or a more complex nested structure such as to
            represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of best solutions in each cell of
            the repertoire, ordered by centroids. The array shape is (num_centroids,).
        fitnesses_depth: an array that contains the fitness of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, depth).
        fitnesses_depth_all: an array that contains the fitness of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, depth, num_evals).
        descriptors: an array that contains the descriptors of best solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        descriptors_depth: an array that contains the descriptors of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, depth, num_descriptors).
        descriptors_depth_all: an array that contains the descriptors of all solutions in
            each cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, depth, num_evals, num_descriptors).
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
    """

    genotypes_depth: Genotype
    fitnesses_depth: Fitness
    fitnesses_depth_all: Fitness
    descriptors_depth: Descriptor
    descriptors_depth_all: Descriptor

    def add_all_evals(
        self,
        batch_of_genotypes: Genotype,
        batch_of_all_descriptors: Descriptor,
        batch_of_all_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> ExtractMERepertoire:
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
            The updated MAP-Elites repertoire.
        """

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]
        out_of_bound = max(
            num_centroids * depth,
            batch_of_all_fitnesses.shape[0],
        )

        # Compute batch of fitnesses and descriptor
        batch_of_fitnesses = fitness_extractor(batch_of_all_fitnesses)
        batch_of_fitnesses = jnp.where(
            jnp.isnan(batch_of_fitnesses), -jnp.inf, batch_of_fitnesses
        )
        batch_of_descriptors = descriptor_extractor(batch_of_all_descriptors)

        # Get indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        @jax.jit
        def _add_per_cell(
            cell_idx: jnp.ndarray,
            cell_genotypes_depth: Genotype,
            cell_fitnesses_depth: Fitness,
            cell_fitnesses_depth_all: Fitness,
            cell_descriptors_depth: Descriptor,
            cell_descriptors_depth_all: Descriptor,
        ) -> Tuple[
            Genotype,
            Fitness,
            Fitness,
            Descriptor,
            Descriptor,
            Genotype,
            Fitness,
            Descriptor,
        ]:
            """
            For a given cell with index cell_idx, filter candidate
            indivs for this cell, and add them to it, reordering so
            highest-fitness individuals are first.

            Args:
              cell_idx: cell index
              cell_genotypes_depth: genotype in the cell
              cell_fitnesses_depth: fitnesses in the cell
              cell_fitnesses_depth_all
              cell_descriptors_depth: descriptors in the cell
              cell_descriptors_depth_all

            Returns:
              new_cell_genotypes_depth
              new_cell_fitnesses_depth
              new_cell_fitnesses_depth_all
              new_cell_descriptors_depth
              new_cell_descriptors_depth_all
              new_cell_genotype: genotype in the top layer of the cell
              new_cell_fitnesses: fitnesses in the top layer of the cell
              new_cell_descriptors: descriptors in the top layer of the cell
            """

            # Order existing and candidate indivs by fitness
            candidate_fitnesses = jnp.where(
                batch_of_indices == cell_idx, batch_of_fitnesses, -jnp.inf
            )
            all_fitnesses = jnp.concatenate(
                [cell_fitnesses_depth, candidate_fitnesses],
                axis=0,
            )
            _, final_indices = jax.lax.top_k(all_fitnesses, depth)

            # First, move around existing indivs to follow order
            cell_indices = jnp.where(
                final_indices < depth,
                final_indices,
                out_of_bound,
            )
            new_cell_genotypes_depth = jax.tree_map(
                lambda x: x.at[cell_indices].get(),
                cell_genotypes_depth,
            )
            new_cell_fitnesses_depth = cell_fitnesses_depth.at[cell_indices].get()
            new_cell_fitnesses_depth_all = cell_fitnesses_depth_all.at[
                cell_indices
            ].get()
            new_cell_descriptors_depth = cell_descriptors_depth.at[cell_indices].get()
            new_cell_descriptors_depth_all = cell_descriptors_depth_all.at[
                cell_indices
            ].get()

            # Second, add the candidate indivs
            candidate_indices = jnp.where(
                final_indices >= depth,
                final_indices - depth,
                out_of_bound,
            )
            depth_indices = jnp.where(
                candidate_indices < out_of_bound,
                jnp.arange(0, depth, step=1),
                out_of_bound,
            )
            new_cell_genotypes_depth = jax.tree_map(
                lambda x, y: x.at[depth_indices].set(y[candidate_indices]),
                new_cell_genotypes_depth,
                batch_of_genotypes,
            )
            new_cell_fitnesses_depth = new_cell_fitnesses_depth.at[depth_indices].set(
                batch_of_fitnesses[candidate_indices]
            )
            new_cell_fitnesses_depth_all = new_cell_fitnesses_depth_all.at[
                depth_indices
            ].set(batch_of_all_fitnesses[candidate_indices])
            new_cell_descriptors_depth = new_cell_descriptors_depth.at[
                depth_indices
            ].set(batch_of_descriptors[candidate_indices])
            new_cell_descriptors_depth_all = new_cell_descriptors_depth_all.at[
                depth_indices
            ].set(batch_of_all_descriptors[candidate_indices])

            # Also return the top layer of the grid
            new_cell_genotype = jax.tree_map(
                lambda x: x.at[0].get(),
                new_cell_genotypes_depth,
            )
            new_cell_fitnesses = new_cell_fitnesses_depth.at[0].get()
            new_cell_descriptors = new_cell_descriptors_depth.at[0].get()

            # Return the updated cell
            return (
                new_cell_genotypes_depth,
                new_cell_fitnesses_depth,
                new_cell_fitnesses_depth_all,
                new_cell_descriptors_depth,
                new_cell_descriptors_depth_all,
                new_cell_genotype,
                new_cell_fitnesses,
                new_cell_descriptors,
            )

        # Add individuals cell by cell
        (
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_descriptors_depth,
            new_descriptors_depth_all,
            new_genotype,
            new_fitnesses,
            new_descriptors,
        ) = jax.vmap(_add_per_cell)(
            jnp.arange(0, num_centroids, step=1),
            self.genotypes_depth,
            self.fitnesses_depth,
            self.fitnesses_depth_all,
            self.descriptors_depth,
            self.descriptors_depth_all,
        )

        return self.replace(  # type:ignore
            genotypes=new_genotype,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses.reshape(-1, 1),
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
        )
    
    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        depth: int,
        num_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        extra_scores: Optional[ExtraScores] = None,
        keys_extra_scores: Tuple[str, ...] = (),
    ) -> ExtractMERepertoire:
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
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            extra_scores: unused extra_scores of the initial genotypes
            depth: depth of the repertoire
            num_evals: number of evaluations per individual
            fitness_extractor: function that takes as input an array of shape
                (batch_size, num_evals) and returns an array of shape
                (batch_size,) containing the aggregated fitness of each individual.
            descriptor_extractor: function that takes as input an array of shape
                (batch_size, num_evals, num_descriptors) and returns an array of shape
                (batch_size, num_descriptors) containing the aggregated descriptor of each
                individual.

        Returns:
            an initialized MAP-Elite repertoire
        """
        if extra_scores is None:
            extra_scores = {}

        extra_scores = {
            key: value
            for key, value in extra_scores.items()
            if key in keys_extra_scores
        }

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=(num_centroids, 1))
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids, depth))
        default_fitnesses_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids, depth, num_evals)
        )
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    depth,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth = jnp.zeros(
            shape=(num_centroids, depth, centroids.shape[-1])
        )
        default_descriptors_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids, depth, num_evals, centroids.shape[-1])
        )

        repertoire = ExtractMERepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            fitnesses_depth_all=default_fitnesses_depth_all,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            descriptors_depth_all=default_descriptors_depth_all,
            centroids=centroids,
            extra_scores=extra_scores,
            keys_extra_scores=keys_extra_scores
        )

        # print("in init", repertoire.genotypes.shape)
        # print("in init", repertoire.fitnesses.shape)

        # Add initial values to the grid
        new_repertoire = repertoire.add_all_evals(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=fitness_extractor,
            descriptor_extractor=descriptor_extractor,
        )

        return new_repertoire  # type: ignore

    def empty(self) -> ExtractMERepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty ExtractMERepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_fitnesses_depth = jnp.full_like(self.fitnesses_depth, -jnp.inf)
        new_fitnesses_depth_all = jnp.full_like(self.fitnesses_depth_all, jnp.nan)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_descriptors_depth = jnp.zeros_like(self.descriptors_depth)
        new_descriptors_depth_all = jnp.full_like(self.descriptors_depth_all, jnp.nan)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros_like(x), self.genotypes_depth
        )
        return ExtractMERepertoire(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
            centroids=self.centroids,
        )

    def _order_repertoire(
        self,
        rows: jnp.ndarray,
        cols: jnp.ndarray,
        key: RNGKey,
    ) -> ExtractMERepertoire:
        """
        Re-order repertoire following extraction.

        Args:
            key: a jax PRNG random key
            rows, cols: position of extracted individuals in (centroid, depth) format

        Returns:
            repertoire: the new repertoire
            key: an updated jax PRNG random key
        """

        # Remove extracted individuals from all grids
        new_genotypes_depth = jax.tree_util.tree_map(
            lambda x: x.at[rows, cols].set(0),
            self.genotypes_depth,
        )
        new_fitnesses_depth = self.fitnesses_depth.at[rows, cols].set(-jnp.inf)
        new_fitnesses_depth_all = self.fitnesses_depth_all.at[rows, cols].set(jnp.nan)
        new_descriptors_depth = self.descriptors_depth.at[rows, cols].set(0)
        new_descriptors_depth_all = self.descriptors_depth_all.at[rows, cols].set(
            jnp.nan
        )

        def re_order_cell(
            genotypes_depth_cell: Genotype,
            fitnesses_depth_cell: Fitness,
            fitnesses_depth_all_cell: Fitness,
            descriptors_depth_cell: Descriptor,
            descriptors_depth_all_cell: Descriptor,
        ) -> Tuple[
            Genotype,
            Fitness,
            Fitness,
            Descriptor,
            Descriptor,
            Genotype,
            Fitness,
            Descriptor,
        ]:
            """
            Re-order a cell after extraction. Put highest fitness first and
            empty slot at the end.

            Inputs:
                genotypes_depth_cell: current genotypes of the cell
                fitnesses_depth_cell
                fitnesses_depth_all_cell
                descriptors_depth_cell
                descriptors_depth_all_cell

            Returns:
                genotypes_depth_cell: new genotypes of the cell
                fitnesses_depth_cell
                fitnesses_depth_all_cell
                descriptors_depth_cell
                descriptors_depth_all_cell
                genotypes_cell: new top layer of the cell
                fitnesses_cell
                descriptors_cell
            """

            # Get re-ordering index for given cell
            index = jnp.argsort(fitnesses_depth_cell)
            index = index[::-1]

            # Re-order given cell
            genotypes_depth_cell = jax.tree_util.tree_map(
                lambda x: x.at[index].get(),
                genotypes_depth_cell,
            )
            fitnesses_depth_cell = fitnesses_depth_cell.at[index].get()
            fitnesses_depth_all_cell = fitnesses_depth_all_cell.at[index].get()
            descriptors_depth_cell = descriptors_depth_cell.at[index].get()
            descriptors_depth_all_cell = descriptors_depth_all_cell.at[index].get()

            # Get the top layer of the cell
            genotypes_cell = jax.tree_map(
                lambda x: x.at[0].get(),
                genotypes_depth_cell,
            )
            fitnesses_cell = fitnesses_depth_cell.at[0].get()
            descriptors_cell = descriptors_depth_cell.at[0].get()

            return (
                genotypes_depth_cell,
                fitnesses_depth_cell,
                fitnesses_depth_all_cell,
                descriptors_depth_cell,
                descriptors_depth_all_cell,
                genotypes_cell,
                fitnesses_cell,
                descriptors_cell,
            )

        # Re-order to put extracted individuals at the end of each cell
        (
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_descriptors_depth,
            new_descriptors_depth_all,
            new_genotypes,
            new_fitnesses,
            new_descriptors,
        ) = jax.vmap(re_order_cell)(
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_descriptors_depth,
            new_descriptors_depth_all,
        )

        # Create the new repertoire
        repertoire = self.replace(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses.reshape(-1, 1),
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
        )

        return repertoire


    def extract_uniform(
        self,
        key: RNGKey,
        num_samples: int,
    ) -> Tuple[ExtractMERepertoire, Genotype, Fitness, Descriptor]:
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

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]

        # Set probability for each individual to be sampled
        reshape_fitnesses_depth = self.fitnesses_depth.flatten()
        p = jnp.where(reshape_fitnesses_depth > -jnp.inf, 1.0, 0.0)

        # Extract num_samples indivs
        p = p / jnp.sum(p)
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(
            subkey,
            num_centroids * depth,
            shape=(num_samples,),
            p=p,
            replace=False,
        )
        rows, cols = jnp.divmod(indices, depth)
        rows = rows.astype(int)
        cols = cols.astype(int)

        # Extract the final genotypes, fitnesses_all and descriptors_all to return
        extract_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[rows, cols].get(),
            self.genotypes_depth,
        )
        extract_fitnesses_all = self.fitnesses_depth_all.at[rows, cols].get()
        extract_descriptors_all = self.descriptors_depth_all.at[rows, cols].get()

        # Re-order repertoire following extraction
        key, subkey = jax.random.split(key)
        repertoire = self._order_repertoire(rows, cols, subkey)

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
        )
    
    def extract_prop(
        self,
        key: RNGKey,
        num_samples: int,
        type_prop: str = "exponential",
    ) -> Tuple[ExtractMERepertoire, Genotype, Fitness, Descriptor]:
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

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]

        # Set probability for each individual to be sampled
        reshape_fitnesses_depth = self.fitnesses_depth.flatten()
        if type_prop == "exponential":
            p = jnp.exp(-jnp.arange(0, depth))
        elif type_prop == "linear":
            p = depth - jnp.arange(0, depth)
        elif type_prop == "harmonic":
            p = 1.0 / (jnp.arange(depth) + 1)
        else:
            assert 0, "!!!ERROR!!! Not implemented type_prop."
        p = jnp.repeat(jnp.expand_dims(p, axis=0), num_centroids, axis=0)
        p = p.flatten()
        p = jnp.where(
            reshape_fitnesses_depth > -jnp.inf,
            p,
            0,
        )

        # Extract num_samples indivs
        p = p / jnp.sum(p)
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(
            subkey,
            num_centroids * depth,
            shape=(num_samples,),
            p=p,
            replace=False,
        )
        rows, cols = jnp.divmod(indices, depth)
        rows = rows.astype(int)
        cols = cols.astype(int)

        # Extract the final genotypes, fitnesses_all and descriptors_all to return
        extract_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[rows, cols].get(),
            self.genotypes_depth,
        )
        extract_fitnesses_all = self.fitnesses_depth_all.at[rows, cols].get()
        extract_descriptors_all = self.descriptors_depth_all.at[rows, cols].get()

        # Re-order repertoire following extraction
        key, subkey = jax.random.split(key)
        repertoire = self._order_repertoire(rows, cols, subkey)

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
        )

    def select(
        self,
        key: RNGKey,
        num_samples: int,
        selector: Optional[Selector[MapElitesRepertoireT]] = None,
    ) -> MapElitesRepertoireT:
        """Selects individuals from the repertoire. Sample only from the best individuals ("first
        layer of the depth") contained in genotypes, fitnesses and descriptors.

        Args:
            key: The random key to use for the selection.
            num_samples: The number of individuals to select
            selector: Ignored selector to use for the selection.

        Returns:
            A repertoire containing the selected individuals.
        """
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty.squeeze()) / jnp.sum(1.0 - grid_empty)

        size_repertoire = self.centroids.shape[0]
        indexes = jnp.arange(size_repertoire)
        key, subkey = jax.random.split(key)
        selected_indexes = jax.random.choice(
            subkey,
            indexes,
            shape=(num_samples,),
            p=p,
            replace=True,
        )

        selected_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[selected_indexes].get(),
            self.genotypes,
        )
        selected_fitnesses = self.fitnesses.at[selected_indexes].get()
        selected_descriptors = self.descriptors.at[selected_indexes].get()

        return MapElitesRepertoire(
            genotypes=selected_genotypes,
            fitnesses=selected_fitnesses,
            descriptors=selected_descriptors,
            centroids=self.centroids,
            extra_scores=None,
            keys_extra_scores=()
        )