from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import flax 

from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)
from qdax.core.map_elites import MAPElites

from qdax_bench.core.containers.extract_repertoire import ExtractMERepertoire
from qdax_bench.core.sampling import average, closest, iqr, mad, median, mode, std
from qdax_bench.core.sampling import multi_sample_scoring_function

class SolutionsBatch(flax.struct.PyTreeNode):
    """Container for a batch of solutions to be evaluated multiple times.

    Args:
        genotypes: a PyTree containing the genotypes of the
            individuals in the population. Each leaf has the
            shape (population_size, num_features).
        fitnesses: an array containing the fitness of the individuals
            in the population. With shape (population_size, num_samples).
        descriptors: an array containing the descriptors of the individuals
            in the population. With shape (population_size, num_samples, descriptor_dim).
        extra_scores: extra scores resulting from the evaluation of the genotypes
            with shape (population_size, num_samples, score_dim).
    """
    emit_genotypes: Genotype
    extract_genotypes: Genotype
    extract_fitnesses: Fitness
    extract_descriptors: Descriptor

class ExtractMAPElites(MAPElites):
    """Core elements of the Extract MAP-Elites algorithm.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
    """

    def __init__(
        self,
        scoring_function: Optional[
            Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]
        ],
        emitter: Emitter,
        metrics_function: Callable[[ExtractMERepertoire], Metrics],
        max_number_evals: int,
        batch_size: int,
        num_samples: int = 1,
        repertoire_num_samples: int = 1,
        repertoire_init: Callable[
            [Genotype, Fitness, Descriptor, Centroid, Optional[ExtraScores], int, int, Callable[[jnp.ndarray], jnp.ndarray], Callable[[jnp.ndarray], jnp.ndarray]],
            ExtractMERepertoire,
        ] = ExtractMERepertoire.init,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray] = average,
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray] = average,
        extract_type: str = "proportional",
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._repertoire_init = repertoire_init

        self._max_number_evals = max_number_evals # Max number of evaluations per individual

        self._num_samples = num_samples # Number of evals for new solutions
        self._repertoire_num_samples = repertoire_num_samples # Number of evals per extract
        self.batch_size = batch_size
        
        self._emit_batch_size = self._emitter._batch_size
        self._extract_batch_size = batch_size - self._emit_batch_size

        self._fitness_extractor = fitness_extractor
        self._descriptor_extractor = descriptor_extractor
        self._extract_type = extract_type

    def reshape_fit_bd(self, fitnesses: Fitness, descriptors: Descriptor) -> Tuple[Fitness, Descriptor]:
        """Reshape fitnesses and descriptors to match the expected shape.

        Args:
            fitnesses: fitnesses of shape (batch_size)
            descriptors: descriptors of shape (batch_size, descriptor_dim)
        Returns:
            reshaped fitnesses and descriptors

        """

        fitnesses = jnp.reshape(fitnesses, (fitnesses.shape[0], 1))
        descriptors = jnp.reshape(descriptors, (descriptors.shape[0], 1, descriptors.shape[1]))
        # Extend results to the good shape with jnp.nan
        fitnesses = jnp.pad(
            fitnesses,
            ((0, 0), (0, self._max_number_evals - 1)),
            "constant",
            constant_values=jnp.nan,
        )
        descriptors = jnp.pad(
            descriptors,
            ((0, 0), (0, self._max_number_evals - 1), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )
        return fitnesses, descriptors
        
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        key: RNGKey,
    ) -> Tuple[ExtractMERepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # Evaluate self._num_samples times
        key, subkey = jax.random.split(key)
        (
            fitnesses,
            descriptors,
            extra_scores,
        ) = multi_sample_scoring_function(
            genotypes, subkey, self._scoring_function, self._num_samples
        )

        # Extend results to the good shape with jnp.nan
        fitnesses = jnp.pad(
            fitnesses,
            ((0, 0), (0, self._max_number_evals - self._num_samples)),
            "constant",
            constant_values=jnp.nan,
        )
        descriptors = jnp.pad(
            descriptors,
            ((0, 0), (0, self._max_number_evals - self._num_samples), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )

        repertoire, emitter_state, metrics = self.init_ask_tell(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            key=key,
            extra_scores=extra_scores,
        )
        return repertoire, emitter_state, metrics

    def init_ask_tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        key: RNGKey,
        extra_scores: Optional[ExtraScores] = None,
    ) -> Tuple[ExtractMERepertoire, Optional[EmitterState], Metrics]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes
        and their evaluations.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: initial fitnesses of the genotypes
            descriptors: initial descriptors of the genotypes
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            key: a random key used for stochastic operations.
            extra_scores: extra scores of the initial genotypes (optional)

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter.
        """
        if extra_scores is None:
            extra_scores = {}
        
        # init the repertoire
        repertoire = self._repertoire_init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
            num_evals = self._max_number_evals,
            fitness_extractor=self._fitness_extractor,
            descriptor_extractor=self._descriptor_extractor,
        )

         # Init emitter
        emit_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[-self._emit_batch_size :].get(), genotypes
        )
        emit_fitnesses = self._fitness_extractor(
            fitnesses.at[-self._emit_batch_size :].get()
        )
        emit_descriptors = self._descriptor_extractor(
            descriptors.at[-self._emit_batch_size :].get()
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[-self._emit_batch_size :].get(),
            extra_scores,
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[:, 0].get(),
            emit_extra_scores,
        )
        key, subkey = jax.random.split(key)
        emitter_state = self._emitter.init(
            key=subkey,
            repertoire=repertoire,
            genotypes=emit_genotypes,
            fitnesses=emit_fitnesses,
            descriptors=emit_descriptors,
            extra_scores=emit_extra_scores,
        )

        # calculate the initial metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
    
    def _extract_repertoire(
        self,
        repertoire: Repertoire,
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Genotype, Fitness, Descriptor]:
        """
        Extract the part of the repertoire that needs to be evaluated.
        The assumption is that these are removed from the repertoire.

        Input:
            repertoire
            random_key
        Returns:
            repertoire: repertoire after extraction
            extract_genotypes: the genotypes extracted from the repertoire
            extract_fitnesses: the corresponding fitnesses, also extracted
            extract_descriptors: the corresponding descriptors, also extracted
        """

        # Extract with prob proportional to ranking
        if self._extract_type == "proportional":
            return repertoire.extract_prop(random_key, self._extract_batch_size, type_prop="exponential")  # type: ignore
        if self._extract_type == "proportional_harmonic":
            return repertoire.extract_prop(random_key, self._extract_batch_size, type_prop="harmonic")  # type: ignore
        if self._extract_type == "proportional_linear":
            return repertoire.extract_prop(random_key, self._extract_batch_size, type_prop="linear")  # type: ignore

        # Extract with uniform prob
        elif self._extract_type == "uniform":
            return repertoire.extract_uniform(random_key, self._extract_batch_size)  # type: ignore

        else:
            assert 0, "!!!ERROR!!! Undefined extract type."
    
    def ask(
        self,
        repertoire: ExtractMERepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[Genotype, ExtraScores, Fitness, Descriptor]:
        """
        Ask the emitter to generate a new batch of genotypes.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key
        """
        key, emit_key, extract_key = jax.random.split(key, 3)
        emit_genotypes, extra_info = self._emitter.emit(repertoire, emitter_state, emit_key)
        
        # Extract from the repertoire
        (
            new_repertoire,
            extract_genotypes,
            extract_fitnesses,
            extract_descriptors,
        ) = self._extract_repertoire(repertoire, extract_key)

        # Create a genotypes to match num_samples and repertoire_num_samples evaluations
        evaluation_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate(
                [
                    jnp.repeat(x, self._repertoire_num_samples, axis=0),
                    jnp.repeat(y, self._num_samples, axis=0),
                ],
                axis=0,
            ),
            extract_genotypes,
            emit_genotypes,
        )

        solutions_batch = SolutionsBatch(
            emit_genotypes=emit_genotypes,
            extract_genotypes=extract_genotypes,
            extract_fitnesses=extract_fitnesses,
            extract_descriptors=extract_descriptors,
        )
        
        return (
            evaluation_genotypes,
            extra_info,
            solutions_batch,
            new_repertoire
        )
    

    def tell(
        self,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        solutions_batch: SolutionsBatch,
        repertoire: ExtractMERepertoire,
        emitter_state: Optional[EmitterState],
        extra_scores: Optional[ExtraScores] = None,
        extra_info: Optional[ExtraScores] = None,
    ) -> Tuple[ExtractMERepertoire, Optional[EmitterState], Metrics]:
        if extra_scores is None:
            extra_scores = {}
        if extra_info is None:
            extra_info = {}

        extract_fitnesses = solutions_batch.extract_fitnesses
        extract_descriptors = solutions_batch.extract_descriptors
        extract_genotypes = solutions_batch.extract_genotypes
        emit_genotypes = solutions_batch.emit_genotypes

        # Get fitness for emit in correct shape
        new_emit_fitnesses = fitnesses[-self._emitter.batch_size * self._num_samples :]
        new_emit_fitnesses = jnp.reshape(
            new_emit_fitnesses, (self._emitter.batch_size, self._num_samples)
        )
        new_emit_descriptors = descriptors[
            -self._emitter.batch_size * self._num_samples :
        ]
        new_emit_descriptors = jnp.reshape(
            new_emit_descriptors,
            (self._emitter.batch_size, self._num_samples, new_emit_descriptors.shape[-1]),
        )
        new_emit_fitnesses = jnp.pad(
            new_emit_fitnesses,
            ((0, 0), (0, self._max_number_evals - self._num_samples)),
            "constant",
            constant_values=jnp.nan,
        )
        new_emit_descriptors = jnp.pad(
            new_emit_descriptors,
            ((0, 0), (0, self._max_number_evals - self._num_samples), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )

        # Concatenate the results of the new evaluations of extract with the old ones
        new_extract_fitnesses = fitnesses[: -self._emitter.batch_size * self._num_samples]
        new_extract_fitnesses = jnp.reshape(
            new_extract_fitnesses, (-1, self._repertoire_num_samples)
        )
        new_extract_fitnesses = jnp.concatenate(
            [
                new_extract_fitnesses,
                extract_fitnesses.at[:, :-self._repertoire_num_samples].get(),
            ],
            axis=1,
        )
        new_extract_descriptors = descriptors[
            : -self._emitter.batch_size * self._num_samples
        ]
        new_extract_descriptors = jnp.reshape(
            new_extract_descriptors,
            (-1, self._repertoire_num_samples, new_extract_descriptors.shape[-1]),
        )
        new_extract_descriptors = jnp.concatenate(
            [
                new_extract_descriptors,
                extract_descriptors.at[:, :-self._repertoire_num_samples].get(),
            ],
            axis=1,
        )


        # Concatenate emit and extract evaluations
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            extract_genotypes,
            emit_genotypes,
        )
        fitnesses = jnp.concatenate(
            [
                new_extract_fitnesses,
                new_emit_fitnesses,
            ],
            axis=0,
        )
        descriptors = jnp.concatenate(
            [
                new_extract_descriptors,
                new_emit_descriptors,
            ],
            axis=0,
        )

        # Separate results of emitter solutions to update the emitter state
        emit_fitnesses = self._fitness_extractor(new_emit_fitnesses)
        emit_descriptors = self._descriptor_extractor(new_emit_descriptors)
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[-self._emitter.batch_size * self._num_samples :].get(),
            extra_scores,
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                (
                    self._emitter.batch_size,
                    self._num_samples,
                )
                + x.shape[1:],
            ),
            emit_extra_scores,
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[:, 0].get(),
            emit_extra_scores,
        )

        # Add everything back to the archive
        # repertoire = self._add_repertoire(
        #     repertoire=repertoire,
        #     genotypes=genotypes,
        #     descriptors=descriptors,
        #     fitnesses=fitnesses,
        #     extra_scores=extra_scores,
        # )
        repertoire = repertoire.add_all_evals(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor = self._fitness_extractor,
            descriptor_extractor = self._descriptor_extractor,
        )

        # Update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=emit_genotypes,
            fitnesses=emit_fitnesses,
            descriptors=emit_descriptors,
            extra_scores={**emit_extra_scores, **extra_info},
        )

        # Update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics
    
    def update(
        self,
        repertoire: ExtractMERepertoire,
        emitter_state: Optional[EmitterState],
        key: RNGKey,
    ) -> Tuple[ExtractMERepertoire, Optional[EmitterState], Metrics]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
        """
        if self._scoring_function is None:
            raise ValueError("Scoring function is not set.")

        # generate offsprings with the emitter
        key, subkey = jax.random.split(key)
        evaluation_genotypes, extra_info, solutions_batch, asked_repertoire = self.ask(repertoire, emitter_state, subkey)

        # scores the offsprings
        key, subkey = jax.random.split(key)
        (fitnesses, descriptors, extra_scores) = self._scoring_function(
            evaluation_genotypes, subkey
        )

        repertoire, emitter_state, metrics = self.tell(
            genotypes=evaluation_genotypes,
            fitnesses=fitnesses,
            solutions_batch=solutions_batch,
            descriptors=descriptors,
            repertoire=asked_repertoire,
            emitter_state=emitter_state,
            extra_scores=extra_scores,
            extra_info=extra_info,
        )
        return repertoire, emitter_state, metrics