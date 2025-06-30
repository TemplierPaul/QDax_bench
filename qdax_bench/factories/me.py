import os 
import matplotlib.pyplot as plt
import jax 
import hydra 
import wandb
from omegaconf import DictConfig, OmegaConf

import warnings
from qdax.core.map_elites import MAPElites
from qdax_bench.utils.setup import setup_qd
from qdax.utils.plotting import plot_map_elites_results

class MEFactory:
    def build(self, cfg):
        task = cfg["task"]
        algo = cfg["algo"]

        (
            centroids, 
            min_bd, 
            max_bd, 
            scoring_fn, 
            metrics_fn, 
            init_variables, 
            key
        ) = setup_qd(cfg)


        emitter = hydra.utils.instantiate(algo["emitter"])
        print("Emitter: ", emitter)

        repertoire_init_fn = hydra.utils.instantiate(
            algo["repertoire_init"]
        )

        map_elites = MAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
            repertoire_init=repertoire_init_fn
        )

        key, subkey = jax.random.split(key)

        repertoire, emitter_state, init_metrics = map_elites.init(
            init_variables, 
            centroids, 
            subkey,
        )

        plot_prefix = algo["plotting"]["algo_name"].replace(" ", "_")

        return (
            min_bd, 
            max_bd, 
            key, 
            map_elites, 
            emitter, 
            repertoire, 
            emitter_state,
            init_metrics,
            plot_prefix,
            scoring_fn,    
            )