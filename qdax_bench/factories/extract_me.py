import os 
import matplotlib.pyplot as plt
import jax 
import hydra 
import wandb
from omegaconf import DictConfig, OmegaConf

import warnings
from qdax.core.map_elites import MAPElites
from qdax_bench.core.extract_mapelites import ExtractMAPElites
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
            init_variables_func, 
            key
        ) = setup_qd(cfg)


        emitter = hydra.utils.instantiate(algo["emitter"])
        print("Emitter: ", emitter)

        repertoire_init_fn = hydra.utils.instantiate(
            algo["repertoire_init"]
        )

        map_elites = ExtractMAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
            repertoire_init=repertoire_init_fn,
            max_number_evals=algo["extract"]["max_number_evals"],
            batch_size=algo["params"]["batch_size"],
            num_samples=algo["extract"]["num_samples"],
            repertoire_num_samples=algo["extract"]["repertoire_num_samples"],
            extract_type=algo["extract"]["extract_type"],
        )

        key, var_key, subkey = jax.random.split(key, 3)

        init_variables = init_variables_func(var_key)

        plot_prefix = algo["plotting"]["algo_name"].replace(" ", "_")

        return (
            min_bd, 
            max_bd, 
            key, 
            map_elites, 
            emitter, 
            init_variables, 
            centroids,
            plot_prefix,
            scoring_fn,    
            )