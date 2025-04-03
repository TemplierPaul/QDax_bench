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

        batch_size = task["es_params"]["popsize"]
        cfg["initial_batch"] = batch_size

        # if hasattr(task, "legacy_spring"):
        #     legacy_spring = task.legacy_spring
        # else:
        #     legacy_spring = False
        #     warnings.warn("Legacy spring not set. Defaulting to False")

        # setup_config = {
        #     "seed": cfg.seed,
        #     "env": task.env_name,
        #     "descriptors": task.descriptors,
        #     "episode_length": task.episode_length,
        #     "stochastic": task.stochastic,
        #     "legacy_spring": legacy_spring,
        #     "policy_hidden_layer_sizes": task.network.policy_hidden_layer_sizes,
        #     "activation": task.network.activation,
        #     "initial_batch": initial_batch,
        #     "num_init_cvt_samples": algo.archive.num_init_cvt_samples,
        #     "num_centroids": algo.archive.num_centroids,
        # }
        
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

        map_elites = MAPElites(
            scoring_function=scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
        )

        # with jax.disable_jit():
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