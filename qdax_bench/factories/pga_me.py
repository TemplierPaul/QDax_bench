import os 
import matplotlib.pyplot as plt
import jax 
import hydra 
import wandb
import warnings

from qdax.core.map_elites import MAPElites
from qdax_bench.utils.setup import setup_pga

class PGAMEFactory:
    def build(self, cfg):
        task = cfg["task"]
        algo = cfg["algo"]

        batch_size = task["es_params"]["popsize"]
        cfg["initial_batch"] = batch_size

        (
            centroids, 
            min_bd, 
            max_bd, 
            scoring_fn, 
            metrics_fn, 
            init_variables, 
            key,
            env,
            policy_network
        ) = setup_pga(cfg)


        emitter = hydra.utils.instantiate(algo["emitter"])(
            policy_network=policy_network,
            env=env,
        )
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