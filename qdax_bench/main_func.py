import hydra

from typing import Dict
from typing import Dict
import time
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
# Jax floating point precision
# os.environ["JAX_ENABLE_X64"] = "True"
import matplotlib.pyplot as plt
import matplotlib

print("Matplotlib backend:", matplotlib.get_backend())
from tqdm import tqdm 
import jax 
import jax.numpy as jnp
import warnings

print("Jax version:", jax.__version__)
# print("Jax backend:", jax.lib.xla_bridge.get_backend().platform)
print("Jax devices:", jax.devices())
device = jax.devices()[0]
# Fail if no GPU is available
if "cuda" not in str(device).lower():
    warnings.warn("No GPU available. Please run on a GPU machine.")

from typing import Dict

from qdax.utils.metrics import default_qd_metrics
from qdax.utils.uncertainty_metrics import reevaluation_function

print("QDax imports done")
from qdax_bench.utils.plotting import plot_map_elites_results
from qdax_bench.utils.loggers import CSVLogger, WandBLogger, CombinedLogger, make_loggers

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# Check there is a gpu
import wandb
import math

print("Imports done")

def oc_if(cond: bool, true_val, false_val):
    cond = str(cond).strip()
    if cond.lower() == "true":
        return true_val
    elif cond.lower() == "false":
        return false_val
    else:
        raise ValueError(f"Invalid condition: {cond}")

OmegaConf.register_new_resolver("oc.if", oc_if)

def clean_algo_cfg(cfg):
    # Resolve all interpolations (e.g., ${oc.select:...})
    resolved = OmegaConf.to_container(cfg.algo, resolve=True)

    # Remove internal helper sections
    for key in ["group_defaults"]:
        if key in resolved:
            resolved.pop(key, None)

    # Convert back to OmegaConf if needed
    return OmegaConf.create(resolved)

def run(cfg: DictConfig) -> None:
    cfg.algo = clean_algo_cfg(cfg)
    
    print(OmegaConf.to_yaml(cfg))
    task = cfg.task
    algo = cfg.algo
    import os
    os.makedirs(cfg.plots_dir, exist_ok=True)

    plotting_function = task.plotting.get("plotting_function", None)
    if plotting_function is None:
        plotting_function = plot_map_elites_results
        print("Using default plotting function:", plotting_function.__name__)
    else:
        plotting_function = hydra.utils.instantiate(cfg.task.plotting.plotting_function)
        print("Using custom plotting function:", plotting_function)

    algo_plotting_function = algo.plotting.get("plotting_function", None)
    if algo_plotting_function is None:
        print("No algorithm plotting function specified.")
    else:
        algo_plotting_function = hydra.utils.instantiate(cfg.algo.plotting.plotting_function)
        print("Using custom algorithm plotting function:", algo_plotting_function)

    algo_factory = hydra.utils.instantiate(cfg.algo.factory)

    (
        min_descriptor, 
        max_descriptor, 
        key, 
        map_elites, 
        emitter, 
        init_variables, 
        centroids,
        plot_prefix,
        scoring_fn, 
        ) = algo_factory.build(OmegaConf.to_container(cfg, resolve=True))

    key, subkey = jax.random.split(key, 2)

    ask_tell = task.get("ask_tell", False)

    if ask_tell:
        # Evaluate initial population
        key_eval, key_init = jax.random.split(subkey)
        fitnesses, descriptors, extra_scores = scoring_fn(init_variables, key_eval)
        
        # Initialize repertoire and emitter state
        repertoire, emitter_state, init_metrics = map_elites.init_ask_tell(
            genotypes=init_variables,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            key=key_init,
            extra_scores=extra_scores,
        )

    else:
        repertoire, emitter_state, init_metrics = map_elites.init(
                init_variables, 
                centroids, 
                subkey,
            )

    # Empty repertoire for corrected metrics
    empty_repertoire = repertoire.replace(
            fitnesses=jnp.ones_like(repertoire.fitnesses) * -jnp.inf
        )

    # Check if emitter has evals_per_gen
    if hasattr(emitter, "evals_per_gen"):
        evals_per_gen = emitter.evals_per_gen
    else: 
        warnings.warn(f"Emitter does not have evals_per_gen attribute. Using batch size of {emitter.batch_size} instead.")
        evals_per_gen = emitter.batch_size

    num_generations = math.ceil(task.total_evaluations / evals_per_gen) 
    log_period = math.ceil(task.total_evaluations / evals_per_gen / cfg.num_loops)

    metrics_names = ["generation", "evaluations", "qd_score", "coverage", "max_fitness", "time"]

    logger, corrected_logger = make_loggers(
        cfg, 
        evals_per_gen=evals_per_gen,
        num_generations=num_generations,
        log_period=log_period,
        metrics_names=metrics_names
    )
    
    print("Total generations:", num_generations)
    print("Log period:", log_period)

    metrics = dict.fromkeys(metrics_names, jnp.array([]))

    init_metrics["generation"] = 0
    init_metrics["evaluations"] = cfg["algo"]["initial_batch"]
    init_metrics["time"] = 0.0
    metrics = jax.tree.map(lambda metric, current_metric: jnp.append(metric, current_metric), metrics, init_metrics)

    logger.log(init_metrics)

    if cfg.corrected_metrics.use:
        corrected_metrics = dict.fromkeys(metrics_names, jnp.array([]))

        corrected_repertoire = reevaluation_function(
                repertoire=repertoire,
                key=key,
                empty_corrected_repertoire=empty_repertoire,
                scoring_fn=scoring_fn, 
                num_reevals=cfg.corrected_metrics.evals,
            )
        corrected_current_metrics = map_elites._metrics_function(corrected_repertoire)
        corrected_current_metrics["generation"] = 0
        corrected_current_metrics["evaluations"] = cfg["algo"]["initial_batch"]
        corrected_current_metrics["time"] = 0.0

        corrected_metrics = jax.tree.map(lambda metric, current_metric: jnp.append(metric, current_metric), corrected_metrics, corrected_current_metrics)

        corrected_logger.log({"corrected_" + k: v for k, v in corrected_current_metrics.items()})
    
    def scan_loop(repertoire, emitter_state, key):
        return jax.lax.scan(
            jax.jit(map_elites.scan_update),
            (repertoire, emitter_state, key),
            (),
            length=log_period,
        )

    def ask_tell_loop(repertoire, emitter_state, key):
        loop_metrics = dict.fromkeys(
            ["qd_score", "coverage", "max_fitness"], 
            jnp.array([]))
        for i in range(log_period):
            key, subkey = jax.random.split(key)
            # Generate solutions
            genotypes, extra_info = jax.jit(map_elites.ask)(repertoire, emitter_state, subkey)

            # Evaluate solutions: get fitness, descriptor and extra scores.
            # This is where custom evaluations on CPU or GPU can be added.
            key, subkey = jax.random.split(key)
            fitnesses, descriptors, extra_scores = scoring_fn(genotypes, subkey)

            # Update MAP-Elites
            repertoire, emitter_state, current_metrics = jax.jit(map_elites.tell)(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                repertoire=repertoire,
                emitter_state=emitter_state,
                extra_scores=extra_scores,
                extra_info=extra_info,
            )

            current_metrics = jax.tree.map(lambda x: jnp.array([x]), current_metrics)
            loop_metrics = jax.tree.map(
                lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0),
                loop_metrics,
                current_metrics
            )
        return (repertoire, emitter_state, key), loop_metrics

    loop_fn = scan_loop
    if ask_tell:
        print("Using ask-tell loop")
        loop_fn = ask_tell_loop

    for i in tqdm(range(cfg.num_loops)):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = loop_fn(
            repertoire=repertoire,
            emitter_state=emitter_state,
            key=key,
        )

        timelapse = time.time() - start_time

        # Metrics
        current_metrics["generation"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["evaluations"] = current_metrics["generation"] * evals_per_gen + cfg["algo"]["initial_batch"]
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)
           
        logger.batch_log(current_metrics)

        key, subkey = jax.random.split(key)
        if cfg.corrected_metrics.use:
            corrected_repertoire = reevaluation_function(
                repertoire=repertoire,
                key=subkey,
                empty_corrected_repertoire=empty_repertoire,
                scoring_fn=scoring_fn, 
                num_reevals=cfg.corrected_metrics.evals,
            )
            corrected_current_metrics = map_elites._metrics_function(corrected_repertoire)

            corrected_current_metrics["generation"] = current_metrics["generation"][-1]
            corrected_current_metrics["evaluations"] = current_metrics["evaluations"][-1]
            corrected_current_metrics["time"] = current_metrics["time"][-1]

            corrected_logger.log({"corrected_" + k: v for k, v in corrected_current_metrics.items()})

            corrected_metrics = jax.tree.map(lambda metric, current_metric: jnp.append(metric, current_metric), corrected_metrics, corrected_current_metrics)


    # to numpy int64
    evals = jax.device_get(metrics["evaluations"]).astype(int)

    if cfg.plot:
        # Create the plots and the grid
        fig, axes = plotting_function(
            env_steps=evals, 
            metrics=metrics, 
            repertoire=repertoire, 
            min_descriptor=min_descriptor, 
            max_descriptor=max_descriptor,
            x_label="Evaluations",
            )
        fig.suptitle(f"{cfg.algo.plotting.algo_name} - {cfg.task.plotting.task_name}")
        fig.tight_layout()

        # figname = f"{cfg.plots_dir}/{cfg.task.plotting.task_name}/{plot_prefix}"+ "_results.png"
        figname = f"{cfg.plots_dir}/{cfg.task.plotting.task_name}_{plot_prefix}".replace(".", "_")  #+ "_results.png"

        os.makedirs(os.path.dirname(figname), exist_ok=True)
        print("Save figure in: ", figname)
        plt.savefig(figname, bbox_inches="tight")

        if algo_plotting_function is not None:
            algo_plotting_function(
                cfg, 
                emitter_state,
                metrics,
                repertoire,
                min_descriptor,
                max_descriptor,
                title=f"{cfg.algo.plotting.algo_name} - {cfg.task.plotting.task_name}"
            )

        if cfg.wandb.use:
            wandb_run.log({"final_results": wandb.Image(fig)})

        if cfg.corrected_metrics.use:
            key, subkey = jax.random.split(key)
            corrected_repertoire = reevaluation_function(
                repertoire=repertoire,
                key=subkey,
                empty_corrected_repertoire=empty_repertoire,
                scoring_fn=scoring_fn, 
                num_reevals=cfg.corrected_metrics.evals,
            )
            evals = jax.device_get(corrected_metrics["evaluations"]).astype(int)

            # Plot corrected metrics
            fig, axes = plotting_function(
                env_steps=evals,
                metrics=corrected_metrics,
                repertoire=corrected_repertoire,
                min_descriptor=min_descriptor, 
                max_descriptor=max_descriptor,
                x_label="Evaluations",
            )
            fig.tight_layout()

            fig.suptitle(f"{cfg.algo.plotting.algo_name} - {cfg.task.plotting.task_name} - Corrected")
            figname = f"{cfg.plots_dir}/{cfg.task.plotting.task_name}_{plot_prefix}_corrected".replace(".", "_") 
            os.makedirs(os.path.dirname(figname), exist_ok=True)
            print("Save corrected figure in: ", figname)
            plt.savefig(figname, bbox_inches="tight")
            if cfg.wandb.use:
                wandb_run.log({"corrected_results": wandb.Image(fig)})

    logger.finish()
    corrected_logger.finish()
        