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

from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.uncertainty_metrics import reevaluation_function

print("QDax imports done")
from qdax_bench.utils.plotting import plot_map_elites_results 

from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# Check there is a gpu
import wandb
import math

print("Imports done")

def set_env_params(cfg: DictConfig) -> Dict:
    if "env_params" not in cfg.algo.keys():
        return cfg
    env_params = cfg.algo.env_params.defaults
    if cfg.task.env_name in cfg.algo.env_params.keys():
        for k, v in cfg.algo.env_params[cfg.task.env_name].items():
            env_params[k] = v
    cfg.algo.env_params = env_params
    return cfg

def run(cfg: DictConfig) -> None:
    # cfg = set_env_params(cfg)
    print(OmegaConf.to_yaml(cfg))
    task = cfg.task
    algo = cfg.algo
    import os
    os.makedirs(cfg.plots_dir, exist_ok=True)

    algo_factory = hydra.utils.instantiate(cfg.algo.factory)

    (
        min_descriptor, 
        max_descriptor, 
        key, 
        map_elites, 
        emitter, 
        repertoire, 
        emitter_state,
        init_metrics,
        plot_prefix,
        scoring_fn, 
        ) = algo_factory.build(OmegaConf.to_container(cfg, resolve=True))
    
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

    wandb_run = None
    if cfg.wandb.use:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict["evals_per_gen"] = evals_per_gen
        cfg_dict["num_generations"] = num_generations
        cfg_dict["log_period"] = log_period

        try:
            # Get QDax commit hash
            cmd = "pip freeze | grep qdax"
            output = os.popen(cmd).read()
            # Format qdax @ git+https://github.com/adaptive-intelligent-robotics/QDax.git@dcdc098fee1dad99f264e80f31208ccfd4a06a12
            qdax_commit = output.split("@")[-1].strip()
            cfg_dict["qdax_commit"] = qdax_commit
            # Add link to commit
            cfg_dict["qdax_commit_link"] = f"https://github.com/adaptive-intelligent-robotics/QDax/commit/{qdax_commit}"
        except:
            print("Could not get QDax commit hash. Skipping.")

        wandb_run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, config=cfg_dict)

    print("Total generations:", num_generations)
    print("Log period:", log_period)

    metrics = dict.fromkeys(["generation", "evaluations", "qd_score", "coverage", "max_fitness", "time"], jnp.array([]))

    init_metrics["generation"] = 0
    init_metrics["evaluations"] = 0
    init_metrics["time"] = 0.0
    metrics = jax.tree.map(lambda metric, current_metric: jnp.append(metric, current_metric), metrics, init_metrics)

    if cfg.wandb.use:
        # print(init_metrics)
        # Log the metrics to wandb
        wandb_run.log(init_metrics)

    if cfg.corrected_metrics.use:
        corrected_metrics = dict.fromkeys(["generation", "evaluations", "qd_score", "coverage", "max_fitness", "time"], jnp.array([]))

        corrected_repertoire = reevaluation_function(
                repertoire=repertoire,
                key=key,
                empty_corrected_repertoire=empty_repertoire,
                scoring_fn=scoring_fn, 
                num_reevals=cfg.corrected_metrics.evals,
            )
        corrected_current_metrics = map_elites._metrics_function(corrected_repertoire)
        corrected_current_metrics["generation"] = 0
        corrected_current_metrics["evaluations"] = 0
        corrected_current_metrics["time"] = 0.0

        corrected_metrics = jax.tree.map(lambda metric, current_metric: jnp.append(metric, current_metric), corrected_metrics, corrected_current_metrics)

        if cfg.wandb.use:
            # Log the metrics to wandb with corrected prefix
            wandb_run.log({"corrected_" + k: v for k, v in corrected_current_metrics.items()})

    # csv_logger = CSVLogger(
    #     "mapelites-logs.csv",
    #     header=list(metrics.keys())
    # )

    map_elites_scan_update = jax.jit(map_elites.scan_update)
    for i in tqdm(range(cfg.num_loops)):
        start_time = time.time()
        (
            repertoire,
            emitter_state,
            key,
        ), current_metrics = jax.lax.scan(
            map_elites_scan_update,
            (repertoire, emitter_state, key),
            (),
            length=log_period,
        )
        timelapse = time.time() - start_time

        # Metrics
        current_metrics["generation"] = jnp.arange(1+log_period*i, 1+log_period*(i+1), dtype=jnp.int32)
        current_metrics["evaluations"] = current_metrics["generation"] * evals_per_gen
        current_metrics["time"] = jnp.repeat(timelapse, log_period)
        metrics = jax.tree.map(lambda metric, current_metric: jnp.concatenate([metric, current_metric], axis=0), metrics, current_metrics)

        if cfg.wandb.use:
            # Log the metrics to wandb
            for i in range(log_period):
                wandb_run.log({k: v[i] for k, v in current_metrics.items()})

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

            if cfg.wandb.use:
                # Log the metrics to wandb with corrected prefix
                wandb_run.log({"corrected_" + k: v for k, v in corrected_current_metrics.items()})

            corrected_metrics = jax.tree.map(lambda metric, current_metric: jnp.append(metric, current_metric), corrected_metrics, corrected_current_metrics)
        # Log
        # csv_logger.log(jax.tree.map(lambda x: x[-1], metrics))    

    # to numpy int64
    evals = jax.device_get(metrics["evaluations"]).astype(int)

    # Create the plots and the grid
    fig, axes = plot_map_elites_results(
        env_steps=evals, 
        metrics=metrics, 
        repertoire=repertoire, 
        min_descriptor=min_descriptor, 
        max_descriptor=max_descriptor,
        x_label="Evaluations",
        )
    fig.suptitle(f"{cfg.algo.plotting.algo_name} - {cfg.task.plotting.task_name}")


    # figname = f"{cfg.plots_dir}/{cfg.task.env_name}/{plot_prefix}"+ "_results.png"
    figname = f"{cfg.plots_dir}/{cfg.task.env_name}_{plot_prefix}" #+ "_results.png"

    os.makedirs(os.path.dirname(figname), exist_ok=True)
    print("Save figure in: ", figname)
    plt.savefig(figname, bbox_inches="tight")
    
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
        fig, axes = plot_map_elites_results(
            env_steps=evals,
            metrics=corrected_metrics,
            repertoire=corrected_repertoire,
            min_descriptor=min_descriptor, 
            max_descriptor=max_descriptor,
            x_label="Evaluations",
        )
        fig.suptitle(f"{cfg.algo.plotting.algo_name} - {cfg.task.plotting.task_name} - Corrected")
        figname = f"{cfg.plots_dir}/{cfg.task.env_name}_{plot_prefix}_corrected"
        os.makedirs(os.path.dirname(figname), exist_ok=True)
        print("Save corrected figure in: ", figname)
        plt.savefig(figname, bbox_inches="tight")
        if cfg.wandb.use:
            wandb_run.log({"corrected_results": wandb.Image(fig)})

    if cfg.wandb.use:
        wandb_run.finish()
        