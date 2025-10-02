from typing import Dict, Optional, Tuple, List

import jax.numpy as jnp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire

def plot_map_elites_results(
    env_steps: jnp.ndarray,
    metrics: Dict,
    repertoire: MapElitesRepertoire,
    min_descriptor: jnp.ndarray,
    max_descriptor: jnp.ndarray,
    x_label: str = "Environment steps",
) -> Tuple[Optional[Figure], Axes]:
    """Plots three usual QD metrics, namely the coverage, the maximum fitness
    and the QD-score, along the number of environment steps. This function also
    plots a visualisation of the final map elites grid obtained. It ensures that
    those plots are aligned together to give a simple and efficient visualisation
    of an optimization process.

    Args:
        env_steps: the array containing the number of steps done in the environment.
        metrics: a dictionary containing metrics from the optimizatoin process.
        repertoire: the final repertoire obtained.
        min_descriptor: the minimal possible values for the descriptor.
        max_descriptor: the maximal possible values for the descriptor.
        x_label: label for the x axis, defaults to environment steps

    Returns:
        A figure and axes with the plots of the metrics and visualisation of the grid.
    """
    # Customize matplotlib params
    font_size = 16
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
    }

    mpl.rcParams.update(params)

    # Visualize the training evolution and final repertoire
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))

    # env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    axes[0].plot(env_steps, metrics["coverage"])
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Coverage in %")
    axes[0].set_title("Coverage evolution during training")
    axes[0].set_aspect(0.95 / axes[0].get_data_ratio(), adjustable="box")

    axes[1].plot(env_steps, metrics["max_fitness"])
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Maximum fitness")
    axes[1].set_title("Maximum fitness evolution during training")
    axes[1].set_aspect(0.95 / axes[1].get_data_ratio(), adjustable="box")

    axes[2].plot(env_steps, metrics["qd_score"])
    axes[2].set_xlabel(x_label)
    axes[2].set_ylabel("QD Score")
    axes[2].set_title("QD Score evolution during training")
    axes[2].set_aspect(0.95 / axes[2].get_data_ratio(), adjustable="box")

    _, axes = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_descriptor,
        maxval=max_descriptor,
        repertoire_descriptors=repertoire.descriptors,
        ax=axes[3],
    )

    return fig, axes

def plot_bbob_results(
    env_steps: jnp.ndarray,
    metrics: Dict,
    repertoire: MapElitesRepertoire,
    min_descriptor: jnp.ndarray,
    max_descriptor: jnp.ndarray,
    x_label: str = "Environment steps",
) -> Tuple[Optional[Figure], Axes]:
    """Plots three usual QD metrics, namely the coverage, the maximum fitness
    and the QD-score, along the number of environment steps. This function also
    plots a visualisation of the final map elites grid obtained. It ensures that
    those plots are aligned together to give a simple and efficient visualisation
    of an optimization process.

    Args:
        env_steps: the array containing the number of steps done in the environment.
        metrics: a dictionary containing metrics from the optimizatoin process.
        repertoire: the final repertoire obtained.
        min_descriptor: the minimal possible values for the descriptor.
        max_descriptor: the maximal possible values for the descriptor.
        x_label: label for the x axis, defaults to environment steps

    Returns:
        A figure and axes with the plots of the metrics and visualisation of the grid.
    """
    # Customize matplotlib params
    font_size = 24
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "figure.titlesize": font_size + 2,
        "text.usetex": False,
        "axes.titlepad": 10,
    }

    mpl.rcParams.update(params)

    # Visualize the training evolution and final repertoire
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))

    # env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    axes[0].plot(env_steps, metrics["coverage"])
    axes[0].set_xlabel(x_label)
    # x: scientific notation for x-axis
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[0].set_ylabel("Coverage in %")
    axes[0].set_title("Coverage evolution during training")
    axes[0].set_aspect(0.95 / axes[0].get_data_ratio(), adjustable="box")

    axes[1].plot(env_steps, - metrics["max_fitness"])
    axes[1].set_xlabel(x_label)
    axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[1].set_yscale("log")
    # axes[1].set_ylim(bottom=1e-6)  # Avoid log(0) issues
    axes[1].set_ylabel("Error (negative fitness)")
    axes[1].set_title("Error evolution during training")
    axes[1].set_aspect(0.95 / axes[1].get_data_ratio(), adjustable="box")

    # Replace inf with -inf
    error_log = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, jnp.log10(-repertoire.fitnesses))
    _, ax = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=error_log,
        minval=min_descriptor,
        maxval=max_descriptor,
        repertoire_descriptors=repertoire.descriptors,
        ax=axes[2],
    )
    ax.set_title("Final repertoire (log error, lower is better)")

    return fig, axes


def plot_repertoire_grid(
    repertoire: MapElitesRepertoire,
    min_descriptor: jnp.ndarray,
    max_descriptor: jnp.ndarray,
    descriptor_labels: Optional[List[str]] = None,
) -> Tuple[Figure, plt.Axes]:
    """
    Plots a high-dimensional MAP-Elites repertoire as a grid of 2D projections.
    
    This function uses an argument signature similar to QDax's 2D plotting utility
    and cleans up artifacts from the internal plotting calls.
    """
    fitnesses = repertoire.fitnesses
    descriptors = repertoire.descriptors
    centroids = repertoire.centroids

    if fitnesses.size == 0:
        print("Warning: Repertoire is empty, nothing to plot.")
        return plt.subplots()

    num_descriptors = descriptors.shape[1]

    # Use provided vmin/vmax or calculate from data
    if num_descriptors == 2:
        print("Note: Repertoire has 2 descriptors, using direct 2D plot.")
        return plot_2d_map_elites_repertoire(
                    centroids=centroids,
                    repertoire_fitnesses=fitnesses,
                    minval=min_descriptor,
                    maxval=max_descriptor,
                    # vmin=vmin,
                    # vmax=vmax,
                )
    
    if descriptor_labels is None:
        descriptor_labels = [f"BD {i+1}" for i in range(num_descriptors)]

    fig, axes = plt.subplots(
        num_descriptors,
        num_descriptors,
        figsize=(num_descriptors * 3, num_descriptors * 3),
        sharex='col',
    )
    if num_descriptors == 1:
        axes = np.array([[axes]])
        
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # Manually link y-axes for off-diagonal plots to fix histogram scaling
    for i in range(num_descriptors):
        for j in range(1, i):
            axes[i, j].sharey(axes[i, 0])

    for i in range(num_descriptors): # Rows (y-axis)
        for j in range(num_descriptors): # Columns (x-axis)
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue

            if i == j: # Diagonal plots: Histograms
                ax.hist(np.asarray(descriptors[:, i]), bins=30, color="#3A528BFF")
                ax.set_xlim(min_descriptor[j], max_descriptor[j])

            else: # Lower Triangle: Call QDax function and clean up
                projected_descriptors = descriptors[:, [j, i]]
                projected_centroids = centroids[:, [j, i]] 
                
                # Use the provided minval and maxval for boundaries
                minval_2d = jnp.array([min_descriptor[j], min_descriptor[i]])
                maxval_2d = jnp.array([max_descriptor[j], max_descriptor[i]])

                axes_before_call = set(fig.axes)
                
                _, ax = plot_2d_map_elites_repertoire(
                    centroids=projected_centroids,
                    repertoire_fitnesses=fitnesses,
                    minval=minval_2d,
                    maxval=maxval_2d,
                    repertoire_descriptors=projected_descriptors,
                    ax=ax,
                    # vmin=vmin,
                    # vmax=vmax,
                )

                # Clean up artifacts created by the internal function
                ax.set_title("")
                ax.set_xlabel("")
                ax.set_ylabel("")
                axes_after_call = set(fig.axes)
                newly_added_axes = list(axes_after_call - axes_before_call)
                if newly_added_axes:
                    fig.delaxes(newly_added_axes[0])
            
            # Set labels on the outside edges
            if j == 0:
                ax.set_ylabel(descriptor_labels[i])
            if i == num_descriptors - 1:
                ax.set_xlabel(descriptor_labels[j])

            # Remove tick labels for inner plots
            if j > 0:
                ax.tick_params(axis='y', length=0, labelleft=False)
            if i < num_descriptors - 1:
                ax.tick_params(axis='x', length=0, labelbottom=False)

    # Add a single, clean color bar
    my_cmap = mpl.colormaps["viridis"]
    cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cbar_ax)
    cbar.set_label('Fitness', rotation=270, labelpad=15)
    
    fig.suptitle("MAP-Elites Repertoire Projection", fontsize=16, y=0.98)
    return fig, axes