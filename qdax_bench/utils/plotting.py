from typing import Dict, Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire
from qdax.utils.plotting import get_voronoi_finite_polygons_2d

# Customize matplotlib params
font_size = 20
mpl_params = {
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "font.size": font_size,
    "text.usetex": False,
    "axes.titlepad": 10,
}

# def plot_2d_map_elites_repertoire(
#     centroids: jnp.ndarray,
#     repertoire_fitnesses: jnp.ndarray,
#     minval: jnp.ndarray,
#     maxval: jnp.ndarray,
#     title: Optional[str] = "MAP-Elites Grid",
#     repertoire_descriptors: Optional[jnp.ndarray] = None,
#     ax: Optional[plt.Axes] = None,
#     vmin: Optional[float] = None,
#     vmax: Optional[float] = None,
#     colormap: str = "viridis",
# ) -> Tuple[Optional[Figure], Axes]:
#     """Plot a visual representation of a 2d map elites repertoire.

#     TODO: Use repertoire as input directly. Because this
#     function is very specific to repertoires.

#     Args:
#         centroids: the centroids of the repertoire
#         repertoire_fitnesses: the fitness of the repertoire
#         minval: minimum values for the descritors
#         maxval: maximum values for the descriptors
#         repertoire_descriptors: the descriptors. Defaults to None.
#         ax: a matplotlib axe for the figure to plot. Defaults to None.
#         vmin: minimum value for the fitness. Defaults to None. If not given,
#             the value will be set to the minimum fitness in the repertoire.
#         vmax: maximum value for the fitness. Defaults to None. If not given,
#             the value will be set to the maximum fitness in the repertoire.

#     Raises:
#         NotImplementedError: does not work for descriptors dimension different
#         from 2.

#     Returns:
#         A figure and axes object, corresponding to the visualisation of the
#         repertoire.
#     """

#     # TODO: check it and fix it if needed
#     grid_empty = repertoire_fitnesses == -jnp.inf
#     num_descriptors = centroids.shape[1]
#     if num_descriptors != 2:
#         raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

#     my_cmap = cm.get_cmap(colormap)

#     fitnesses = repertoire_fitnesses
#     if vmin is None:
#         vmin = float(jnp.min(fitnesses[~grid_empty]))
#     if vmax is None:
#         vmax = float(jnp.max(fitnesses[~grid_empty]))


#     mpl.rcParams.update(mpl_params)

#     # create the plot object
#     fig = None
#     if ax is None:
#         fig, ax = plt.subplots(facecolor="white", edgecolor="white")

#     assert (
#         len(jnp.array(minval).shape) < 2
#     ), f"minval : {minval} should be float or couple of floats"
#     assert (
#         len(jnp.array(maxval).shape) < 2
#     ), f"maxval : {maxval} should be float or couple of floats"

#     if len(jnp.array(minval).shape) == 0 and len(jnp.array(maxval).shape) == 0:
#         ax.set_xlim(minval, maxval)
#         ax.set_ylim(minval, maxval)
#     else:
#         ax.set_xlim(minval[0], maxval[0])
#         ax.set_ylim(minval[1], maxval[1])

#     ax.set(adjustable="box", aspect="equal")

#     # create the regions and vertices from centroids
#     regions, vertices = get_voronoi_finite_polygons_2d(centroids)

#     norm = Normalize(vmin=vmin, vmax=vmax)

#     # fill the plot with contours
#     for region in regions:
#         polygon = vertices[region]
#         ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

#     # fill the plot with the colors
#     for idx, fitness in enumerate(fitnesses):
#         if fitness > -jnp.inf:
#             region = regions[idx]
#             polygon = vertices[region]

#             ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

#     # if descriptors are specified, add points location
#     if repertoire_descriptors is not None:
#         descriptors = repertoire_descriptors[~grid_empty]
#         ax.scatter(
#             descriptors[:, 0],
#             descriptors[:, 1],
#             c=fitnesses[~grid_empty],
#             cmap=my_cmap,
#             s=10,
#             zorder=0,
#         )

#     # aesthetic
#     ax.set_xlabel("Behavior Dimension 1")
#     ax.set_ylabel("Behavior Dimension 2")
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
#     cbar.ax.tick_params(labelsize=font_size)

#     ax.set_title(title)
#     ax.set_aspect("equal")

#     return fig, ax

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
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))

    # env_steps = jnp.arange(num_iterations) * episode_length * batch_size

    axes[0].plot(env_steps, metrics["coverage"])
    axes[0].set_xlabel(x_label)
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[0].set_ylabel("Coverage in %")
    axes[0].set_title("Coverage evolution during training")
    axes[0].set_aspect(0.95 / axes[0].get_data_ratio(), adjustable="box")

    axes[1].plot(env_steps, metrics["max_fitness"])
    axes[1].set_xlabel(x_label)
    axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    axes[1].set_ylabel("Maximum fitness")
    axes[1].set_title("Maximum fitness evolution during training")
    axes[1].set_aspect(0.95 / axes[1].get_data_ratio(), adjustable="box")

    axes[2].plot(env_steps, metrics["qd_score"])
    axes[2].set_xlabel(x_label)
    axes[2].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
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