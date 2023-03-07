import matplotlib.pyplot as plt
import numpy as np


def plot_cube_scatter3d(vector_U, cubes_collocations,
                        figsize_opt=(14, 12),
                        cmap_opt="seismic",
                        marker_size_opt=150,
                        alpha_opt=0.75,
                        title_opt="k = 1, N = 10, L = 1",
                        xlabel_opt="X axis",
                        ylabel_opt="Y axis",
                        filename_opt="painting_scalar.png"):
    fig = plt.figure(figsize=figsize_opt)
    ax = plt.axes(projection="3d")
    color_map = plt.get_cmap(cmap_opt)
    scatter_plot = ax.scatter3D(cubes_collocations[:, 0],
                                cubes_collocations[:, 1],
                                cubes_collocations[:, 2],
                                c=vector_U,
                                cmap=color_map,
                                s=marker_size_opt,
                                alpha=alpha_opt)
    plt.colorbar(scatter_plot)
    plt.xlabel(xlabel_opt)
    plt.ylabel(ylabel_opt)
    plt.title(title_opt)
    plt.savefig(filename_opt)
    plt.show()
    return 0