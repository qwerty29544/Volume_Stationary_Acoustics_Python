import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def linear_grid(lower=-1.0, upper=1.0, number_of_cells=10):
    """
    Linear 1d grid for 1d problems

    :param lower: lower real value for left boundary of space
    :param upper: upper real value for right boundary of space
    :param number_of_cells: number of collocation points or cells
    :return: tuple(colloc, grid, h):
    :colloc: **np.ndarray** with **shape=(number_of_cells,)** - array of collocations
    :grid: **np.ndarray** with **shape=(number_of_cells + 1,)** - array of grid boundaries
    :h: real value of cell width
    """
    x_grid = np.linspace(lower, upper, number_of_cells + 1)
    x_colloc = (x_grid[1:] + x_grid[:-1]) / 2
    h = np.abs(upper - lower) / number_of_cells
    return x_colloc, x_grid, h