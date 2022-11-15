import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from VIEM.refractions.refractions_1d import step_refr_1d


if __name__ == "__main__":
    grid_1d = np.linspase(-2.0, 2.0, 20)
    x_vec = step_refr_1d(grid_1d)
    print(x_vec)