import numpy as np
import numba as nb


@nb.njit(fastmath=True)
def step_refr_3d_sphere(collocations_3d, center=np.array([0.0, 0.0, 0.0]), radius=1.0, refraction=1.0 + 1.0j):
    return (np.sqrt((collocations_3d - center).dot((collocations_3d - center).T)) <= radius) * refraction


@nb.njit(fastmath=True)
def step_refr_3d_cube(collocations_3d, center=np.array([0.0, 0.0, 0.0]), radius=1.0, refraction=1.0 + 1.0j):
    return (np.sum(np.abs(collocations_3d - center)) <= radius) * refraction