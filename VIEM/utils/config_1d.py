import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import numba as nb
from VIEM.shapes.linear_1d import linear_grid
from VIEM.refractions.refractions_1d import step_refr_1d, gauss_refr_1d
from VIEM.utils.save_load_vecs import save_np_file_txt, load_np_file_txt
from VIEM.waves.waves_1d import wave_narmonic_1d
from VIEM.kernels.kernels_1d import kernel_helmholtz_1d_pos
from VIEM.iterations.two_sdg_symmetric_1d import TwoSGD_nu_1d_sim
from VIEM.visualisation.visual_1d import plot_1d_result_refr_acoustics


class Problem1dAcoustic:
    """
    Configuration class for solving 1d acoustic problem

    :var _filepath: str variable for configuration file from **filename** argument
    :var _dir_path:
    :var _arrays_path:
    :var _images_path:
    :var _lower:
    :var _upper:
    :var _k:
    :var _e:
    :var _n:
    :var _orientation:
    :var _refr_opts:
    """
    def __init__(self, filename=None):
        self._filepath = filename
        self._dir_path = "./result_dir"
        self._arrays_path = self._dir_path + "/array_result"
        self._images_path = self._dir_path + "/image_result"
        self._lower = -1.0
        self._upper = 1.0
        self._k = 1.0
        self._e = 1.0
        self._n = 10
        self._orientation = 1.0
        self._refr_opts = {
            "refr_1": {
                "type": "step",
                "lower": -0.6,
                "upper": 0.7,
                "refr_real": 1.0,
                "refr_imag": 1.0
            }
        }
        # Вызов получения данных из конфига
        self._parse_config()
        self._set_tree_dirs()

        self.collocations = None
        self.refractions_colloc = None
        self.free_vec = None
        self.matrix_row = None
        self.result = None
        self.iterations = None
        self.accuracy_on_iter = None
        self.cell_width = None

    def _parse_config(self):
        with open(self._filepath, "r") as jsonfile:
            data_config = json.load(jsonfile)

        if data_config is None:
            return False

        self._dir_path = data_config.get("dir_path")
        self._arrays_path = self._dir_path + "/array_result"
        self._images_path = self._dir_path + "/image_result"
        self._lower = data_config.get("lower")
        self._upper = data_config.get("upper")
        self._k = data_config.get("k")
        self._e = data_config.get("e")
        self._n = data_config.get("n")
        self._orientation = data_config.get("orientation")
        self._refr_opts = data_config.get("refr_opts")
        return True

    def _set_tree_dirs(self):
        if os.path.isdir(self._dir_path):
            shutil.rmtree(self._dir_path)
        os.mkdir(self._dir_path)
        os.mkdir(self._arrays_path)
        os.mkdir(self._images_path)
        return True

    @staticmethod
    @nb.jit(fastmath=True, forceobj=True)
    def _refr_compute(collocations, n, refr_opts):
        refraction = np.zeros((n,), complex)
        for i in nb.prange(len(refr_opts)):
            dict = refr_opts.get("refr_" + str(i + 1))
            if dict.get("type") == "step":
                refraction += step_refr_1d(collocations_1d=collocations,
                                           low_bound=dict.get("lower"),
                                           upper_bound=dict.get("upper"),
                                           refraction=dict.get("refr_real") + 1.0j * dict.get("refr_imag"))
            elif dict.get("type") == "gauss":
                refraction += gauss_refr_1d(collocations_1d=collocations,
                                            mean_real=dict.get("mean_real"),
                                            std_real=dict.get("std_real"),
                                            mean_imag=dict.get("mean_imag"),
                                            std_imag=dict.get("std_imag"))
        return refraction

    def set_problem(self, filename=None):
        self.collocations, _, self.cell_width = linear_grid(self._lower, self._upper, self._n)
        self.refractions_colloc = self._refr_compute(self.collocations, self._n, self._refr_opts)
        self.free_vec = wave_narmonic_1d(self.collocations, self._k, self._e, self._orientation)
        self.matrix_row = -1.0 * (self._k**2) * kernel_helmholtz_1d_pos(x=self.collocations[0, None],
                                                                        y=self.collocations[None, :],
                                                                        k=self._k)[0] * self.cell_width
        return True

    def compute_problem(self):
        self.result, self.iterations, self.accuracy_on_iter = TwoSGD_nu_1d_sim(matrix_A=self.matrix_row,
                                                                               vector_f=self.free_vec,
                                                                               vector_nu=self.refractions_colloc)
        return self.result, self.iterations, self.accuracy_on_iter

    def save_results(self):
        # Запись эксперимента------------------------------------
        with open(self._dir_path + "/experiment_info.txt", "w") as result_file:
            result_file.write(f"Одномерная задача распространения акустической волны в среде с неоднородным индексом рефракции\n")
            result_file.write(f"Одномерная среда задана отрезком [{self._lower}, {self._upper}]\n")
            result_file.write(f"Количество разбиений среды N = {self._n}\n")
            result_file.write(f"Шаг сетки h = {(self._upper - self._lower) / self._n}\n\n")
            result_file.write(f"Направление волны d = {np.sign(self._orientation)}\n")
            result_file.write(f"Значение волнового числа k = {self._k}\n")
            result_file.write(f"Значение аплитуды волны E0 = {self._e}\n\n")
            result_file.write(f"Среда характеризуется индексом рефракции:\n")
            result_file.write(f"{self._refr_opts}")
            result_file.write(f"\n-----------------------------------\nРезультаты итераций:\n")
            for iter in range(len(self.iterations)):
                result_file.write(f"Умножений матрицы на вектор k = {self.iterations[iter]}, точность на итерации acc = {self.accuracy_on_iter[iter]}\n")
            result_file.write("-----------------------------------")

        # Численное решение задачи в точках коллокации ----------
        save_np_file_txt(array_numpy=self.result,
                         filename=self._arrays_path + "/result.txt")
        # Визуализация результата -------------------------------
        plot_1d_result_refr_acoustics(collocations=self.collocations,
                                      vec_result=np.real(self.result),
                                      vec_refr=np.real(self.refractions_colloc),
                                      filepath=self._images_path + "/result_refr_real.png",
                                      ylab1="Распросранение действительной части волны в среде",
                                      ylab2="Действительный индекс рефракции среды",
                                      title_arg="Решение одномерной задачи акустики с неоднородным индексом рефракции")

        plot_1d_result_refr_acoustics(collocations=self.collocations,
                                      vec_result=np.imag(self.result),
                                      vec_refr=np.imag(self.refractions_colloc),
                                      filepath=self._images_path + "/result_refr_imag.png",
                                      ylab1="Распросранение мнимой части волны в среде",
                                      ylab2="Мнимый индекс рефракции среды",
                                      title_arg="Решение одномерной задачи акустики с неоднородным индексом рефракции")

        # Визуализация сходимости итераций ----------------------
        plt.figure(figsize=(8, 8), dpi=200)
        plt.title("График сходимости итераций по метрике останова")
        plt.plot(self.iterations, self.accuracy_on_iter, color="#00AA11", linestyle="-", linewidth=1, label="Модиф. град.")
        plt.xlabel("Умножения матрицы на вектор")
        plt.ylabel("Метрика останова итераций")
        plt.legend()
        plt.grid()
        plt.savefig(self._images_path + "/result_iterations.png")





