import numpy as np
import matplotlib.pyplot as plt


def plot_1d_result_refr_acoustics(collocations,
                                  vec_result,
                                  vec_refr,
                                  filepath=None,
                                  figsize_arg=None,
                                  dpi=None,
                                  title_arg=None,
                                  xlab=None,
                                  ylab1=None,
                                  ylab2=None):
    if figsize_arg is None:
        figsize_arg = (8, 8)
    if dpi is None:
        dpi = 200
    if title_arg is None:
        title_arg = "Решение одномерной задачи акустики"
    if xlab is None:
        xlab = "Сетка одномерной задачи X"
    if ylab1 is None:
        ylab1 = "Распространение волны в среде"
    if ylab2 is None:
        ylab2 = "Индекс рефракции среды"
    if filepath is None:
        filepath = "result_refr.png"

    plt.figure(figsize=figsize_arg, dpi=dpi)
    plt.title("Решение одномерной задачи акустики")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(collocations, vec_result, 'g-')
    ax2.plot(collocations, vec_refr, 'b-')
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab1, color='g')
    ax2.set_ylabel(ylab2, color='b')
    plt.grid()
    plt.savefig(filepath)