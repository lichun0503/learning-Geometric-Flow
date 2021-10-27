# coding=utf-8

import torch
import numpy as np
import matplotlib as mpl


def figsize(scale, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


# I make my own newfig and savefig functions
def newfig(width, nplots=1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    if crop == True:
        #        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        #        plt.savefig('{}.pgf'.format(filename))
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))




def AC_2D_init(x):
    ## bubble
    r = 0.2 * np.sqrt(2)
    epsilon = 0.01
    a_x = - r / np.sqrt(2)
    a_y = r / np.sqrt(2)
    b_x = r / np.sqrt(2)
    b_y = -r / np.sqrt(2)
    #u_init = np.tanh((0.2 - np.sqrt((x[:, 1] - 0.3) ** 2 + (x[:, 2] - 0.5) ** 2)) / (np.sqrt(2) * 0.03))
    #u_init = np.tanh((x[:, 2] - H_o*np.cos(k*x[:, 1]))/np.sqrt(2)/eta)
    #u_init = (10 * np.maximum((0.04 - (x[:, 1] - 0.2) ** 2 - (x[:, 2] - 0.65) ** 2), 0) + \
    #          12 * np.maximum((0.03 - (x[:, 1] - 0.5) ** 2 - (x[:, 2] - 0.2) ** 2), 0) + \
    #          +12 * np.maximum((0.03 - (x[:, 1] - 0.8) ** 2 - (x[:, 2] - 0.55) ** 2), 0) + \
    #         np.tanh((0.2 - np.sqrt((x[:, 1] - 0.3) ** 2 + (x[:, 2] - 0.3) ** 2)) / (np.sqrt(2) * 0.03))+ \
    #          np.tanh((0.2 - np.sqrt((x[:, 1] - 0.3) ** 2 + (x[:, 2] - 0.5) ** 2)) / (np.sqrt(2) * 0.03))) * \
    #       np.tanh((0.2 - np.sqrt((x[:, 1] - 0.3) ** 2 + (x[:, 2] - 0.5) ** 2)) / (np.sqrt(2) * 0.03))
    u_init = np.tanh((-r + np.sqrt((x[:, 1] - a_x) ** 2 + (x[:, 2] - a_y) ** 2))) / 2 * np.sqrt(2) * epsilon + \
            0.5 * np.tanh((-r + np.sqrt((x[:, 1] - b_x) ** 2 + (x[:, 2] - b_y) ** 2))) / 2 * np.sqrt(2) * epsilon
    noise = 0.9
    u_init = u_init + noise * np.std(u_init) * np.random.randn(u_init.shape[0])
    return u_init


# from plotting import newfig, savefig

def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs),
                               create_graph=True)


def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))



