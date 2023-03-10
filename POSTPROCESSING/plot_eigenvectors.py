# --- Python 3.9 ---
"""
@File    :   plot_eigenvectors.py
@Time    :   2023/02/16
@Author  :   Galen Ng
@Desc    :   Debug plotting script that probably won't get used in the final code
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys
import copy
import json

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots
import seaborn as sns
from helperFuncs import load_jld


plt.style.use(niceplots.get_style())  # all settings
# --- Adjust default options for matplotlib ---
myOptions = {
    "font.size": 20,
    "font.family": "sans-serif",  # set to "serif" to get the same as latex
    # "font.sans-serif": ["Helvetica"], # this does not work on all systems
    # "text.usetex": True, # use external latex for all text
    "text.latex.preamble": [
        r"\usepackage{lmodern}",  # latin modern font
        r"\usepackage{amsmath}",  # for using equation commands
        r"\usepackage{helvet}",  # should make latex serif in helvet now
        r"\usepackage{sansmath}",
        r"\sansmath",  # supposed to force math to be rendered in serif font
    ],
}
plt.rcParams.update(myOptions)

# ==============================================================================
#                         Main driver
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="output dir", type=str, default="../DebugOutput/")
    parser.add_argument("--nflow", help="number of flows", type=int, default=1)
    args = parser.parse_args()

    dataDir = "../DebugOutput/"

    # for ax in axes:
    #     niceplots.adjust_spines(ax, outward=True)
    # plt.savefig(fname, format="pdf")
    # print("Saved to:", fname)
    # plt.close()
    cm = sns.color_palette("crest", n_colors=args.nflow)
    # ************************************************
    #     Read data from file
    # ************************************************
    # Remember julia is transposed from python
    f = load_jld(dataDir + f"sorted-eigenvectors-002.jld")
    sorted_evec_r = np.asarray(f["evec_r"]).T
    nmodes = sorted_evec_r.shape[1]  # number of modes is number of columns
    nNodes = sorted_evec_r.shape[0] // 2  # number of eigenvalues is number of rows
    ybar = np.linspace(0, 1, nNodes)
    fig, axes = plt.subplots(
        nrows=nmodes, ncols=2, sharex=True, sharey=True, constrained_layout=True, figsize=(14, 5 * nmodes)
    )
    for nFlow in range(args.nflow):
        f = load_jld(dataDir + f"sorted-eigenvectors-{nFlow+1:03d}.jld")
        sorted_evec_r = np.asarray(f["evec_r"]).T
        sorted_evec_i = np.asarray(f["evec_i"]).T
        for mode in range(nmodes):
            ax = axes[mode, 0]
            ax.plot(ybar, sorted_evec_r[:nNodes, mode], label="Real", c=cm[nFlow])
            ax.set_ylabel(f"Mode {mode+1}", rotation=0, labelpad=40)
            ax = axes[mode, 1]
            ax.plot(ybar, sorted_evec_i[:nNodes, mode], label="Imag", c=cm[nFlow])
    axes[0, 0].set_title("Real")
    axes[0, 1].set_title("Imag")
    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
    fname = "eigenvector-debug.pdf"
    plt.savefig(f"{args.output}/{fname}", format="pdf")
    plt.close()

    # ---------------------------
    #   Plot corr matrix
    # ---------------------------
    # for nFlow in range(args.nflow - 1):
    #     f = load_jld(dataDir + f"corrMatrix-{nFlow+2:03d}.jld")
    #     corr = np.asarray(f["corrMat"]).T
    #     ax = sns.heatmap(corr, cmap="viridis", vmin=0, vmax=1)
    #     ax.set_title(f"Correlation matrix")
    #     ax.set_xlabel("New modes")
    #     ax.set_ylabel("Old\nmodes", rotation=0, labelpad=40)
    #     ax.set_aspect("equal")
    #     plt.savefig(f"{args.output}/corr-{nFlow+2:03d}.pdf", format="pdf")
    #     plt.close()
