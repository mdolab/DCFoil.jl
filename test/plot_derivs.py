# --- Python 3.9 ---
"""
@File    :   plot_derivs.py
@Time    :   2023/03/21
@Author  :   Galen Ng
@Desc    :   Plot the flutter derivatives
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
import seaborn as sns
from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots

current = os.path.dirname(os.path.realpath(__file__))  # Getting the parent directory name
sys.path.append(os.path.dirname(current))  # adding the parent directory to the sys. path.
from POSTPROCESSING.helperFuncs import load_jld
from POSTPROCESSING.helperPlotFuncs import set_my_plot_settings

# ==============================================================================
#                         COMMON VARS
# ==============================================================================


# ==============================================================================
#                         Helper functions
# ==============================================================================
def plot_adderivs(fname):
    dosave = not not fname
    derivs = load_jld("../../eigenDerivs.jld2")

    # ---------------------------
    #   Load data
    # ---------------------------
    table = []
    data = load_jld("FWDDiff.jld2")
    derivs = np.asarray(data["derivs"])
    steps = np.asarray(data["steps"])
    funcVal = np.asarray(data["funcVal"])

    data = load_jld("CENTDiff.jld2")
    cderivs = np.asarray(data["derivs"])
    csteps = np.asarray(data["steps"])
    cfuncVal = np.asarray(data["funcVal"])

    data = load_jld("FINDiff.jld2")
    diffderivs = np.asarray(data["derivs"])

    # Put it into the table
    for ii, step in enumerate(steps):
        table.append([f"{step:.1e}", f"{derivs[ii]:0.15f}", cderivs[ii]])
    table.append(["f Exact", f"{funcVal:.15f}", cfuncVal])

    # --- Create figure object ---
    fig, axes = plt.subplots(nrows=2, sharex=True, constrained_layout=True, figsize=(14, 10))

    ax = axes[0]
    ax.semilogx(steps, derivs, "o-", c=cm[0], label="FWD diff")
    ax.semilogx(csteps, cderivs, "o-", c=cm[1], label="CENT diff")
    yticks = [-500, 800]
    yticks.append(diffderivs[-1][0])
    ax.axhline(diffderivs[-1][0], c=cm[2], ls="--", label="FinDiff jl")
    ax.set_yticks(yticks)
    ax.set_xlim(left=steps[-1] / 5, right=steps[0] * 5)
    ax.invert_xaxis()
    ax.set_ylabel(r"$\frac{d}{d\theta_f} \left( KS_{fl} \right)$", rotation=0, labelpad=40)
    ax.legend(labelcolor="linecolor", loc="best", frameon=False)
    mathStr = r"$f(\theta_f=15\degree)=$"
    ax.set_title(f"Flutter derivative wrt fiber angle\n{mathStr}{funcVal:.2f}")

    ax = axes[1]
    ax.set_title("Derivative accuracy")

    plt.show(block=(not dosave))
    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()

    headers = ["h", "fwd", "cent"]
    tablefmt = "latex"
    print(tabulate(table, headers, tablefmt=tablefmt, floatfmt=".15f"))


def plot_pkderivs(fname):
    dosave = not not fname

    # ************************************************
    #     Load data
    # ************************************************
    data = load_jld("./FWDDiff-BT2.jld2")
    bt2fdderivs = np.asarray(data["derivs"])

    data = load_jld("./RAD-BT2.jld2")
    bt2adderivs = np.asarray(data["derivs"])

    data = load_jld("./FWDDiff-COMP2.jld2")
    comp2fdderivs = np.asarray(data["derivs"])

    data = load_jld("./RAD-COMP2.jld2")
    comp2adderivs = np.asarray(data["derivs"])

    breakpoint()
    # Create figure object
    fig, axes = plt.subplots(nrows=1, sharex=True, constrained_layout=True, figsize=(14, 10))

    plt.show(block=(not dosave))
    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()


# ==============================================================================
#                         Main driver
# ==============================================================================
if __name__ == "__main__":
    cm, fs_lgd, fs, ls, markers = set_my_plot_settings()

    # # ************************************************
    # #     Basic fiber angle deriv plot
    # # ************************************************
    # fname = "derivs.pdf"

    # plot_adderivs(fname)

    # ************************************************
    #     Compare RAD derivatives
    # ************************************************
    fname = "rad-derivs.pdf"
    plot_pkderivs(fname)
