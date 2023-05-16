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
import h5py
import seaborn as sns
from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots


def load_jld(filename: str):
    """
    Load data from a .jld file
    f is a dictionary
    """
    f = h5py.File(filename, "r")
    return f


# ==============================================================================
#                         Main driver
# ==============================================================================

derivs = load_jld("../../eigenDerivs.jld")
breakpoint()
if __name__ == "__main__":
    fname = "derivs.pdf"

    dosave = not not fname

    plt.style.use(niceplots.get_style())  # all settings
    # --- Adjust default options for matplotlib ---
    myOptions = {
        "font.size": 25,
        "font.family": "sans-serif",  # set to "serif" to get the same as latex
        # "font.sans-serif": ["Helvetica"],  # this does not work on all systems
        "text.usetex": False,  # use external latex for all text
        "text.latex.preamble": [
            r"\usepackage{lmodern}",  # latin modern font
            r"\usepackage{amsmath}",  # for using equation commands
            r"\usepackage{helvet}",  # should make latex serif in helvet now
            r"\usepackage{sansmath}",
            r"\sansmath",  # supposed to force math to be rendered in serif font
        ],
    }
    plt.rcParams.update(myOptions)
    niceColors = sns.color_palette("tab10")
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
    cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # ---------------------------
    #   Load data
    # ---------------------------
    table = []
    data = load_jld("FWDDiff.jld")
    derivs = np.asarray(data["derivs"])
    steps = np.asarray(data["steps"])
    funcVal = np.asarray(data["funcVal"])

    data = load_jld("CENTDiff.jld")
    cderivs = np.asarray(data["derivs"])
    csteps = np.asarray(data["steps"])
    cfuncVal = np.asarray(data["funcVal"])

    data = load_jld("FINDiff.jld")
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
