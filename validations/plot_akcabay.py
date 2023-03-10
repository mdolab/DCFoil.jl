# --- Python 3.9 ---
"""
@File    :   plot_kramer.py
@Time    :   2023/02/06
@Author  :   Galen Ng
@Desc    :   Use the composite beam from Akcabay's 2019 CS paper:
Akcabay, D. T., & Young, Y. L. (2019). Steady and Dynamic Hydroelastic Behavior of Composite Lifting Surfaces. Composite Structures, 227(December 2018), 111240. https://doi.org/10.1016/j.compstruct.2019.111240
to validate DCFoil

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

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots
import seaborn as sns

current = os.path.dirname(os.path.realpath(__file__))  # Getting the parent directory name
# adding the parent directory to the sys. path.
sys.path.append(os.path.dirname(current))
from POSTPROCESSING.helperFuncs import load_jld

# ************************************************
#     Training values from paper
# ************************************************
# Import static hydroelastic solves
from akcabay_src.akcabayData import speed_hbar, hbar_neg15, speed_psi, psi_neg15

fname = "akcabay.pdf"
analysisDir = "../OUTPUT/akcabay_U0-"
speeds = np.arange(1, 40 + 1, 1)

if __name__ == "__main__":

    # --- Read data ---
    for ii, speed in enumerate(speeds):
        dataDir = f"{analysisDir}{speed:.1f}/"
        funcs = json.load(f"{dataDir}/static/funcs.json")

    dosave = not not fname

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
    niceColors = sns.color_palette("tab10")
    # niceColors = matplotlib.cm.get_cmap("tab20").colors
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
    cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create figure object
    fig, axes = plt.subplots(nrows=1, sharex=True, constrained_layout=True, figsize=(11, 8))

    # ---------------------------
    #   Wet modal validation
    # ---------------------------
    ax = axes
    for ii in range(len(wetFreqHz)):
        ax.plot(fiberAngles, wetNatFreqs[:, ii], label=f"Mode {ii+1} (DCFoil)", c=cm[ii])
    for ii in range(len(wetFreqHz)):
        ax.plot(wetThetaDeg[ii], wetFreqHz[ii], label=f"Mode {ii+1} (ABAQUS)", c=cm[ii], alpha=0.5)
    ax.set_xlabel(r"$\theta_f$ [$\degree$]")
    ax.set_ylabel(r"$f$ [Hz]", rotation=0.0, labelpad=50)
    ax.legend(fontsize=15, labelcolor="linecolor", loc="best", frameon=False, ncol=2)
    ax.set_title("Modal validation")

    # Structural modal validation
    for ii in range(len(dryFreqHz)):
        ax.plot(-fiberAngles, structNatFreqs[:, ii], label=f"Mode {ii+1} (DCFoil)", c=cm[ii])
    for ii in range(len(dryFreqHz)):
        ax.plot(dryThetaDeg[ii], dryFreqHz[ii], label=f"Mode {ii+1} (ABAQUS)", c=cm[ii], alpha=0.5)

    ax.annotate("Dry modes", xy=(0.25, 0.5), ha="center", xycoords="axes fraction", size=15, color="gray")
    ax.annotate("Wet modes", xy=(0.75, 0.5), ha="center", xycoords="axes fraction", size=15, color="blue")

    ax.set_ylim(bottom=0.0)

    plt.show(block=(not dosave))
    niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()
