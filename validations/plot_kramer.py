# --- Python 3.9 ---
"""
@File    :   plot_kramer.py
@Time    :   2023/02/06
@Author  :   Galen Ng
@Desc    :   Use the composite beam from Kramer's CS paper:
Kramer, M. R., Liu, Z., Young, Y. L. (2013). Free vibration of cantilevered composite plates in air and in water. Composite Structures
to validate DCFoil

The reference values were computed in:
Liao, Y., Garg, N., Martins, J. R. R. A., Young, Y. L. (2019). Viscous fluid–structure interaction response of composite hydrofoils. Composite Structures

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
from kramer_src.kramerData import wetFreqHz, wetThetaDeg, dryFreqHz, dryThetaDeg

fname = "kramer.pdf"
analysisDir = "../OUTPUT/kramer_theta-"
fiberAngles = np.arange(0, 90 + 10, 10)
nModes = 5  # number of modes analyzed in DCFoil

if __name__ == "__main__":

    # --- Read data ---
    structNatFreqs = np.zeros((len(fiberAngles), nModes))
    wetNatFreqs = np.zeros((len(fiberAngles), nModes))
    for ii, fiber in enumerate(fiberAngles):
        dataDir = f"{analysisDir}{fiber:.1f}/"
        structJLData = load_jld(f"{dataDir}/modal/structModal.jld")
        wetJLData = load_jld(f"{dataDir}/modal/wetModal.jld")

        # NOTE: Julia stores data in column major order so it is transposed
        structNatFreqs[ii, :] = np.asarray(structJLData["structNatFreqs"])
        wetNatFreqs[ii, :] = np.asarray(wetJLData["wetNatFreqs"])

    dosave = not not fname

    plt.style.use(niceplots.get_style())  # all settings
    # --- Adjust default options for matplotlib ---
    myOptions = {
        "font.size": 35,
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
        ax.plot(fiberAngles, wetNatFreqs[:, ii], label=f"Mode {ii+1}", c=cm[ii])
    for ii in range(len(wetFreqHz)):
        ax.plot(
            wetThetaDeg[ii],
            wetFreqHz[ii],
            # label=f"Mode {ii+1} (ABAQUS)",
            c=cm[ii],
            alpha=0.5,
        )
    ax.set_xlabel(r"$\theta_f$ [$\degree$]")
    ax.set_ylabel(r"$f$ [Hz]", rotation=0.0, labelpad=50)
    ax.legend(fontsize=25, labelcolor="linecolor", loc="best", frameon=False, ncol=1)
    ax.set_title("Modal validation")

    # Structural modal validation
    for ii in range(len(dryFreqHz)):
        ax.plot(-fiberAngles, structNatFreqs[:, ii], label=f"Mode {ii+1} (DCFoil)", c=cm[ii])
    for ii in range(len(dryFreqHz)):
        ax.plot(dryThetaDeg[ii], dryFreqHz[ii], label=f"Mode {ii+1} (ABAQUS)", c=cm[ii], alpha=0.5)

    ax.annotate("Dry modes", xy=(0.25, 0.5), ha="center", xycoords="axes fraction", size=25, color="gray", alpha=0.5)
    ax.annotate("Wet modes", xy=(0.75, 0.5), ha="center", xycoords="axes fraction", size=25, color="blue", alpha=0.5)

    ax.set_ylim(bottom=0.0)

    plt.show(block=(not dosave))
    niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()
