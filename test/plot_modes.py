# --- Python 3.10 ---
"""
@File    :   test_plotmodes.py
@Time    :   2023/09/26
@Author  :   Galen Ng
@Desc    :   Super basic plotting to see discrepancy in mode shapes
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# from tabulate import tabulate
import seaborn as sns

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots
current = os.path.dirname(os.path.realpath(__file__))  # Getting the parent directory name
# adding the parent directory to the sys. path.
sys.path.append(os.path.dirname(current))
from POSTPROCESSING.helperFuncs import get_bendingtwisting, load_jld, compute_normFactorModeShape
from POSTPROCESSING.helperPlotFuncs import set_my_plot_settings

# ==============================================================================
#                         COMMON PARAMS
# ==============================================================================
FileDirs = {
    "BT2": "../OUTPUT/2023-09-29_akcabay-swept_cfrp_f15.0_w-15.0_bt2",
    "COMP2": "../OUTPUT/2023-09-29_akcabay-swept_cfrp_f15.0_w-15.0_comp2",
}
fname = "test.pdf"

ls = ["-", "--", "-.", ":"]
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    dosave = not not fname

    cm, fs_lgd, fs, ls, markers = set_my_plot_settings()

    # --- Load data ---
    plotDryBendData = {}
    plotDryTwistData = {}
    dryNormFactor = {}
    dryFreqData = {}
    plotWetBendData = {}
    plotWetTwistData = {}
    wetNormFactor = {}
    wetFreqData = {}
    for key, val in FileDirs.items():
        jldFile = val + f"/modal/structModal.jld2"
        data = load_jld(jldFile)
        dryPyData = np.asarray(data["structModeShapes"])
        structnatData = np.asarray(data["structNatFreqs"])
        jldFile = val + f"/modal/wetModal.jld2"
        data = load_jld(jldFile)
        wetPyData = np.asarray(data["wetModeShapes"])
        wetnatData = np.asarray(data["wetNatFreqs"])

        NMODE = 3

        plotDryBendData[key] = []
        plotDryTwistData[key] = []
        plotWetBendData[key] = []
        plotWetTwistData[key] = []
        wetFreqData[key] = wetnatData
        dryFreqData[key] = structnatData

        if key == "COMP2":
            nDOF = 9
        else:
            nDOF = 4

        for mm in range(NMODE):
            bend, twist = get_bendingtwisting(dryPyData[mm, :], nDOF=nDOF)
            plotDryBendData[key].append(bend)
            plotDryTwistData[key].append(twist)
            bend, twist = get_bendingtwisting(wetPyData[mm, :], nDOF=nDOF)
            plotWetBendData[key].append(bend)
            plotWetTwistData[key].append(twist)

        # Mode shapes
        plotDryBendData[key] = np.array(plotDryBendData[key])
        plotDryTwistData[key] = np.array(plotDryTwistData[key])
        plotWetBendData[key] = np.array(plotWetBendData[key])
        plotWetTwistData[key] = np.array(plotWetTwistData[key])

        # Normalization factor on mode shapes
        ModeShapeDict = {
            "bend": plotDryBendData[key],
            "twist": plotDryTwistData[key],
        }
        dryNormFactor[key] = compute_normFactorModeShape(ModeShapeDict)
        print(f"Norm factor for {key} is {dryNormFactor[key]}")
        ModeShapeDict = {
            "bend": plotWetBendData[key],
            "twist": plotWetTwistData[key],
        }
        wetNormFactor[key] = compute_normFactorModeShape(ModeShapeDict)

    # Create figure object
    fig, axes = plt.subplots(nrows=NMODE, ncols=2, sharex=True, constrained_layout=True, figsize=(14, 10))

    for mm in range(NMODE):
        ax = axes[mm, 0]

        elemCtr = 0
        for k, v in FileDirs.items():
            ax.plot(plotDryBendData[k][mm, :] / dryNormFactor[k][mm], ls=ls[0], c=cm[elemCtr], alpha=0.5)
            ax.plot(plotDryTwistData[k][mm, :] / dryNormFactor[k][mm], ls=ls[1], c=cm[elemCtr], label=key, alpha=0.5)
            ax.annotate(
                f"{k} ({dryFreqData[k][mm]:.2f}Hz)",
                c=cm[elemCtr],
                xy=(0.01, 0.01 + elemCtr * 0.2),
                xycoords="axes fraction",
            )
            elemCtr += 1

        ax.set_ylabel("Normalized Dry\nMode Shape", rotation="horizontal", ha="right")
        ax.set_title(f"Mode {mm+1}")
        ax.set_ylim(-1, 1)
        # ax.set_xlim(0, 29)

        ax = axes[mm, 1]
        elemCtr = 0
        for k, v in FileDirs.items():
            ax.plot(plotWetBendData[k][mm, :] / wetNormFactor[k][mm], ls=ls[0], c=cm[elemCtr], label=key, alpha=0.5)
            ax.plot(plotWetTwistData[k][mm, :] / wetNormFactor[k][mm], ls=ls[1], c=cm[elemCtr], label=key, alpha=0.5)
            ax.annotate(
                f"{k} ({wetFreqData[k][mm]:.2f}Hz)",
                c=cm[elemCtr],
                xy=(0.01, 0.01 + elemCtr * 0.2),
                xycoords="axes fraction",
            )
            elemCtr += 1

        ax.set_ylabel("Wet Mode\nShape", rotation="horizontal", ha="right")
        ax.set_title(f"Mode {mm+1}")
        ax.set_ylim(-1, 1)
        # ax.set_xlim(0, 29)

    plt.show(block=(not dosave))
    # niceplots.all()
    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()
