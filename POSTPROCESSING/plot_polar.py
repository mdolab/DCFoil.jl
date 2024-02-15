# --- Python 3.10 ---
"""
@File    :   plot_polar.py
@Time    :   2024/02/14
@Author  :   Galen Ng
@Desc    :   Polars
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import json
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

# from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots
from helperFuncs import (
    load_jld,
    readlines,
    get_bendingtwisting,
    postprocess_flutterevals,
    find_DivAndFlutterPoints,
)
from helperPlotFuncs import (
    plot_static2d,
    plot_dragbuildup,
    plot_deformedShape,
    plot_naturalModeShapes,
    plot_modeShapes,
    plot_vg_vf_rl,
    plot_wingPlanform,
    plot_dlf,
    plot_forced,
    set_my_plot_settings,
)

# ==============================================================================
#                         Command line arguments
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--cases", type=str, default=None, help="Folder dir")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--is_paper", action="store_true", default=False)
args = parser.parse_args()

# ==============================================================================
#                         COMMON PLOT SETTINGS
# ==============================================================================
dataDir = "../OUTPUT/"
# labels = ["NOFS", "FS"]
labels = ["-15", "+15", "0"]
cm, fs_lgd, fs, ls, markers = set_my_plot_settings(args.is_paper)
alphas = np.arange(-5, 15.5, 0.5)
fiberangles = [0.0, 15.0, -15.0]
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    # Echo the args
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print(30 * "-")

    # ************************************************
    #     I/O
    # ************************************************

    # Output plot directory
    outputDir = f"./PLOTS/{args.cases}/"
    if args.output is not None:
        outputDir += args.output

    # Create output directory if it doesn't exist
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    # ************************************************
    #     Read in results
    # ************************************************
    CLDict = {}
    CDDict = {}
    CLFSDict = {}
    CDFSDict = {}
    SolverOptions = {}

    # Input data read directory
    for fiberang in fiberangles:
        caseDirs = []
        caseFSDirs = []
        if args.cases is not None:
            for alpha in alphas:
                caseDirs.append(
                    dataDir + args.cases + f"/f{fiberang:.1f}_w0.0_alfa{alpha:.2f}"
                )
                caseFSDirs.append(
                    dataDir
                    + args.cases.replace("t-foil", "t-foil-fs")
                    + f"/f{fiberang:.1f}_w0.0_alfa{alpha:.2f}"
                )
        else:
            raise ValueError("Please specify a case to run postprocessing on")

        AlfaList = []
        CLDict[fiberang] = []
        CDDict[fiberang] = []
        for ii, caseDir in enumerate(caseDirs):

            # --- Read in DVDict ---
            DVDict = json.load(open(f"{caseDir}/init_DVDict.json"))
            AlfaList.append(DVDict["α₀"])

            # --- Read in funcs ---
            try:
                funcs = json.load(open(f"{caseDir}/funcs.json"))
            except FileNotFoundError:
                funcs = None
                print("No funcs.json file found...")

            CLDict[fiberang].append(funcs["cl"])
            CD = funcs["cdi"] + funcs["cds"] + funcs["cdpr"] + funcs["cdj"]
            CDDict[fiberang].append(CD)

        CLFSDict[fiberang] = []
        CDFSDict[fiberang] = []
        AlfaList = []
        for ii, caseDir in enumerate(caseFSDirs):

            # --- Read in DVDict ---
            DVDict = json.load(open(f"{caseDir}/init_DVDict.json"))
            AlfaList.append(DVDict["α₀"])

            # --- Read in funcs ---
            try:
                funcs = json.load(open(f"{caseDir}/funcs.json"))
            except FileNotFoundError:
                funcs = None
                print("No funcs.json file found...")

            CLFSDict[fiberang].append(funcs["cl"])
            CD = funcs["cdi"] + funcs["cds"] + funcs["cdpr"] + funcs["cdj"]
            CDFSDict[fiberang].append(CD)

    # ************************************************
    #     Drag polar
    # ************************************************
    fname = f"{outputDir}/drag-polar.pdf"
    dosave = not not fname

    # Create figure object
    fig, axes = plt.subplots(
        nrows=1, ncols=2, sharey=True, constrained_layout=True, figsize=(14, 6)
    )

    for ii, fiberang in enumerate(fiberangles):
        axes[0].plot(
            CDDict[fiberang],
            CLDict[fiberang],
            label=f"$\\theta_f={fiberang}^" + "{\\circ}$",
            c=cm[ii],
        )
        axes[0].plot(
            CDFSDict[fiberang],
            CLFSDict[fiberang],
            # label=f"$\\theta_f={fiberang}^" + "{\\circ}$",
            c=cm[ii],
            ls="--",
        )
        axes[1].plot(
            AlfaList,
            CLDict[fiberang],
            # label=f"$\\theta_f{fiberang}^" + "{\\circ}$",
            c=cm[ii],
        )
        axes[1].plot(
            AlfaList,
            CLFSDict[fiberang],
            # label=f"$\\theta_f{fiberang}^" + "{\\circ}$",
            c=cm[ii],
            ls="--",
        )

    axes[1].axvline(2.0, color="gray", alpha=0.5)
    axes[1].set_xticks([-5, 0, 2, 10, 15])
    axes[0].set_xlim(0.0, 0.5)
    axes[0].set_ylim(-0.25, 1.0)

    axes[0].legend(
        fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1
    )

    axes[0].set_xlabel("$C_D$")
    axes[0].set_ylabel("$C_L$", rotation="horizontal", ha="right")
    axes[1].set_xlabel("$\\alpha_r$ [$^{\\circ}$]")

    plt.show(block=(not dosave))
    # niceplots.all()
    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()
