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
parser.add_argument("--cases", type=str, default=[], nargs="+", help="Full case folder names in order")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--is_paper", action="store_true", default=False)
args = parser.parse_args()

# ==============================================================================
#                         COMMON PLOT SETTINGS
# ==============================================================================
dataDir = "../OUTPUT/"
# labels = ["NOFS", "FS"]
labels = ["-15", "0", "+15"]
compName = "rudder"
cm, fs_lgd, fs, ls, markers = set_my_plot_settings(args.is_paper)
alphas = np.arange(-5, 15.5, 0.5)
fiberangles = [-15, 0.0, 15.0]

# ************************************************
#     EXPERIMENTAL DATA
# ************************************************
CLCDX = [
    0.015514018691588782,
    0.01775700934579439,
    0.022242990654205604,
    0.02785046728971962,
    0.031495327102803734,
    0.038130841121495326,
    0.04514018691588785,
    0.05523364485981308,
]
CLCDY = [
    0.013011152416356864,
    0.08884758364312273,
    0.19144981412639406,
    0.2568773234200744,
    0.35650557620817847,
    0.4204460966542751,
    0.5423791821561339,
    0.6241635687732342,
]

CLALPHAX = [
    -0.1904761904761907,
    0.8095238095238093,
    1.746031746031746,
    2.7777777777777777,
    3.761904761904762,
    4.809523809523809,
    5.7777777777777795,
    6.746031746031746,
]

CLALPHAY = [
    0.010905730129390001,
    0.0878003696857671,
    0.18835489833641406,
    0.254898336414048,
    0.3554528650646949,
    0.41903881700554524,
    0.5388170055452863,
    0.6231053604436227,
]
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
    outputDir = f"./PLOTS/{args.cases[0]}/"
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
                caseDirs.append(dataDir + args.cases[0] + f"/f{fiberang:.1f}_w0.0_alfa{alpha:.2f}")
                caseFSDirs.append(
                    dataDir + args.cases[0].replace("t-foil", "t-foil-fs") + f"/f{fiberang:.1f}_w0.0_alfa{alpha:.2f}"
                )
        else:
            raise ValueError("Please specify a case to run postprocessing on")

        AlfaList = []
        CLDict[fiberang] = []
        CDDict[fiberang] = []
        for ii, caseDir in enumerate(caseDirs):

            # --- Read in DVDict ---
            try:
                DVDict = json.load(open(f"{caseDir}/init_DVDict.json"))
            except FileNotFoundError:
                DVDict = json.load(open(f"{caseDir}/init_DVDict-comp001.json"))
            AlfaList.append(DVDict["α₀"])

            # --- Read in funcs ---
            try:
                funcs = json.load(open(f"{caseDir}/funcs.json"))
            except FileNotFoundError:
                funcs = None
                print("No funcs.json file found...")

            CLDict[fiberang].append(funcs[f"cl-{compName}"])
            CD = funcs[f"cdi-{compName}"] + funcs[f"cds-{compName}"] + funcs[f"cdpr-{compName}"] + funcs[f"cdj-{compName}"]
            CDDict[fiberang].append(CD)

        CLFSDict[fiberang] = []
        CDFSDict[fiberang] = []
        AlfaList = []
        for ii, caseDir in enumerate(caseFSDirs):

            # --- Read in DVDict ---
            try:
                DVDict = json.load(open(f"{caseDir}/init_DVDict.json"))
            except FileNotFoundError:
                DVDict = json.load(open(f"{caseDir}/init_DVDict-comp001.json"))
            AlfaList.append(DVDict["α₀"])

            # --- Read in funcs ---
            try:
                funcs = json.load(open(f"{caseDir}/funcs.json"))
            except FileNotFoundError:
                funcs = None
                print("No funcs.json file found...")

            CLFSDict[fiberang].append(funcs[f"cl-{compName}"])
            CD = funcs[f"cdi-{compName}"] + funcs[f"cds-{compName}"] + funcs[f"cdpr-{compName}"] + funcs[f"cdj-{compName}"]
            CDFSDict[fiberang].append(CD)

    # Rigid results
    caseDirs = []
    caseFSDirs = []
    for alpha in alphas:
        caseDirs.append(dataDir + args.cases[1] + f"/f0.0_w0.0_alfa{alpha:.2f}")
        caseFSDirs.append(
            dataDir + args.cases[1].replace("t-foil", "t-foil-fs") + f"/f0.0_w0.0_alfa{alpha:.2f}"
        )

    AlfaList = []
    CLDict["rigid"] = []
    CDDict["rigid"] = []
    for ii, caseDir in enumerate(caseDirs):

        # --- Read in DVDict ---
        try:
            DVDict = json.load(open(f"{caseDir}/init_DVDict.json"))
        except FileNotFoundError:
            DVDict = json.load(open(f"{caseDir}/init_DVDict-comp001.json"))
        AlfaList.append(DVDict["α₀"])

        # --- Read in funcs ---
        try:
            funcs = json.load(open(f"{caseDir}/funcs.json"))
        except FileNotFoundError:
            funcs = None
            print("No funcs.json file found...")

        CLDict["rigid"].append(funcs[f"cl-{compName}"])
        CD = funcs[f"cdi-{compName}"] + funcs[f"cds-{compName}"] + funcs[f"cdpr-{compName}"] + funcs[f"cdj-{compName}"]
        CDDict["rigid"].append(CD)

    CLFSDict["rigid"] = []
    CDFSDict["rigid"] = []
    AlfaList = []
    for ii, caseDir in enumerate(caseFSDirs):

        # --- Read in DVDict ---
        try:
            DVDict = json.load(open(f"{caseDir}/init_DVDict.json"))
        except FileNotFoundError:
            DVDict = json.load(open(f"{caseDir}/init_DVDict-comp001.json"))
        AlfaList.append(DVDict["α₀"])

        # --- Read in funcs ---
        try:
            funcs = json.load(open(f"{caseDir}/funcs.json"))
        except FileNotFoundError:
            funcs = None
            print("No funcs.json file found...")

        CLFSDict["rigid"].append(funcs[f"cl-{compName}"])
        CD = funcs[f"cdi-{compName}"] + funcs[f"cds-{compName}"] + funcs[f"cdpr-{compName}"] + funcs[f"cdj-{compName}"]
        CDFSDict["rigid"].append(CD)

    # ************************************************
    #     Drag polar
    # ************************************************
    fname = f"{outputDir}/drag-polar.pdf"
    dosave = not not fname

    # Create figure object
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, constrained_layout=True, figsize=(14, 6))

    for ii, fiberang in enumerate(fiberangles):
        axes[0].plot(
            CDDict[fiberang],
            CLDict[fiberang],
            label=f"$\\theta_f={fiberang}^" + "{\\circ}$",
            c=cm[ii],
        )
        cdtick = np.min(CDDict[fiberang])
        axes[0].plot(
            CDFSDict[fiberang],
            CLFSDict[fiberang],
            # label=f"$\\theta_f={fiberang}^" + "{\\circ}$",
            c=cm[ii],
            ls="--",
            # alpha=0.3,
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
            # alpha=0.3,
        )

    axes[0].plot(CLCDX, CLCDY, "ko", label="Experiment (Ref. 1)")
    axes[1].plot(CLALPHAX, CLALPHAY, "ko", label="Exp.")
    axes[1].axvline(2.0, color="gray", alpha=0.5)
    axes[1].set_xticks([-5, 0, 2, 10, 15])
    axes[0].set_xticks([cdtick, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0].set_xlim(0.0, 0.3)
    axes[0].set_ylim(-0.15, 0.8)

    axes[0].legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

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
