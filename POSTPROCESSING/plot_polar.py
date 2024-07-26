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
parser.add_argument(
    "--cases", type=str, default=[], nargs="+", help="Full case folder names in order"
)
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
    0.012840637345966462,
    0.013536750545987842,
    0.014978522214719226,
    0.018184044550279745,
    0.022636330484294104,
    0.028149298891296807,
    0.03153621715626051,
    0.03811359625609688,
    0.044875851198852776,
    0.04985482446016917,
    0.054828534352292124,
    0.06211825718628461,
    0.07153216272955931,
    0.07811202670514095,
    0.08663690134283315,
    0.09535431669545556,
    0.10476801743132018,
    0.11240761838255457,
    0.12093610343051993,
    0.12127039262268167,
    0.13937381621188963,
]
CLCDY = [
    -0.04517763085136606,
    -0.1450421009946805,
    0.009576720379859571,
    0.08237511917592077,
    0.1847111398956447,
    0.25534909238365566,
    0.3558746590972339,
    0.4214787716874877,
    0.5396983936639973,
    0.600052621725689,
    0.6227632333159132,
    0.7045836558943922,
    0.7431605851709374,
    0.8265365290912126,
    0.8548470175119031,
    0.9905921799393163,
    1.0277043266190033,
    1.072620005363368,
    1.126752148057875,
    0.9783554120505197,
    0.9531559663134216,
]

CLALPHAX = [
    -2.1292517173802965,
    -1.2102720739110193,
    -0.15893147914821526,
    0.8043920161424527,
    1.767161507807458,
    2.7744488458894825,
    3.803089942586615,
    4.766545595230735,
    5.772915383687375,
    6.867763463020445,
    7.7869157417525745,
    8.794703314297033,
    9.841204004702517,
    10.809721812613214,
    11.74998428224171,
    12.88952464883304,
    13.809343783272704,
    14.816928362122264,
    14.869301693987888,
]

CLALPHAY = [
    -0.15158929127392007,
    -0.04966395313292904,
    0.0058250113482549715,
    0.07962485774969252,
    0.18476801327677772,
    0.2519417588220121,
    0.3513413914128549,
    0.41766430338546434,
    0.5367493365365628,
    0.6114311228777771,
    0.703589448387739,
    0.742461928986369,
    0.8237060954963207,
    0.8516997379978533,
    0.9898640378087984,
    1.016707249134287,
    1.0711375461346493,
    1.1214945980159592,
    0.9698495458153604,
]

CDALPHAX = [
    -2.227407649226012,
    -1.2218494195836769,
    -0.17656395525748536,
    0.7683334189200934,
    1.7964460583481934,
    2.781035666782614,
    3.765285367223905,
    4.811191819895567,
    5.778562519536328,
    6.829024188039362,
    7.7484214626467445,
    8.75534605511295,
    9.8012958544673,
    10.768771196410368,
    11.797274898301852,
    12.347874692703114,
    12.804350261838279,
    13.789304682415823,
    14.820632724320637,
    14.818077992297459,
]

CDALPHAY = [
    0.01278952969036215,
    0.012169106736405644,
    0.014237188275296325,
    0.017566937206897365,
    0.02189972083735295,
    0.02766765120778683,
    0.0310045596113509,
    0.0375139497970417,
    0.045216990237921695,
    0.053034097608913006,
    0.06161425305854437,
    0.07076605702091598,
    0.07758546268119373,
    0.08603690486817803,
    0.0931665672367551,
    0.09982142996648415,
    0.1033966858933422,
    0.11177375271138684,
    0.13910309485528577,
    0.12083165142552289,
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
                caseDirs.append(
                    dataDir + args.cases[0] + f"/f{fiberang:.1f}_w0.0_alfa{alpha:.2f}"
                )
                caseFSDirs.append(
                    dataDir
                    + args.cases[0].replace("t-foil", "t-foil-fs")
                    + f"/f{fiberang:.1f}_w0.0_alfa{alpha:.2f}"
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
            AlfaList.append(DVDict["alfa0"])

            # --- Read in funcs ---
            try:
                funcs = json.load(open(f"{caseDir}/funcs.json"))
            except FileNotFoundError:
                funcs = None
                print("No funcs.json file found...")

            CLDict[fiberang].append(funcs[f"cl-{compName}"])
            CD = (
                funcs[f"cdi-{compName}"]
                + funcs[f"cds-{compName}"]
                + funcs[f"cdpr-{compName}"]
                + funcs[f"cdj-{compName}"]
            )
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
            AlfaList.append(DVDict["alfa0"])

            # --- Read in funcs ---
            try:
                funcs = json.load(open(f"{caseDir}/funcs.json"))
            except FileNotFoundError:
                funcs = None
                print("No funcs.json file found...")

            CLFSDict[fiberang].append(funcs[f"cl-{compName}"])
            CD = (
                funcs[f"cdi-{compName}"]
                + funcs[f"cds-{compName}"]
                + funcs[f"cdpr-{compName}"]
                + funcs[f"cdj-{compName}"]
            )
            CDFSDict[fiberang].append(CD)

    # Rigid results
    caseDirs = []
    caseFSDirs = []
    for alpha in alphas:
        caseDirs.append(dataDir + args.cases[1] + f"/f0.0_w0.0_alfa{alpha:.2f}")
        caseFSDirs.append(
            dataDir
            + args.cases[1].replace("t-foil", "t-foil-fs")
            + f"/f0.0_w0.0_alfa{alpha:.2f}"
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
        AlfaList.append(DVDict["alfa0"])

        # --- Read in funcs ---
        try:
            funcs = json.load(open(f"{caseDir}/funcs.json"))
        except FileNotFoundError:
            funcs = None
            print("No funcs.json file found...")

        CLDict["rigid"].append(funcs[f"cl-{compName}"])
        CD = (
            funcs[f"cdi-{compName}"]
            + funcs[f"cds-{compName}"]
            + funcs[f"cdpr-{compName}"]
            + funcs[f"cdj-{compName}"]
        )
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
        AlfaList.append(DVDict["alfa0"])

        # --- Read in funcs ---
        try:
            funcs = json.load(open(f"{caseDir}/funcs.json"))
        except FileNotFoundError:
            funcs = None
            print("No funcs.json file found...")

        CLFSDict["rigid"].append(funcs[f"cl-{compName}"])
        CD = (
            funcs[f"cdi-{compName}"]
            + funcs[f"cds-{compName}"]
            + funcs[f"cdpr-{compName}"]
            + funcs[f"cdj-{compName}"]
        )
        CDFSDict["rigid"].append(CD)

    # ************************************************
    #     Drag polar
    # ************************************************
    fname = f"{outputDir}/drag-polar.pdf"
    dosave = not not fname

    # Create figure object
    fig, axes = plt.subplots(
        nrows=1, ncols=3, constrained_layout=True, figsize=(20, 4.5)
    )

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

        ax = axes[2]
        ax.plot(AlfaList, CDDict[fiberang], c=cm[ii])
        ax.plot(AlfaList, CDFSDict[fiberang], c=cm[ii], ls="--")

    # --- Rigid ---
    axes[0].plot(
        CDDict["rigid"],
        CLDict["rigid"],
        label=f"Rigid",
        c=cm[4],
    )
    cdtick = np.min(CDDict["rigid"])
    axes[0].plot(
        CDFSDict["rigid"],
        CLFSDict["rigid"],
        c=cm[4],
        ls="--",
    )
    axes[1].plot(
        AlfaList,
        CLDict["rigid"],
        # label=f"$\\theta_f{"rigid"}^" + "{\\circ}$",
        c=cm[4],
    )
    axes[1].plot(
        AlfaList,
        CLFSDict["rigid"],
        # label=f"$\\theta_f{"rigid"}^" + "{\\circ}$",
        c=cm[4],
        ls="--",
        # alpha=0.3,
    )
    axes[2].plot(AlfaList, CDDict["rigid"], c=cm[4])

    axes[0].plot(CLCDX, CLCDY, "ko", label="Expt. (Ref. 1)")
    axes[0].set_xticks([cdtick, 0.1, 0.2, 0.3, 0.4, 0.5])
    axes[0].set_xlim(0.0, 0.2)
    axes[0].set_ylim(-0.15, 0.8)
    axes[1].plot(CLALPHAX, CLALPHAY, "ko", label="Exp.")
    axes[1].axvline(2.0, color="gray", alpha=0.5)
    axes[1].set_xticks([-5, 0, 2, 10, 15])
    axes[1].set_ylim(-0.15, 0.8)

    axes[2].plot(CDALPHAX, CDALPHAY, "ko", label="Exp.")
    axes[2].axvline(2.0, color="gray", alpha=0.5)
    axes[2].set_ylim(0.0, 0.2)
    axes[2].set_xticks([-5, 0, 2, 10, 15])
    axes[2].set_xlabel("$\\alpha_r$ [$^{\\circ}$]")
    axes[2].set_ylabel("$C_D$", rotation="horizontal", ha="right", va="center")

    axes[0].legend(
        fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1
    )

    axes[0].set_xlabel("$C_D$")
    axes[0].set_ylabel("$C_L$", rotation="horizontal", ha="right", va="center")
    axes[1].set_ylabel("$C_L$", rotation="horizontal", ha="right", va="center")
    axes[1].set_xlabel("$\\alpha_r$ [$^{\\circ}$]")

    plt.show(block=(not dosave))
    # niceplots.all()
    for ax in axes.flatten():
        niceplots.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()
