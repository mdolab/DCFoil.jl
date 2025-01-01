# --- Python 3.9 ---
"""
@File    :   postprocessing.py
@Time    :   2023/02/02
@Author  :   Galen Ng
@Desc    :   Make plots using python.
Use this file as a starting template for your own postprocessing scripts.

"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import json
import argparse
from pathlib import Path
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt


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
parser.add_argument("--is_static", action="store_true", default=False)
parser.add_argument("--drag_buildup", action="store_true", default=False)
parser.add_argument("--is_forced", action="store_true", default=False)
parser.add_argument("--is_modal", action="store_true", default=False)
parser.add_argument("--is_flutter", action="store_true", default=False)
parser.add_argument("--secondset", action="store_true", help="Custom second data set", default=False)
parser.add_argument(
    "--make_eigenvectors",
    help="Do you want to make the hydroelastic mode shape plots and movie?",
    action="store_true",
    default=False,
)
parser.add_argument("--debug_plots", help="flutter debug plots", action="store_true", default=False)
parser.add_argument("--batch", help="Run pytecplot in batch", action="store_true", default=False)
parser.add_argument("--elem", type=int, default=1, help="Type of beam element: 0=BT2, 1=COMP2")
parser.add_argument("--is_paper", action="store_true", default=False)
args = parser.parse_args()

# ==============================================================================
#                         COMMON PLOT SETTINGS
# ==============================================================================
dataDir = "../OUTPUT/"
labels = ["NOFS", "FS"]
labels = ["$-15^{\\circ}$", "$0^{\\circ}$", "$15^{\\circ}$"]
cm, fs_lgd, fs, ls, markers = set_my_plot_settings(args.is_paper)
alphas = [1.0, 0.5]

# ==============================================================================
#                         Hydrofoil params
# ==============================================================================
meanChord = 1.0
sweepAng = 0.0  # rad

print("Mean chord:\t", meanChord)
print("Sweep angle:\t", np.rad2deg(sweepAng))
# ==============================================================================
#                         Main driver
# ==============================================================================
if __name__ == "__main__":
    # Echo the args
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print(30 * "-")

    if args.elem == 0:
        N_DOF = 4
    elif args.elem == 1:
        N_DOF = 9
    # ************************************************
    #     I/O
    # ************************************************
    # Input data read directory
    caseDirs = []
    if args.cases is not None:
        for case in args.cases:
            caseDirs.append(dataDir + case)
    else:
        raise ValueError("Please specify a case to run postprocessing on")

    # Output plot directory
    outputDir = f"./PLOTS/{args.cases[0]}/"
    if args.output is not None:
        outputDir += args.output

    # Create output directory if it doesn't exist
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    # ************************************************
    #     Read in results
    # ************************************************
    DVDictDict = {}
    funcsDict = {}
    SolverOptions = {}

    for ii, caseDir in enumerate(caseDirs):
        key = args.cases[ii]

        # --- Read in DVDict ---
        try:
            DVDictDict[key] = json.load(open(f"{caseDir}/init_DVDict.json"))
        except FileNotFoundError:
            # There's a chance it's the multicomp case
            DVDictDict[key] = json.load(open(f"{caseDir}/init_DVDict-comp001.json"))
        SolverOptions[key] = json.load(open(f"{caseDir}/solverOptions.json"))

        # --- Read in funcs ---
        try:
            funcsDict[key] = json.load(open(f"{caseDir}/funcs.json"))
        except FileNotFoundError:
            funcs = None
            print("No funcs.json file found...")

        # try:
        #     nodes = np.linspace(0, DVDictDict[key]["s"], DVDictDict[key]["nNodes"], endpoint=True)
        # except KeyError:
        #     try:
        #         nodes = np.linspace(0, DVDictDict[key]["s"], DVDictDict[key]["neval"], endpoint=True)
        #     except KeyError:
        #         try:
        #             nodes = np.linspace(0, DVDictDict[key]["s"], SolverOptions[key]["nNodes"], endpoint=True)
        #         except KeyError:
        #             nodes = np.linspace(
        #                 0, DVDictDict[key]["s"], SolverOptions[key]["appendageList"][0]["nNodes"], endpoint=True
        #             )

    if args.secondset:
        for ii, caseDir in enumerate(caseDirs):
            key = args.cases[ii]
            newkey = key.replace("t-foil", "t-foil-fs")
            fscaseDir = caseDir.replace("t-foil", "t-foil-fs")

            # --- Read in DVDict ---
            DVDictDict[newkey] = json.load(open(f"{fscaseDir}/init_DVDict.json"))
            SolverOptions[newkey] = json.load(open(f"{fscaseDir}/solverOptions.json"))

            # --- Read in funcs ---
            funcsDict[newkey] = json.load(open(f"{fscaseDir}/funcs.json"))

    # ==============================================================================
    #                         READ IN DATA
    # ==============================================================================
    # ************************************************
    #     Static hydroelastic
    # ************************************************
    if args.is_static or args.drag_buildup:
        # ************************************************
        #     Read in data
        # ************************************************
        print("Reading in static hydroelastic data...")
        bendingDict = {}
        twistingDict = {}
        liftDict = {}
        momentDict = {}

        for ii, caseDir in enumerate(caseDirs):
            key = args.cases[ii]
            # --- Read bending ---
            fname = f"{caseDir}/static/bending.dat"
            bendingDict[key] = np.loadtxt(fname)

            # --- Read twisting ---
            fname = f"{caseDir}/static/twisting.dat"
            twistingDict[key] = np.loadtxt(fname)

            # --- Read lift ---
            fname = f"{caseDir}/static/lift.dat"
            liftDict[key] = np.loadtxt(fname)

            # --- Read moment ---
            fname = f"{caseDir}/static/moments.dat"
            momentDict[key] = np.loadtxt(fname)

        if args.secondset:
            for ii, caseDir in enumerate(caseDirs):
                key = args.cases[ii].replace("t-foil", "t-foil-fs")
                caseDir = caseDir.replace("t-foil", "t-foil-fs")
                # --- Read bending ---
                fname = f"{caseDir}/static/bending.dat"
                bendingDict[key] = np.loadtxt(fname)

                # --- Read twisting ---
                fname = f"{caseDir}/static/twisting.dat"
                twistingDict[key] = np.loadtxt(fname)

                # --- Read lift ---
                fname = f"{caseDir}/static/lift.dat"
                liftDict[key] = np.loadtxt(fname)

                # --- Read moment ---
                fname = f"{caseDir}/static/moments.dat"
                momentDict[key] = np.loadtxt(fname)

    # ************************************************
    #     Dynamic hydroelastic
    # ************************************************
    if args.is_forced:
        print("Reading in forced vibration data")
        for ii, caseDir in enumerate(caseDirs):
            key = args.cases[ii]

            # --- Read frequencies ---
            fExtSweep = np.loadtxt(f"{caseDir}/forced/freqSweep.dat")

            # --- Read tip bending ---
            dynTipBending = np.asarray(load_jld(f"{caseDir}/forced/tipBendDyn.jld2")["data"]).T

            # --- Read tip twisting ---
            dynTipTwisting = np.rad2deg(np.asarray(load_jld(f"{caseDir}/forced/tipTwistDyn.jld2")["data"]).T)

            # --- Read lift ---
            dynLiftRAO = np.asarray(load_jld(f"{caseDir}/forced/totalLiftRAO.jld2")["data"]).T

            # --- Read moment ---
            dynMomentRAO = np.asarray(load_jld(f"{caseDir}/forced/totalMomentRAO.jld2")["data"]).T

            # --- Read wave amp ---
            waveAmpSpectrum = np.asarray(load_jld(f"{caseDir}/forced/waveAmpSpectrum.jld2")["data"]).T

            # --- Read RAO (general xfer fcn for deflections) ---
            rao = np.asarray(load_jld(f"{caseDir}/forced/RAO.jld2")["data"]).T
            deflectionRAO = np.asarray(load_jld(f"{caseDir}/forced/deflectionRAO.jld2")["data"]).T

    # ************************************************
    #     Flutter and modal solutions
    # ************************************************
    if args.is_modal:
        # ************************************************
        #     Read in data
        # ************************************************
        print("Reading in modal data...")

        structNatFreqsDict = {}
        structBendModesDict = {}
        structTwistModesDict = {}
        wetNatFreqsDict = {}
        wetBendModesDict = {}
        wetTwistModesDict = {}

        for ii, caseDir in enumerate(caseDirs):
            key = args.cases[ii]
            DVDict = DVDictDict[key]

            structJLData = load_jld(f"{caseDir}/modal/structModal.jld2")
            wetJLData = load_jld(f"{caseDir}/modal/wetModal.jld2")

            # NOTE: Julia stores data in column major order so it is transposed
            structNatFreqsDict[key] = np.asarray(structJLData["structNatFreqs"])
            structModes = np.asarray(structJLData["structModeShapes"])
            wetNatFreqsDict[key] = np.asarray(wetJLData["wetNatFreqs"])
            wetModes = np.asarray(wetJLData["wetModeShapes"])

            # Turn into the right states
            nModes = structModes.shape[0]
            try:
                structBendModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
                structTwistModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
                wetBendModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
                wetTwistModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
            except KeyError:
                try:
                    structBendModesDict[key] = np.zeros((nModes, DVDict["neval"]))
                    structTwistModesDict[key] = np.zeros((nModes, DVDict["neval"]))
                    wetBendModesDict[key] = np.zeros((nModes, DVDict["neval"]))
                    wetTwistModesDict[key] = np.zeros((nModes, DVDict["neval"]))
                except KeyError:
                    structBendModesDict[key] = np.zeros((nModes, SolverOptions[key]["nNodes"]))
                    structTwistModesDict[key] = np.zeros((nModes, SolverOptions[key]["nNodes"]))
                    wetBendModesDict[key] = np.zeros((nModes, SolverOptions[key]["nNodes"]))
                    wetTwistModesDict[key] = np.zeros((nModes, SolverOptions[key]["nNodes"]))
            for ii in range(nModes):
                # Dry
                bend, twist = get_bendingtwisting(structModes[ii, :], nDOF=N_DOF)
                structBendModesDict[key][ii, :] = bend
                structTwistModesDict[key][ii, :] = twist
                # Wet
                bend, twist = get_bendingtwisting(wetModes[ii, :], nDOF=N_DOF)
                wetBendModesDict[key][ii, :] = bend
                wetTwistModesDict[key][ii, :] = twist

    if args.is_flutter:
        # ************************************************
        #     Read in data
        # ************************************************
        print("Reading in flutter data...")

        flutterSolDict = {}
        instabPtsDict = {}
        for ii, caseDir in enumerate(caseDirs):
            key = args.cases[ii]
            # Remember to transpose data since Julia stores in column major order
            eigs_i = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigs_i.jld2")["data"]).T / (
                2 * np.pi
            )  # put in units of Hz
            eigs_r = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigs_r.jld2")["data"]).T / (
                2 * np.pi
            )  # put in units of Hz
            evecs_i = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigenvectors_i.jld2")["data"]).T
            evecs_r = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigenvectors_r.jld2")["data"]).T
            flowHistory = np.asarray(load_jld(f"{caseDir}/pkFlutter/flowHistory.jld2")["data"]).T
            iblank = np.asarray(load_jld(f"{caseDir}/pkFlutter/iblank.jld2")["data"]).T

            # --- Post process the solution ---
            flutterSolDict[key] = postprocess_flutterevals(
                iblank,
                flowHistory[:, 1],
                flowHistory[:, 0],
                flowHistory[:, 2],
                eigs_r,
                eigs_i,
                R_r=evecs_r,
                R_i=evecs_i,
            )
            # You only need to know the stability point on one processor really
            instabPtsDict[key] = find_DivAndFlutterPoints(flutterSolDict[key], "pvals_r", "U", altKey="pvals_i")

        # ************************************************
        #     Debug code
        # ************************************************
        if args.debug_plots:
            fname = f"{outputDir}/vg_vf_rl_plot.pdf"
            debugDir = "../DebugOutput/"
            testlines = readlines(f"{debugDir}/eigenvalues-001.dat")
            nModes = len(testlines) - 2
            nFlows = 157
            vSweep = []
            flowList = []
            fSweep = np.zeros((nModes, nFlows))
            gSweep = np.zeros((nModes, nFlows))
            iblankSweep = np.zeros((nModes, nFlows))

            flowIter = 1
            for ii in range(nFlows):
                lines = readlines(f"{debugDir}/eigenvalues-%03i.dat" % (flowIter))
                iblankLines = readlines(f"{debugDir}/iblank-%03i.dat" % (flowIter))

                nModes = len(lines) - 2

                critSpeed = lines[0].split(":")[1].rstrip("\n")

                vSweep.append(float(critSpeed))
                for jj in range(nModes):
                    line = lines[jj + 2]
                    iblankLine = iblankLines[jj + 2]
                    line = line.split()
                    g = float(line[0])
                    f = float(line[1])
                    gSweep[jj, ii] = g
                    fSweep[jj, ii] = f
                    iblankSweep[jj, ii] = iblankLine

                flowList.append(flowIter)
                flowIter += 1

            flutterSol = postprocess_flutterevals(iblankSweep, None, np.array(vSweep), gSweep, fSweep)
            fig, axes = plot_vg_vf_rl(
                flutterSol,
                ls=ls,
                # units="kts",
                # marker="o",
                # showRLlabels=True,
            )

            # axes[0, 0].set_xlim(8.4, 8.7)
            # axes[0, 0].set_ylim(-15, 2.5)
            # axes[1, 0].set_ylim(-10, 250)
            # axes[1, 1].set_xlim(-50, 10)

            # # --- Debug nflow jumping ---
            # niceColors = sns.color_palette("tab10")
            # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
            # cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            # for mode in range(nModes):
            #     iic = mode % len(cm)
            #     axes[0, 1].plot(flowList, gSweep[mode, :], label=f"Mode {mode+1}", color=cm[iic])
            # axes[0, 1].set_xlabel("nFlow")
            # axes[0, 1].legend(fontsize=10, labelcolor="linecolor", loc="best", frameon=False)
            # axes[0, 1].set_xlim(50, 60)
            # axes[0, 1].set_xlim(0, nFlows)

            dosave = not not fname
            plt.show(block=(not dosave))
            if dosave:
                plt.savefig(fname, format="pdf")
                print("Saved to:", fname)
            plt.close()

    # ==============================================================================
    #                         Plot results
    # ==============================================================================

    # ************************************************
    #     Visualize geometry
    # ************************************************
    for key in args.cases:
        fname = f"{outputDir}/wing-geom-{key}.pdf"
        DVDict = DVDictDict[key]
        try:
            fig, axes = plot_wingPlanform(DVDict, nNodes=SolverOptions[key]["nNodes"], cm=cm)
            dosave = not not fname
            plt.show(block=(not dosave))
            if dosave:
                plt.savefig(fname, format="pdf")
                print("Saved to:", fname)
            plt.close()
        except KeyError:
            print("Skipping wing planform plot")

    # if args.is_static:
    # # ---------------------------
    # #   2D plots
    # # ---------------------------
    # fname = f"{outputDir}/static-spanwise.pdf"
    # dosave = not not fname

    # # Create figure object
    # fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, constrained_layout=True, figsize=(18, 8))
    # lpad = 40
    # ytickBend = []
    # ytickTwist = []

    # for iic, key in enumerate(args.cases):
    #     funcs = funcsDict[key]
    #     bending = bendingDict[key]
    #     twisting = np.rad2deg(twistingDict[key])
    #     spanLift = liftDict[key]
    #     spanMoment = momentDict[key]

    #     # If you used a half strip on the root, we're doubling the lift and moment there
    #     spanLift[0] *= 2
    #     spanMoment[0] *= 2

    #     if iic < 1:
    #         ytickBend.append(np.max(np.abs(bending)))
    #         ytickTwist.append(np.max(np.abs(twisting)))

    #     fig, axes = plot_static2d(
    #         fig,
    #         axes,
    #         nodes,
    #         bending,
    #         twisting,
    #         spanLift,
    #         spanMoment,
    #         funcs,
    #         labels[iic],
    #         cm[iic],
    #         fs_lgd,
    #         iic,
    #         solverOptions=SolverOptions[key],
    #     )

    # fiberAngle = np.rad2deg(DVDictDict[key]["theta_f"])
    # flowSpeed = SolverOptions[key]["Uinf"]
    # AOA = DVDictDict[key]["alfa0"]
    # sweepAngle = np.rad2deg(DVDictDict[key]["sweep"])
    # flowString = "$U_{\\infty} = " + f"{flowSpeed*1.9438:.1f}$" + "\,kts"
    # alfaString = "$\\alpha_0$ = " + f"{AOA:.1f}" + "$^{\\circ}$"
    # sweepString = "$\\Lambda$ = " + f"{sweepAngle:.1f}" + "$^{\\circ}$"
    # fiberString = "$\\theta_f$ = " + f"{fiberAngle:.1f}" + "$^{\\circ}$"
    # titleTxt = flowString + ", " + alfaString + ", " + sweepString + ", " + fiberString

    # fig.suptitle(titleTxt)

    # axes[0, 0].legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

    # # --- Put second set after legend call so it's not labeled ---
    # if args.secondset:
    #     for iic, key in enumerate(args.cases):
    #         newkey = key.replace("t-foil", "t-foil-fs")
    #         funcs = funcsDict[newkey]
    #         bending = bendingDict[newkey]
    #         twisting = np.rad2deg(twistingDict[newkey])
    #         spanLift = liftDict[newkey]
    #         spanMoment = momentDict[newkey]

    #         # If you used a half strip on the root, we're doubling the lift and moment there
    #         spanLift[0] *= 2
    #         spanMoment[0] *= 2

    #         fig, axes = plot_static2d(
    #             fig,
    #             axes,
    #             nodes,
    #             bending,
    #             twisting,
    #             spanLift,
    #             spanMoment,
    #             funcs,
    #             labels[iic],
    #             cm[iic],
    #             fs_lgd,
    #             iic,
    #             solverOptions=SolverOptions[key],
    #             ls="--",
    #         )

    # yticks = np.array(ytickBend)
    # # ytickBend = np.concatenate((yticks, -yticks)).tolist()
    # ytickBend = yticks.tolist()
    # axes[0, 0].set_yticks(ytickBend + [0.0])
    # axes[0, 0].set_ylim(-0.2 * np.max(ytickBend), 1.1 * np.max(ytickBend))

    # yticks = np.array(ytickTwist)
    # # ytickTwist = np.concatenate((yticks, -yticks)).tolist()
    # ytickTwist = yticks.tolist()
    # axes[0, 1].set_yticks(ytickTwist + [0.0])
    # axes[0, 1].set_ylim(-0.1 * np.max(ytickTwist), 1.1 * np.max(ytickTwist))

    # xticks = [0.0, DVDictDict[key]["s"]]
    # axes[0, 0].set_xticks(xticks)

    # plt.show(block=(not dosave))
    # for ii, ax in enumerate(axes.flatten()):
    #     # niceplots.adjust_spines(ax, ["left", "bottom"], outward=True)
    #     # if ii < 2:
    #     ax.yaxis.set_label_position("right")
    #     niceplots.adjust_spines(ax, ["right", "bottom"], outward=True)
    # if dosave:
    #     plt.savefig(fname, format="pdf")
    #     print("Saved to:", fname)
    # plt.close()

    # # ---------------------------
    # #   3D visualization
    # # ---------------------------
    # fname = f"{outputDir}/static-3Dshape.pdf"
    # dosave = not not fname

    # # Create figure object
    # fig = plt.figure(constrained_layout=True, figsize=(14, 8))

    # for iic, key in enumerate(args.cases):
    #     DVDict = DVDictDict[key]
    #     funcs = funcsDict[key]
    #     bending = bendingDict[key]
    #     twisting = np.rad2deg(twistingDict[key])

    #     # TODO: figure out how to get the proper ax for plot_surface()
    #     ax = plot_deformedShape(fig, ax, DVDict, bending, twisting)

    # plt.show(block=(not dosave))
    # niceplots.adjust_spines(ax, outward=True)
    # if dosave:
    #     plt.savefig(fname, format="pdf")
    #     print("Saved to:", fname)
    # plt.close()

    if args.drag_buildup:
        fname = f"{outputDir}/drag-buildup.pdf"
        dosave = not not fname

        fig, axes = plt.subplots(nrows=2, ncols=3, sharey=True, constrained_layout=True, figsize=(15, 10))

        for iic, key in enumerate(args.cases):
            funcs = funcsDict[key]

            fig, axes = plot_dragbuildup(
                fig,
                axes,
                funcs,
                labels[iic],
                cm,
                fs_lgd,
                iic,
                solverOptions=SolverOptions[key],
            )

        if args.secondset:
            for iic, key in enumerate(args.cases):
                newkey = key.replace("t-foil", "t-foil-fs")
                funcs = funcsDict[newkey]

                fig, axes = plot_dragbuildup(
                    fig,
                    axes,
                    funcs,
                    labels[iic],
                    cm,
                    fs_lgd,
                    iic + 3,
                    solverOptions=SolverOptions[key],
                )

        fiberAngle = np.rad2deg(DVDictDict[key]["theta_f"])
        flowSpeed = SolverOptions[key]["Uinf"]
        AOA = DVDictDict[key]["alfa0"]
        sweepAngle = np.rad2deg(DVDictDict[key]["sweep"])
        flowString = "$U_{\\infty} = " + f"{flowSpeed*1.9438:.1f}$" + "\,kts"
        alfaString = "$\\alpha_0$ = " + f"{AOA:.1f}" + "$^{\\circ}$"
        sweepString = "$\\Lambda$ = " + f"{sweepAngle:.1f}" + "$^{\\circ}$"
        fiberString = "$\\theta_f$ = " + f"{fiberAngle:.1f}" + "$^{\\circ}$"
        titleTxt = flowString + ", " + alfaString + ", " + sweepString + ", " + fiberString

        fig.suptitle(titleTxt)

        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

    if args.is_forced:
        # --- File name ---
        fname = f"{outputDir}/forced-dynamics.pdf"

        # Create figure object
        nrows = 2
        ncols = 3
        figsize = (6 * ncols, 6 * nrows)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            constrained_layout=True,
            figsize=figsize,
        )

        fiberAngle = np.rad2deg(DVDictDict[key]["theta_f"])
        flowSpeed = SolverOptions[key]["Uinf"]
        # --- Plot ---
        fig, axes = plot_forced(
            fig,
            axes,
            fExtSweep,
            waveAmpSpectrum,
            deflectionRAO,
            dynLiftRAO,
            dynMomentRAO,
            rao,
            flowSpeed,
            fs_lgd,
            args.elem,
            cm,
        )

        fig.suptitle("Frequency response spectra")

        axes[0, 0].set_xlim(left=0.0, right=200)
        for ax in axes.flatten():
            ax.set_ylim(bottom=0.0)

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

    if args.is_modal:
        # --- File name ---
        fname = f"{outputDir}/modal-struct.pdf"

        # Create figure object
        nrows = 2
        ncols = 2
        figsize = (9 * ncols, 4 * nrows)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            constrained_layout=True,
            figsize=figsize,
        )
        for ii, key in enumerate(args.cases):
            # --- Pack up data to plot ---
            modeShapeData = {
                "structBM": structBendModesDict[key],
                "structTM": structTwistModesDict[key],
                "wetBM": wetBendModesDict[key],
                "wetTM": wetTwistModesDict[key],
            }
            modeFreqData = {
                "structNatFreqs": structNatFreqsDict[key],
                "wetNatFreqs": wetNatFreqsDict[key],
            }

            # --- Plot ---
            fig, axes = plot_naturalModeShapes(
                fig,
                axes,
                y=nodes,
                nModes=nModes,
                modeShapes=modeShapeData,
                modeFreqs=modeFreqData,
                ls=ls[ii],
            )

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

    # NOTE: This is the only post processing code that is parallel
    if args.is_flutter:
        # ************************************************
        #       Standard v-g, v-f, R-L plots
        # ************************************************
        # --- File name ---
        fname = f"{outputDir}/vg-vf-rl.pdf"

        # --- Create figure object ---
        fact = 0.85  # scale size
        figsize = (18 * fact, 13 * fact)
        fig, axes = plt.subplots(
            nrows=2,
            ncols=2,
            sharex="col",
            sharey="row",
            constrained_layout=True,
            figsize=figsize,
        )

        instabSpeedTicks = []
        instabFreqTicks = []
        units = "kts"
        # units = "m/s"
        for ii, key in enumerate(args.cases):
            if ii == 0:
                annotateModes = True
            else:
                annotateModes = False

            # can force to see modes in legend label
            # annotateModes = False

            # --- Plot ---
            fig, axes = plot_vg_vf_rl(
                fig,
                axes,
                flutterSol=flutterSolDict[key],
                cm=cm,
                # ls=ls[ii],
                alpha=alphas[ii],
                units=units,
                # marker="o",
                showRLlabels=True,
                annotateModes=annotateModes,
                nShift=62,
                instabPts=instabPtsDict[key],
            )
            if units == "kts":
                unitFactor = 1.9438
            else:
                unitFactor = 1.0
            instabSpeedTicks.append(instabPtsDict[key][0][0] * unitFactor)
            instabFreqTicks.append(instabPtsDict[key][0][-1])

        # --- Set limits ---
        # # IMOCA60 paper
        # axes[0, 0].set_ylim(top=0.5, bottom=-4)
        # axes[1, 0].set_ylim(top=22.0)
        # axes[1, 0].set_yticks([0, 10, 15, 20] + instabFreqTicks)
        axes[0, 0].set_xticks([10, 60] + instabSpeedTicks)
        # # axes[0, 0].set_xticks([10, 60])

        # # akcabay limits
        # # swept flutter
        # axes[0, 0].set_ylim(top=30,bottom=-400)
        # axes[0, 0].set_xlim(left=160,right=175)
        # axes[0, 0].set_xticks([160, 175] + instabSpeedTicks)
        # axes[1,0].set_yticks([0, 200, 400, 600, 800] + instabFreqTicks)
        # # static div
        # axes[0, 0].set_ylim(top=20, bottom=-80)
        # axes[0, 0].set_xlim(right=35)
        # axes[0, 0].set_xticks([20, 35] + [instabPtsDict[key][0][0]])
        # axes[1, 0].set_ylim(top=510)

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

        # ************************************************
        #     Damping loss plots
        # ************************************************
        # --- File name ---
        fname = f"{outputDir}/dlf.pdf"

        # --- Create figure ---
        fact = 0.85  # scale size
        figsize = (18 * fact, 6 * fact)
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex="col", constrained_layout=True, figsize=figsize)
        for ii, key in enumerate(args.cases):
            # --- Plot ---
            fig, axes = plot_dlf(
                fig,
                axes,
                flutterSol=flutterSolDict[key],
                cm=cm,
                semichord=0.5 * meanChord,
                sweepAng=sweepAng,
                ls=ls[ii],
                units="kts",
                nShift=500,
            )
            axes[0].set_ylim(-0.1, 0.8)
            axes[0].set_xlim(right=50, left=5)
            # axes[1].set_ylim(-0.1, 1.2)
            # axes[1].set_xlim(left=10)

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

        # ************************************************
        #     Hydroelastic mode shapes
        # ************************************************
        # These plots can only work for one case
        if args.make_eigenvectors:
            try:
                from mpi4py import MPI  # for the flutter processing only
            except ImportError:
                print("mpi4py not installed, skipping mpi4py import")
            comm = MPI.COMM_WORLD
            # --- File dirname ---
            dirname = f"{outputDir}/hydroelastic-mode-shapes/"
            # Create output directory if it doesn't exist
            Path(dirname).mkdir(parents=True, exist_ok=True)

            vRange = [
                0.20,
                50.0,
            ]  # set the speed range to plot in whatever units you choose
            fact = 0.85  # scale size
            for ii, key in enumerate(args.cases):
                # --- Plot ---
                plot_modeShapes(
                    comm,
                    chordVec=DVDictDict[key]["c"],
                    vRange=vRange,
                    fact=fact,
                    y=nodes / nodes[-1],
                    flutterSol=flutterSolDict[key],
                    ls="-",
                    units="kts",
                    outputDir=dirname,
                )
            fps = 50
            if comm.rank == 0:
                os.system(f"mkdir -p ./MOVIES/")
            mm = comm.rank  # mode number
            os.system(f"ffmpeg -r {fps} -i ./{dirname}/mode{mm}_%04d.png movie.mp4")
            os.system(f"mv movie.mp4 ./MOVIES/{args.cases[0]}/mode{mm}.mp4")
