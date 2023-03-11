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
from helperFuncs import load_jld, readlines, get_bendingtwisting, postprocess_flutterevals, find_DivAndFlutterPoints
from helperPlotFuncs import plot_mode_shapes, plot_vg_vf_rl, plot_wing, plot_dlf, plot_forced

# ==============================================================================
#                         Main driver
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--cases", type=str, default=[], nargs="+", help="Full case folder names in order")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--is_static", action="store_true", default=False)
    parser.add_argument("--is_forced", action="store_true", default=False)
    parser.add_argument("--is_modal", action="store_true", default=False)
    parser.add_argument("--is_flutter", action="store_true", default=False)
    parser.add_argument("--debug_plots", help="flutter debug plots", action="store_true", default=False)
    parser.add_argument("--batch", help="Run pytecplot in batch", action="store_true", default=False)
    args = parser.parse_args()
    # Echo the args
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    print(30 * "-")

    # ************************************************
    #     I/O
    # ************************************************
    # Input data read directory
    dataDir = "../OUTPUT/"
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

    for ii, caseDir in enumerate(caseDirs):
        key = args.cases[ii]
        # --- Read in DVDict ---
        DVDictDict[key] = json.load(open(f"{caseDir}/init_DVDict.json"))
        # --- Read in funcs ---
        try:
            funcsDict[key] = json.load(open(f"{caseDir}/funcs.json"))
        except FileNotFoundError:
            funcs = None
            print("No funcs.json file found...")

        nodes = np.linspace(0, DVDictDict[key]["s"], DVDictDict[key]["nNodes"], endpoint=True)

    # ************************************************
    #     Plot settings
    # ************************************************
    plt.style.use(niceplots.get_style("doumont-light"))  # all settings
    # --- Adjust default options for matplotlib ---
    myOptions = {
        "font.size": 25,
        "font.family": "sans-serif",
        # "font.sans-serif": ["Helvetica"],
        # "text.usetex": True,
        "text.latex.preamble": [
            r"\usepackage{lmodern}",
            r"\usepackage{amsmath}",
            r"\usepackage{helvet}",
            r"\usepackage{sansmath}",
            r"\sansmath",
        ],
    }
    plt.rcParams.update(myOptions)

    # Linestyles
    ls = ["-", "--", "-.", ":"]
    # ==============================================================================
    #                         READ IN DATA
    # ==============================================================================
    # ************************************************
    #     Static hydroelastic
    # ************************************************
    if args.is_static:
        # ************************************************
        #     Read in data
        # ************************************************
        print("Reading in static hydroelastic data")
        bendingDict = {}
        twistingDict = {}
        liftDict = {}
        momentDict = {}

        for ii, caseDir in enumerate(caseDirs):
            key = args.cases[ii]
            # --- Read bending ---
            fname = f"{caseDir}/bending.dat"
            bendingDict[key] = np.loadtxt(fname)

            # --- Read twisting ---
            fname = f"{caseDir}/twisting.dat"
            twistingDict[key] = readlines(fname)

            # --- Read lift ---
            fname = f"{caseDir}/lift.dat"
            liftDict[key] = np.loadtxt(fname)

            # --- Read moment ---
            fname = f"{caseDir}/moment.dat"
            momentDict[key] = np.loadtxt(fname)

    # ************************************************
    #     Dynamic hydroelastic
    # ************************************************
    if args.is_forced:
        print("Reading in forced vibration data")
        # --- Read frequencies ---
        fExtSweep = np.loadtxt(f"{dataDir}/forced/freqSweep.dat")

        # --- Read tip bending ---
        dynTipBending = np.asarray(load_jld(f"{dataDir}/forced/tipBendDyn.jld")["data"]).T

        # --- Read tip twisting ---
        dynTipTwisting = np.rad2deg(np.asarray(load_jld(f"{dataDir}/forced/tipTwistDyn.jld")["data"]).T)

        # --- Read tip lift ---
        dynLift = np.asarray(load_jld(f"{dataDir}/forced/totalLiftDyn.jld")["data"]).T

        # --- Read tip moment ---
        dynMoment = np.asarray(load_jld(f"{dataDir}/forced/totalMomentDyn.jld")["data"]).T

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

            structJLData = load_jld(f"{caseDir}/modal/structModal.jld")
            wetJLData = load_jld(f"{caseDir}/modal/wetModal.jld")

            # NOTE: Julia stores data in column major order so it is transposed
            structNatFreqsDict[key] = np.asarray(structJLData["structNatFreqs"])
            structModes = np.asarray(structJLData["structModeShapes"])
            wetNatFreqsDict[key] = np.asarray(wetJLData["wetNatFreqs"])
            wetModes = np.asarray(wetJLData["wetModeShapes"])

            # Turn into the right states
            nModes = structModes.shape[0]
            nDOF = 4  # TODO: should be an option somehow
            structBendModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
            structTwistModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
            wetBendModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
            wetTwistModesDict[key] = np.zeros((nModes, DVDict["nNodes"]))
            for ii in range(nModes):
                structBendModesDict[key][ii, :], structTwistModesDict[key][ii, :] = get_bendingtwisting(
                    structModes[ii, :], nDOF=nDOF
                )
                wetBendModesDict[key][ii, :], wetTwistModesDict[key][ii, :] = get_bendingtwisting(
                    wetModes[ii, :], nDOF=nDOF
                )

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
            eigs_i = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigs_i.jld")["data"]).T / (
                2 * np.pi
            )  # put in units of Hz
            eigs_r = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigs_r.jld")["data"]).T / (
                2 * np.pi
            )  # put in units of Hz
            evecs_i = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigenvectors_i.jld")["data"]).T
            evecs_r = np.asarray(load_jld(f"{caseDir}/pkFlutter/eigenvectors_r.jld")["data"]).T
            flowHistory = np.asarray(load_jld(f"{caseDir}/pkFlutter/flowHistory.jld")["data"]).T
            iblank = np.asarray(load_jld(f"{caseDir}/pkFlutter/iblank.jld")["data"]).T

            flutterSolDict[key] = postprocess_flutterevals(
                iblank, flowHistory[:, 1], flowHistory[:, 0], flowHistory[:, 2], eigs_r, eigs_i
            )
            instabPtsDict[key] = find_DivAndFlutterPoints(flutterSolDict[key], "pvals_r", "U")
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

                speed = lines[0].split(":")[1].rstrip("\n")

                vSweep.append(float(speed))
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
            #     axes[0, 1].plot(flowList, gSweep[mode, :], label=f"Mode {mode+1}", c=cm[iic])
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

    # Default visualize wing geometry
    for key in args.cases:
        fname = f"{outputDir}/wing-geom-{key}.pdf"
        DVDict = DVDictDict[key]
        fig, axes = plot_wing(DVDict)

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

    if args.is_static:
        fname = f"{outputDir}/static_spanwise.pdf"
        dosave = not not fname

        cl = funcs["cl"]
        lift = funcs["lift"]
        mom = funcs["moment"]
        cmy = funcs["cmy"]
        liftTitle = f"Lift ({lift:0.1f}N, CL={cl:.2f})"
        momTitle = f"Mom. ({mom:.1f}N-m, CM={cmy:.2f})"

        # Create figure object
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, constrained_layout=False, figsize=(14, 10))

        ax = axes[0]
        ax.plot(nodes, bending, label="Bending")
        ax.set_ylabel("$w$ [m]", rotation=0, labelpad=20)

        ax = axes[1]
        ax.plot(nodes, twisting, label="Twisting")
        ax.set_ylabel("$\psi$ [$^{\\circ}$]", rotation=0, labelpad=20)

        ax = axes[2]
        ax.plot(nodes, lift, label="Lift")
        ax.set_ylabel("$L$ [N]", rotation=0, labelpad=20)
        ax.set_xlabel("$y$ [m]")
        ax.set_title(liftTitle)

        ax = axes[3]
        ax.plot(nodes, moment, label="Moment")
        ax.set_ylabel("$M$ [N-m/m]", rotation=0, labelpad=20)
        ax.set_xlabel("$y$ [m]")
        ax.set_title(momTitle)

        fiberAngle = DVDict["θ"] * 180 / np.pi
        flowSpeed = DVDict["U∞"]
        AOA = DVDict["α₀"]
        sweepAngle = DVDict["Λ"] * 180 / np.pi
        flowString = "$U_{\\infty} =$" + f"{flowSpeed:.1f}" + "m/s"
        alfaString = "$\\alpha_0$ = " + f"{AOA:.1f}" + "$^{\\circ}$"
        sweepString = "$\\Lambda$ = " + f"{sweepAngle:.1f}" + "$^{\\circ}$"
        fiberString = "$\\theta$ = " + f"{fiberAngle:.1f}" + "$^{\\circ}$"
        titleTxt = flowString + "," + alfaString + sweepString + fiberAngle

        fig.suptitle(titleTxt)

        plt.show(block=(not dosave))
        # niceplots.all()
        for ax in axes:
            niceplots.adjust_spines(ax, outward=True)
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

    if args.is_forced:
        # --- File name ---
        fname = f"{outputDir}/forced_dynamics.pdf"

        # --- Plot ---
        fig, axes = plot_forced(
            fExtSweep,
            dynTipBending,
            dynTipTwisting,
            dynLift,
            dynMoment,
            fname=fname,
        )

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
            fig, axes = plot_mode_shapes(
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

    if args.is_flutter:
        # ---------------------------
        #   Standard v-g, v-f, R-L plots
        # ---------------------------
        # --- File name ---
        fname = f"{outputDir}/vg_vf_rl.pdf"

        # Create figure object
        fact = 1  # scale size
        figsize = (18 * fact, 13 * fact)
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", sharey="row", constrained_layout=True, figsize=figsize)
        for ii, key in enumerate(args.cases):
            if ii == 0:
                annotateModes = True
            else:
                annotateModes = False
            # --- Plot ---
            fig, axes = plot_vg_vf_rl(
                fig,
                axes,
                flutterSol=flutterSolDict[key],
                ls=ls[ii],
                # units="kts",
                # marker="o",
                showRLlabels=True,
                annotateModes=annotateModes,
                nShift=50,
            )

            # # --- Set limits ---
            # axes[0,0].set_ylim(top=20)
            # axes[0,0].set_xlim(right=40, left=25)
            axes[0, 0].set_ylim(top=15,bottom=-10)
            axes[0, 0].set_xlim(right=190, left=170)
            # axes[0, 0].set_ylim(top=1, bottom=-5)
            # axes[0, 0].set_xlim(right=50, left=5)
            # axes[1, 1].set_xlim(right=1, left=-5)
            # axes[1, 1].set_ylim(top=20, bottom=0)

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()

        # ---------------------------
        #   Damping loss plots
        # ---------------------------
        # --- File name ---
        fname = f"{outputDir}/dlf.pdf"

        # Create figure object
        fact = 1  # scale size
        figsize = (18 * fact, 6 * fact)
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex="col", constrained_layout=True, figsize=figsize)
        for ii, key in enumerate(args.cases):
            # --- Plot ---
            fig, axes = plot_dlf(
                fig,
                axes,
                flutterSol=flutterSolDict[key],
                semichord=0.5 * np.mean(DVDictDict[key]["c"]),
                sweepAng=DVDictDict[key]["Λ"],
                ls="-",
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
