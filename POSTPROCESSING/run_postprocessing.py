# --- Python 3.9 ---
"""
@File    :   postprocessing.py
@Time    :   2023/02/02
@Author  :   Galen Ng
@Desc    :   Make plots using python
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys
import copy
import json
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots
from helperFuncs import load_jld, readlines, get_bendingtwisting
from divAndFlutterPlots import plot_mode_shapes, plot_vg_vf_rl

# ==============================================================================
#                         Main driver
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--is_static", action="store_true", default=False)
    parser.add_argument("--is_forced", action="store_true", default=False)
    parser.add_argument("--is_modal", action="store_true", default=False)
    parser.add_argument("--is_flutter", action="store_true", default=False)
    parser.add_argument("--debug_plots", action="store_true", default=False)
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
    if args.case is not None:
        dataDir += args.case
    # Output plot directory
    outputDir = "./PLOTS/"
    if args.output is not None:
        outputDir += args.output
    # Create output directory if it doesn't exist
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    # ************************************************
    #     Read in results
    # ************************************************
    # --- Read in DVDict ---
    DVDict = json.load(open(f"{dataDir}/init_DVDict.json"))
    # --- Read in funcs ---
    try:
        funcs = json.load(open(f"{dataDir}/funcs.json"))
    except:
        funcs = None
        print("No funcs.json file found.")

    nodes = np.linspace(0, DVDict["s"], DVDict["neval"], endpoint=True)

    # ************************************************
    #     Plot settings
    # ************************************************
    plt.style.use(niceplots.get_style("doumont-light"))  # all settings
    # --- Adjust default options for matplotlib ---
    myOptions = {
        "font.size": 20,
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
    #                         Static hydroelastic
    # ==============================================================================
    if args.is_static:
        # ************************************************
        #     Read in data
        # ************************************************
        print("Reading in static hydroelastic data")
        # --- Read bending ---
        fname = f"{dataDir}/bending.dat"
        bending = np.loadtxt(fname)

        # --- Read twisting ---
        fname = f"{dataDir}/twisting.dat"
        twisting = readlines(fname)

        # --- Read lift ---
        fname = f"{dataDir}/lift.dat"
        lift = np.loadtxt(fname)

        # --- Read moment ---
        fname = f"{dataDir}/moment.dat"
        moment = np.loadtxt(fname)

    # ==============================================================================
    #                         Dynamic hydroelastic
    # ==============================================================================
    if args.is_forced:
        print("Reading in forced vibration data")
        # --- Read frequencies ---
        fname = f"{dataDir}/FreqSweep.dat"
        freqs = np.loadtxt(fname)

        # --- Read tip bending ---
        fname = f"{dataDir}/TipBendDyn.dat"
        dynTipBending = np.loadtxt(fname)

        # --- Read tip twisting ---
        fname = f"{dataDir}/TipTwistDyn.dat"
        dynTipTwisting = np.loadtxt(fname) * 180 / np.pi  # CONVERT TO DEGREES

        # --- Read tip lift ---
        fname = f"{dataDir}/TipLiftDyn.dat"
        dynTipLift = np.loadtxt(fname)

        # --- Read tip moment ---
        fname = f"{dataDir}/TipMomentDyn.dat"
        dynTipMoment = np.loadtxt(fname)

    # ==============================================================================
    #                         Flutter and modal solutions
    # ==============================================================================
    if args.is_modal:
        # ************************************************
        #     Read in data
        # ************************************************
        print("Reading in modal data...")
        structJLData = load_jld(f"{dataDir}/modal/structModal.jld")
        wetJLData = load_jld(f"{dataDir}/modal/wetModal.jld")

        # NOTE: Julia stores data in column major order so it is transposed
        structNatFreqs = np.asarray(structJLData["structNatFreqs"])
        structModes = np.asarray(structJLData["structModeShapes"])
        wetNatFreqs = np.asarray(wetJLData["wetNatFreqs"])
        wetModes = np.asarray(wetJLData["wetModeShapes"])

        # Turn into the right states
        nModes = structModes.shape[0]
        nDOF = 4  # TODO: should be an option somehow
        structBendModes = np.zeros((nModes, DVDict["neval"]))
        structTwistModes = np.zeros((nModes, DVDict["neval"]))
        wetBendModes = np.zeros((nModes, DVDict["neval"]))
        wetTwistModes = np.zeros((nModes, DVDict["neval"]))
        for ii in range(nModes):
            structBendModes[ii, :], structTwistModes[ii, :] = get_bendingtwisting(structModes[ii, :], nDOF=nDOF)
            wetBendModes[ii, :], wetTwistModes[ii, :] = get_bendingtwisting(wetModes[ii, :], nDOF=nDOF)

    if args.is_flutter:
        # --- Read in data ---
        print("Reading in flutter data...")

        fname = f"{outputDir}/vg_vf_rl_plot.pdf"
        # breakpoint()
        # ************************************************
        #     Debug code
        # ************************************************
        if args.debug_plots:
            debugDir = "../DebugOutput/"
            testlines = readlines(f"{debugDir}/eigenvalues-001.dat")
            nModes = len(testlines) - 2
            nFlows = 100
            vSweep = []
            fSweep = np.zeros((nModes, nFlows))
            gSweep = np.zeros((nModes, nFlows))

            flowIter = 1
            for ii in range(nFlows):
                lines = readlines(f"{debugDir}/eigenvalues-%03i.dat" % (flowIter))

                nModes = len(lines) - 2

                speed = lines[0].split(":")[1].rstrip("\n")

                vSweep.append(float(speed))
                for jj in range(nModes):
                    line = lines[jj + 2]
                    line = line.split()
                    g = float(line[0])
                    f = float(line[1])
                    # if g == f and f < 1e-10:
                    #     g = np.nan
                    #     f = np.nan
                    #     print(speed)
                    #     print("mode:", jj + 1)
                    #     print("Setting to nan because g == f == 0")
                    # else:
                    gSweep[jj, ii] = g
                    fSweep[jj, ii] = f

                flowIter += 1

            fig, axes = plot_vg_vf_rl(
                vSweep,
                fSweep,
                gSweep,
                nModes=4,
                ls=ls,
                # units="kts",
                # marker="o",
                # showRLlabels=True,
                modeList=[
                    0,
                    1,
                    2,
                    3,
                    4,
                    # weird modes
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                ],
            )

            # axes[0, 0].set_xlim(7.9, 14.0)
            axes[1, 0].set_ylim(-20, 350)
            axes[1, 1].set_xlim(-50, 10)

            dosave = not not fname
            plt.show(block=(not dosave))
            if dosave:
                plt.savefig(fname, format="pdf")
                print("Saved to:", fname)
            plt.close()

    # ==============================================================================
    #                         Plot results
    # ==============================================================================
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

    # TODO:
    # if is_dynamic:
    #     visuals = plot(
    #         [freqs, freqs],
    #         [dynTipBending, dynTipTwisting],
    #         label=["" ""],
    #         layout=(2, 2),
    #         xlabel="Frequency [Hz]",
    #         ylabel=[L"w_{\textrm{tip}}" * " [m]" L"\psi_{\textrm{tip}}" * " " * L"[^{\circ}]" "test" "test"],
    #     )
    #     titleTxt = L"U_{\infty} = %$flowSpeed \textrm{\,m/s}, α_0 = %$AOA^{\circ}, \Lambda = %$sweepAngle^{\circ}, θ_f = %$fiberAngle^{\circ}"
    #     title = plot(title=titleTxt, grid=false, xticks=false, yticks=false, showaxis=false, bottom_margin=-50Plots.px)
    #     plot(title, visuals, layout=@layout([A{0.1h}; B]))

    #     savefig(outputDir * "tip_dynamics.pdf")
    # end

    if args.is_modal:
        # --- File name ---
        fname = f"{outputDir}/modal-struct.pdf"

        # --- Pack up data to plot ---
        modeShapeData = {
            "structBM": structBendModes,
            "structTM": structTwistModes,
            "wetBM": wetBendModes,
            "wetTM": wetTwistModes,
        }
        modeFreqData = {
            "structNatFreqs": structNatFreqs,
            "wetNatFreqs": wetNatFreqs,
        }

        # --- Plot ---
        fig, axes = plot_mode_shapes(
            y=nodes,
            nModes=nModes,
            modeShapes=modeShapeData,
            modeFreqs=modeFreqData,
            ls=ls,
        )

        dosave = not not fname
        plt.show(block=(not dosave))
        if dosave:
            plt.savefig(fname, format="pdf")
            print("Saved to:", fname)
        plt.close()
