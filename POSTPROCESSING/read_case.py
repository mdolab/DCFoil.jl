# --- Python 3.11 ---
"""
@File          :   read_case.py
@Date created  :   2025/05/24
@Last modified :   2025/05/24
@Author        :   Galen Ng
@Desc          :   Read a recorder file and look at results.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import json
import sys
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse
from pprint import pprint as pp
from tabulate import tabulate
import juliacall

# ==============================================================================
# Extension modules
# ==============================================================================
from helperFuncs import (
    load_jld,
    readlines,
    get_bendingtwisting,
    postprocess_flutterevals,
    find_DivAndFlutterPoints,
)
from helperPlotFuncs import plot_dragbuildup, plot_dimdragbuildup, plot_vg_vf_rl, set_my_plot_settings, plot_dlf, plot_forced
import niceplots as nplt
import openmdao.api as om

current = os.path.dirname(os.path.realpath(__file__))  # Getting the parent directory name
# adding the parent directory to the sys. path.
sys.path.append(os.path.dirname(current))
from dcfoil.SETUP import setup_dcfoil
from dcfoil.SPECS.point_specs import boatSpds, Fliftstars, opdepths, alfa0


# ==============================================================================
#                         Command line arguments
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--use_serif", help="use serif", action="store_true", default=False)
parser.add_argument("--name", type=str, default=None, help="Name of the case to read .sql file")
parser.add_argument(
    "--base", type=str, default=None, help="Name of the base case to read .sql file and compare against"
)
parser.add_argument("--cases", type=str, default=[], nargs="+", help="Full case folder names in order")
parser.add_argument("--freeSurf", action="store_true", default=False, help="Use free surface corrections")
parser.add_argument("--flutter", action="store_true", default=False, help="Run flutter analysis")
parser.add_argument("--forced", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--pts", type=str, default="3", help="Performance point IDs to run, e.g., 3 is p3")
parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
args = parser.parse_args()
# --- Echo the args ---
print(30 * "-")
print("Arguments are", flush=True)
for arg in vars(args):
    print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
print(30 * "-", flush=True)

# ==============================================================================
#                         DCFoil setup
# ==============================================================================
jl = juliacall.newmodule("DCFoil")
jl.include("../src/io/MeshIO.jl")  # mesh I/O for reading inputs in
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2
files = {
    "gridFile": [
        f"../INPUT/mothrudder_foil_stbd_mesh.dcf",
        f"../INPUT/mothrudder_foil_port_mesh.dcf",
    ]
}

Grid = jl.DCFoil.add_meshfiles(files["gridFile"], {"junction-first": True})

# ==============================================================================
#                         Other settings
# ==============================================================================
outputDir = "PLOTS/"
plotname = f"opt_hist.pdf"
dragplotname = f"drag_hist.pdf"
spanliftname = f"spanwise_properties"

density = 1025.0
semispan = 0.9
nCol = 40  # number of collocation points in code

spanScale = 0.4  # scale factor for the span because of OpenMDAO scaling

# dataDir = "../dcfoil/run_OMDCFoil_out/"
dataDir = "../dcfoil/OUTPUT/"
cm, fs_lgd, fs, linestyles, markers = set_my_plot_settings(args.use_serif)
alphas = [1.0, 0.5]

nNodes = 10
nNodesStrut = 5

dvDictInfo = {  # dictionary of design variable parameters
    "twist": {
        "lower": -15.0,
        "upper": 15.0,
        "scale": 1.0 / 30,
        "value": np.zeros(8 // 2),
    },
    "sweep": {
        "lower": 0.0,
        "upper": 30.0,
        "scale": 1 / 30.0,
        "value": 0.0,
    },
    # "dihedral": { # THIS DOES NOT WORK
    #     "lower": -5.0,
    #     "upper": 5.0,
    #     "scale": 1,
    #     "value": 0.0,
    # },
    "taper": {  # the tip chord can change, but not the root
        "lower": [1.0, 0.5],
        "upper": [1.0, 1.4],
        "scale": 1.0,
        "value": np.ones(2) * 1.0,
    },
    "span": {
        "lower": -0.2,
        "upper": 0.2,
        "scale": 1 / (0.4),
        "value": 0.0,
    },
}
otherDVs = {
    "alfa0": {
        "lower": -10.0,
        "upper": 10.0,
        "scale": 1.0 / (20),  # the scale was messing with the DV bounds
        "value": 4.0,
    },
    "toc": {
        "lower": 0.09,
        "upper": 0.18,
        "scale": 1.0,
        "value": 0.10 * np.ones(nNodes),
    },
    "theta_f": {
        "lower": np.deg2rad(-30),
        "upper": np.deg2rad(30),
        "scale": 1.0 / (np.deg2rad(60)),
        "value": 0.0,
    },
}

# ==============================================================================
#                         Functions
# ==============================================================================
def compute_elliptical(Ltotal, Uinf, semispan, rhof=1000, full_wing=False):
    """
    Plots the elliptical lift distribution given a total lift.

    Parameters
    ----------
    Ltotal : float
        Total lift force [N]
    full_wing : bool, optional
        If True, plots the lift distribution for a full wing, by default True
    """
    sloc = np.linspace(0.0, semispan, 200)
    if full_wing:
        sloc = np.hstack([-sloc[::-1], sloc[1:]])
    # Elliptical lift distribution
    Gamma0 = Ltotal * 4 / np.pi / (rhof * Uinf * 2 * semispan)  # bounding Gamma0

    gamma_s = Gamma0 * np.sqrt(1 - (2 * sloc / (2 * semispan)) ** 2)  # Elliptical lift distribution in gamma
    Lprime = rhof * Uinf * gamma_s  # Lift per unit span Kutta Joukowski

    return sloc, Lprime, gamma_s


def plot_dragiter():
    dosave = not not dragplotname

    # Create figure object
    fig, axes = plt.subplots(nrows=2, ncols=npts, sharex=True, constrained_layout=True, figsize=(8 * npts, 12))

    for (ii, ptName) in enumerate(probList):
        systemName = f"dcfoil_{ptName}"

        waveDrag = drag_vals[f"Dw_{ptName}"]
        profileDrag = drag_vals[f"Dpr_{ptName}"]
        inducedDrag = drag_vals[f"Fdrag_{ptName}"]
        waveDrag_cd = drag_vals[f"CDw_{ptName}"]
        profileDrag_cd = drag_vals[f"CDpr_{ptName}"]
        inducedDrag_cd = drag_vals[f"CDi_{ptName}"]

        totalDrags = np.array(waveDrag) + np.array(profileDrag) + np.array(inducedDrag)
        print("Total drag values:", totalDrags)

        ax = axes[0, ii]
        ax.plot(
            range(0, NITER),
            np.array(profileDrag) / totalDrags * 100,
            label="Profile drag",
            color=cm[1],
            # ls=linestyles[1],
        )
        ax.plot(
            range(0, NITER),
            np.array(inducedDrag) / totalDrags * 100,
            label="Induced drag",
            color=cm[2],
            # ls=linestyles[2],
        )
        ax.plot(
            range(0, NITER),
            np.array(waveDrag) / totalDrags * 100,
            label="Wave drag",
            color=cm[0],  # ls=linestyles[0]
        )
        ax.legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1)
        ax.set_ylim(bottom=0.0, top=100.0)
        yticks_list = [
            (waveDrag)[-1].item() / totalDrags[-1].item() * 100,
            (profileDrag)[-1].item() / totalDrags[-1].item() * 100,
            (inducedDrag)[-1].item() / totalDrags[-1].item() * 100,
        ]
        ax.set_yticks([0, 50, 100] + yticks_list)
        ax.set_xlim(left=0)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Drag\nbreakdown\n[\%]", rotation="horizontal", ha="left", va="center")

        ax = axes[1, ii]
        ax.plot(
            range(0, NITER),
            np.array(profileDrag),
            label="Profile drag",
            color=cm[1],
            # ls=linestyles[1],
        )
        ax.plot(
            range(0, NITER),
            np.array(inducedDrag),
            label="Induced drag",
            color=cm[2],
            # ls=linestyles[2],
        )
        ax.plot(
            range(0, NITER),
            np.array(waveDrag),
            label="Wave drag",
            color=cm[0],  # ls=linestyles[0]
        )
        yticks_list = [
            (waveDrag)[-1].item(),
            (profileDrag)[-1].item(),
            (inducedDrag)[-1].item(),
        ]
        ax.set_yticks(yticks_list)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("Drag\nbreakdown\n[N]", rotation="horizontal", ha="left", va="center")

    for ax in axes.flatten():
        ax.set_xlabel("Iteration")
        nplt.adjust_spines(ax, ["right", "bottom"], outward=True)
    plt.savefig(dragplotname, format="pdf")
    print("Saved to:", dragplotname)
    plt.close()

def plot_spanwise():

    drag_vals = {}
    for ptName in probList:
        drag_vals[f"Dw_{ptName}"] = []
        drag_vals[f"Dpr_{ptName}"] = []
        drag_vals[f"Fdrag_{ptName}"] = []
        drag_vals[f"CDw_{ptName}"] = []
        drag_vals[f"CDpr_{ptName}"] = []
        drag_vals[f"CDi_{ptName}"] = []

    # Create figure object
    fig, axes = plt.subplots(nrows=5, ncols=npts, sharex="col", constrained_layout=True, figsize=(10 * npts, 16))

    for icase, case in enumerate(args.cases):
        datafname = f"../dcfoil/OUTPUT/{case}/{case}.sql"

        cr = om.CaseReader(datafname)

        # ax = axes[0, 0]
        ax = axes.flatten()[0]
        ax.annotate(
            f"{case}",
            xy=(0.0, 0.3 - 0.13 * icase),
            xycoords="axes fraction",
            color=cm[icase],
            ha="left",
            fontsize=fs_lgd,
        )

        for (ii, ptName) in enumerate(probList):

            systemName = f"dcfoil_{ptName}"

            Uinf = boatSpds[ptName]  # boat speed [m/s]
            axes[0, ii].set_title(f"{ptName}: $U_\infty={Uinf:.1f}$ m/s", fontsize=fs * 1.1)

            dcfoil_cases = cr.list_cases(f"root.{systemName}", recurse=False)
            # for case_num, case_id in enumerate(dcfoil_cases[:-1]):
            case_id = dcfoil_cases[-1]
            case_num = -1
            dcfoil_case = cr.get_case(case_id)

            # dcfoil_case.inputs

            waveDrag = dcfoil_case.outputs[f"{systemName}.Dw"]
            profileDrag = dcfoil_case.outputs[f"{systemName}.Dpr"]
            inducedDrag = dcfoil_case.outputs[f"{systemName}.Fdrag"]
            waveDrag_cd = dcfoil_case.outputs[f"{systemName}.CDw"]
            profileDrag_cd = dcfoil_case.outputs[f"{systemName}.CDpr"]
            inducedDrag_cd = dcfoil_case.outputs[f"{systemName}.CDi"]
            drag_vals[f"Dw_{ptName}"].append(waveDrag)
            drag_vals[f"Dpr_{ptName}"].append(profileDrag)
            drag_vals[f"Fdrag_{ptName}"].append(inducedDrag)
            drag_vals[f"CDw_{ptName}"].append(waveDrag_cd)
            drag_vals[f"CDpr_{ptName}"].append(profileDrag_cd)
            drag_vals[f"CDi_{ptName}"].append(inducedDrag_cd)

            spanwise_force_vector = dcfoil_case.outputs[f"{systemName}.forces_dist"]
            circ_dist = dcfoil_case.outputs[f"{systemName}.gammas"]
            aeroNodesXYZ = dcfoil_case.outputs[f"{systemName}.collocationPts"]
            femNodesXYZ = dcfoil_case.outputs[f"{systemName}.nodes"]
            spanwise_cl = dcfoil_case.outputs[f"{systemName}.cl"]
            displacements_col = dcfoil_case.outputs[f"{systemName}.displacements_col"]

            ventilationCon = dcfoil_case.outputs[f"{systemName}.ksvent"]

            try:
                spanVal = design_vars_vals["span"][case_num] * spanScale
            except KeyError:
                spanVal = 0.0
            TotalLift = dcfoil_case.outputs[f"{systemName}.Flift"]
            ax = axes[0,ii]
            ax.annotate(f"{TotalLift[0]:.0f} N",
                xy=(0.5, 0.3 - 0.13 * icase),
                xycoords="axes fraction",
                color=cm[icase],
                ha="left",
                fontsize=fs_lgd,
            )
            print("Total lift:", TotalLift)
            print("Span value:", spanVal)
            sloc, Lprime, gamma_s = compute_elliptical(TotalLift, Uinf, semispan + spanVal, density)

            ax = axes[0, ii]
            rhoU = density * Uinf  # rho * U
            ax.plot(sloc, Lprime, "-", c="k", alpha=0.5, label="Elliptical  distribution")
            ax.plot(aeroNodesXYZ[1, :], circ_dist[nCol // 2 :] * Uinf * rhoU, c=cm[icase])
            # ax.plot(sloc, gamma_s, "-", c="k", alpha=0.5, label="Elliptical lift distribution")
            # ax.plot(aeroNodesXYZ[1, :], circ_dist[nCol // 2 :] * Uinf, "-")
            if icase == 0:
                ax.legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

            # ax.set_ylabel("Lift [N]", rotation="horizontal", ha="right", va="center")
            if ii == 0:
                # ax.set_ylabel("$\\Gamma$ [m$^2$/s]", rotation="horizontal", ha="right", va="center")
                ax.set_ylabel("$L'$ [N/m]", rotation="horizontal", ha="right", va="center")
            # ax.set_ylim(bottom=0.0, top=150)

            ax = axes[1, ii]
            ax.plot(aeroNodesXYZ[1, :], spanwise_cl, c=cm[icase])
            # plot horizontal line for cl_in
            ax.axhline(np.max(spanwise_cl - ventilationCon), ls="--", label="cl Ventilation", color="magenta")
            ax.annotate(
                "$c_{\ell_{in}}=$" + f"${np.max(spanwise_cl - ventilationCon):.2f}$",
                xy=(0.85, 0.99),
                xycoords="axes fraction",
                color="magenta",
            )
            if ii == 0:
                ax.set_ylabel("$c_\ell$", rotation="horizontal", ha="right", va="center")
            ax.set_ylim(top=np.max(spanwise_cl - ventilationCon) * 1.1, bottom=0.0)

            # --- Twist distribution ---
            ax = axes[2, ii]
            # ax.plot(aeroNodesXYZ[1, :], np.rad2deg(displacements_col[4, nCol // 2 :]), label="Deflections")
            # spanY = np.linspace(0, semispan + spanVal, len(design_vars_vals["twist"][case_num]) + 1)
            # twistDist = np.hstack((0.0, design_vars_vals["twist"][case_num]))
            # ax.plot(spanY, twistDist,"s", label="Jig twist (FFD)",zorder=10, clip_on=False)

            ptVec = dcfoil_case.inputs[f"{systemName}.hydroelastic.liftingline.ptVec"]
            LECoords, TECoords = jl.LiftingLine.repack_coords(
                ptVec, 3, len(ptVec) // 3
            )  # repack the ptVec to a 3D array
            idxTip = jl.LiftingLine.get_tipnode(LECoords)

            midchords, _, _, _, pretwistDist = jl.LiftingLine.compute_1DPropsFromGrid(
                LECoords,
                TECoords,
                Grid.nodeConn,
                idxTip,
                appendageOptions=appendageOptions,
                appendageParams=appendageParams,
            )
            pretwistAeroNodes = np.interp(
                aeroNodesXYZ[1, :], midchords[1, :idxTip], np.rad2deg(pretwistDist[:idxTip])
            )  # interpolate to match the aero nodes
            # ax.plot(midchords[1,:idxTip], np.rad2deg(pretwistDist[:idxTip]), label="Jig twist (1D props)")
            ax.plot(aeroNodesXYZ[1, :], pretwistAeroNodes, c=cm[icase], label="Jig twist", alpha=0.5)
            ax.plot(
                aeroNodesXYZ[1, :],
                np.rad2deg(displacements_col[4, nCol // 2 :]) + pretwistAeroNodes,
                c=cm[icase],
                ls="-",
                label="In-flight twist",
            )

            if ii == 0:
                ax.set_ylabel("Twist [deg]", rotation="horizontal", ha="right", va="center")
            ax.set_ylim(bottom=-3.0, top=8.0)
            if icase == len(args.cases) - 1:
                ax.legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

            ax = axes[3, ii]
            ax.plot(aeroNodesXYZ[1, :], displacements_col[2, nCol // 2 :], c=cm[icase])
            if ii == 0:
                ax.set_ylabel("OOP bending [m]", rotation="horizontal", ha="right", va="center")
            tipConstraint = semispan * 0.05
            ax.axhline(tipConstraint, ls="--", color="magenta")
            ax.set_ylim(top=tipConstraint * 1.1)
            ax.annotate("$\\delta_{\\textrm{max}}$", xy=(0.92, 0.99), xycoords="axes fraction", color="magenta")
            ax.set_yticks(np.arange(0, 0.01, 0.005).tolist() + [tipConstraint])
            ax.set_xticks([0.0, semispan, semispan + spanVal.item()])

            ax = axes[4, ii]
            ax.plot(femNodesXYZ[:, 1], design_vars_vals["toc"][case_num, :], c=cm[icase])
            if ii == 0:
                ax.set_ylabel("$t/c$", rotation="horizontal", ha="right", va="center")

    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)
    for ax in axes[-1, :]:
        ax.set_xlabel("Spanwise position [m]")
    plt.suptitle(f"Spanwise properties for {args.name}", fontsize=fs * 1.1)
    plt.savefig(spanliftname + f".pdf", format="pdf")
    print("Saved to:", spanliftname + f".pdf")
    plt.close()

def plot_dragbuildupcomp():
    Dtotal = 0
    Dtotalopt = 0
    fname = f"drag_buildup_{args.name}-vs-{args.base}.pdf"
    # Create figure object
    fig, axes = plt.subplots(nrows=1, sharex=True, constrained_layout=True, figsize=(14, 10))
    for (icase, case) in enumerate(args.cases):

        # basename = f"../dcfoil/run_OMDCfoil_out/{case}.sql"
        basename = f"../dcfoil/OUTPUT/{case}/{case}.sql"
        cr = om.CaseReader(basename)

        for ii, ptName in enumerate(probList):
            systemName = f"dcfoil_{ptName}"
            dcfoil_cases = cr.list_cases(f"root.{systemName}", recurse=False)

            # last case
            case_id = dcfoil_cases[-1]
            case_num = -1
            dcfoil_case = cr.get_case(case_id)

            waveDrag = dcfoil_case.outputs[f"{systemName}.Dw"]
            profileDrag = dcfoil_case.outputs[f"{systemName}.Dpr"]
            inducedDrag = dcfoil_case.outputs[f"{systemName}.Fdrag"]
            waveDrag_cd = dcfoil_case.outputs[f"{systemName}.CDw"]
            profileDrag_cd = dcfoil_case.outputs[f"{systemName}.CDpr"]
            inducedDrag_cd = dcfoil_case.outputs[f"{systemName}.CDi"]
            basefuncs = {
                "cdpr": profileDrag_cd[0],
                "cdi": inducedDrag_cd[0],
                "cdw": waveDrag_cd[0],
            }

            basedragfuncs = {
                "Dpr": profileDrag[0],
                "Di": inducedDrag[0],
                "Dw": waveDrag[0],
            }

            Dtotal += basedragfuncs["Dpr"] + basedragfuncs["Di"] + basedragfuncs["Dw"]

        includes = ["cdpr", "cdi", "cdw"]

        # fig, axes = plot_dragbuildup(fig, axes, basefuncs, "Baseline", cm, 15, 0, includes=includes)
        # fig, axes = plot_dragbuildup(fig, axes, optfuncs, "Optimized", cm, 15, 1, includes=includes)

        includes = ["Dpr", "Di", "Dw"]
        fig, axes = plot_dimdragbuildup(fig, axes, basedragfuncs, f"Drag breakdown", f"{case} ($D={Dtotal:.1f}$)", cm, 15, icase, includes=includes)

        # axes.annotate(f"$D={Dtotalbase:.1f}$", xy=(axes-fractions), xycoords="axes fraction")

    axes.legend(fontsize=fs_lgd, labelcolor="linecolor", loc="best", frameon=False, ncol=1)
    nplt.adjust_spines(axes, outward=True)

    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()

# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    probList = []
    for pt in args.pts:
        probList.append(f"p{pt}")
    npts = len(probList)

    (
        ptVec,
        nodeConn,
        appendageParams,
        appendageOptions,
        solverOptions,
        npt_wing,
        npt_wing_full,
        n_node,
    ) = setup_dcfoil.setup(nNodes, nNodesStrut, args, None, files, [0.5,1.0], None)

    datafname = f"../dcfoil/run_OMDCfoil_out/{args.name}.sql"
    datafname = f"../dcfoil/OUTPUT/{args.cases[0]}/{args.name}.sql"

    cr = om.CaseReader(datafname)

    driver_cases = cr.list_cases("driver", recurse=False, out_stream=None)

    # ************************************************
    #     Last case
    # ************************************************
    last_case = cr.get_case(driver_cases[-1])

    objectives = last_case.get_objectives()
    design_vars = last_case.get_design_vars()
    constraints = last_case.get_constraints()
    print("obj:\t", objectives["Dtot"])
    print("dv:")
    pp(design_vars)
    print("constraints:")
    pp(constraints)
    # print(constraints["CL"])

    # ************************************************
    #     Plot path of DVs
    # ************************************************
    design_vars_vals = {}
    for dv, val in design_vars.items():
        design_vars_vals[dv] = []

    objectives_vals = {}

    for obj, val in objectives.items():
        objectives_vals[obj] = []

    constraints_vals = {}

    for con, val in constraints.items():
        constraints_vals[con] = []

    NDV = len(design_vars_vals)
    NITER = len(driver_cases)
    NCON = len(constraints_vals)
    for case in driver_cases:
        current_case = cr.get_case(case)
        case_design_vars = current_case.get_design_vars()
        for dv, val in case_design_vars.items():
            design_vars_vals[dv].append(val)

        case_objectives = current_case.get_objectives()
        for obj, val in case_objectives.items():
            objectives_vals[obj].append(val)

        case_constraints = current_case.get_constraints()
        for con, val in case_constraints.items():
            constraints_vals[con].append(val)

    for dv, val in case_design_vars.items():
        design_vars_vals[dv] = np.array(design_vars_vals[dv])
    for obj, val in case_objectives.items():
        objectives_vals[obj] = np.array(objectives_vals[obj])
    for con, val in case_constraints.items():
        constraints_vals[con] = np.array(constraints_vals[con])

    dosave = not not plotname

    # Create figure object
    nrows = np.max([NDV, NCON + 1])
    # fig, opthistaxes = plt.subplots(nrows=nrows, ncols=2, sharex=True, constrained_layout=True, figsize=(16, 3 * nrows))
    fig, opthistaxes = plt.subplots(nrows=2, ncols=nrows, sharex=True, constrained_layout=True, figsize=(5*nrows, 8))

    for ii, dv in enumerate(design_vars_vals):
        # ax = opthistaxes[ii, 0]
        ax = opthistaxes[0, ii]

        # Check if dv key is in dvDictInfo, if not use otherDVs
        if dv in dvDictInfo:
            scaleFactor = 1 / dvDictInfo[dv]["scale"]
        elif dv in otherDVs:
            scaleFactor = 1 / otherDVs[dv]["scale"]
        elif "alfa0" in dv:
            scaleFactor = 1 / otherDVs["alfa0"]["scale"]

        if dv in ["theta_f"]:
            design_vars_vals[dv] *= 180 / np.pi * scaleFactor  # convert to degrees
        if design_vars_vals[dv].ndim != 1:
            for jj in range(design_vars_vals[dv].shape[1]):
                ax.plot(
                    range(0, NITER),
                    design_vars_vals[dv][:, jj] * scaleFactor,
                    label=f"{dv}-{jj}",
                    color=cm[jj],
                    ls=linestyles[jj % len(linestyles)],
                )

            ax.legend(
                fontsize=fs_lgd,
                labelcolor="linecolor",
                loc="upper left",
                frameon=False,  # ncol=design_vars_vals[dv].shape[1]
            )
        else:
            ax.plot(range(0, NITER), design_vars_vals[dv])

        ax.set_ylabel(f"{dv}", rotation="horizontal", ha="right", va="center")
        # ax.set_ylim(bottom=0.0)

        # print(f"{dv} values:")
        # print(tabulate(design_vars_vals[dv], headers="keys", tablefmt="grid"))
        # print(30 * "-")
        # breakpoint()

    # ax = opthistaxes[0, 1]
    ax = opthistaxes[1, 0]
    ax.plot(range(0, NITER), objectives_vals[obj], label="Dtot")
    ax.set_ylabel(f"{obj}", rotation="horizontal", ha="right", va="center")

    for ii, con in enumerate(constraints_vals):
        # ax = opthistaxes[1 + ii, 1]
        ax = opthistaxes[1, 1 + ii]
        ax.plot(range(0, NITER), constraints_vals[con])
        ax.set_ylabel(f"{con.lstrip('dcfoil.')}", rotation="horizontal", ha="right", va="center")

    for ax in opthistaxes.flatten():
        nplt.adjust_spines(ax, outward=True)

    for ax in opthistaxes[-1, :]:
        ax.set_xlabel("Iteration")
    if dosave:
        plt.savefig(plotname, format="pdf")
        print("Saved to:", plotname)
    plt.close()

    # ************************************************
    #     Spanwise history too
    # ************************************************
    if not args.flutter and not args.forced:
        plot_spanwise()

    #     # ************************************************
    #     #     Also plot drag breakdown vs. iteration
    #     # ************************************************
    #    plot_dragiter()

    # ************************************************
    #     Compare drag buildup with base case
    # ************************************************
    if not args.flutter and not args.forced:
        plot_dragbuildupcomp()

    # Input data read directory
    caseDirs = []
    if args.cases is not None:
        for case in args.cases:
            for ptName in probList:
                caseDirs.append(dataDir + case + f"/{ptName}")
    else:
        raise ValueError("Please specify a case to run postprocessing on")
    
    if args.flutter:
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
        units = "m/s"
        for ii, key in enumerate(args.cases):
            if ii == 0:
                annotateModes = True
            else:
                annotateModes = False

            # # can force to see modes in legend label
            # annotateModes = False

            # --- Plot ---
            fig, axes = plot_vg_vf_rl(
                fig,
                axes,
                flutterSol=flutterSolDict[key],
                cm=cm,
                ls=linestyles[ii],
                alpha=alphas[ii],
                units=units,
                # marker="o",
                showRLlabels=True,
                annotateModes=annotateModes,
                # nShift=62,
                instabPts=instabPtsDict[key],
            )
            if units == "kts":
                unitFactor = 1.9438
            else:
                unitFactor = 1.0

            if len(instabPtsDict[key]) != 0:
                instabSpeedTicks.append(instabPtsDict[key][0][0] * unitFactor)
                instabFreqTicks.append(instabPtsDict[key][0][-1])

        # --- Set limits ---
        # axes[0, 0].set_xticks([10, 60] + instabSpeedTicks)
        # axes[1, 0].set_yticks([0, 200, 400, 600, 800] + instabFreqTicks)

        # axes[0, 0].set_ylim(top=30, bottom=-400)
        # axes[1, 0].set_ylim(top=0.005, bottom=0)
        # axes[0, 0].set_xlim(left=160, right=175)

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
                ls=linestyles[ii],
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

    if args.forced:
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
            rao = np.asarray(load_jld(f"{caseDir}/forced/GenXferFcn.jld2")["data"]).T
            deflectionRAO = np.asarray(load_jld(f"{caseDir}/forced/deflectionRAO.jld2")["data"]).T

            # ************************************************
            #     Plot the frequency spectra
            # ************************************************
            fname = f"{outputDir}/forced-dynamics.pdf"

            # Create figure object
            nrows = 2
            ncols = 4
            figsize = (20 * ncols, 4 * nrows)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex="col",
                constrained_layout=True,
                figsize=figsize,
            )

            # fiberAngle = np.rad2deg(DVDictDict[key]["theta_f"])
            # flowSpeed = SolverOptions[key]["Uinf"]
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
                boatSpds[ptName],
                fs_lgd,
                cm,
            )

            fig.suptitle("Frequency response spectra")

            for ax in axes.flatten():
                ax.set_xlim(left=0.0)

            for ax in axes[0, :].flatten():
                ax.set_ylim(bottom=0.0)

            dosave = not not fname
            plt.show(block=(not dosave))
            if dosave:
                plt.savefig(fname, format="pdf")
                print("Saved to:", fname)
            plt.close()