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
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

# from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
import niceplots as nplt
import openmdao.api as om


datafname = f"./run_OMDCfoil_out/dcfoil.sql"
plotname = f"opt_hist.pdf"
plot2name = f"drag_hist.pdf"
spanliftname = f"spanwise_lift"

# ==============================================================================
#                         Other settings
# ==============================================================================
linestyles = ["-", "--", "-.", ":"]
niceColors = sns.color_palette("tab10")
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]

Uinf = 11
semispan = 0.333


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
    Lprime = rhof * Uinf * gamma_s  # Lift per unit span

    return sloc, Lprime


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_serif", help="use serif", action="store_true", default=False)
    args = parser.parse_args()

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
    print("dv:\t", design_vars["alfa0"])
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
    fig, axes = plt.subplots(nrows=NDV, ncols=2, sharex=True, constrained_layout=True, figsize=(13, 10))

    for ii, dv in enumerate(design_vars_vals):
        ax = axes[ii, 0]

        if design_vars_vals[dv].ndim != 1:
            for jj in range(design_vars_vals[dv].shape[1]):
                ax.plot(
                    range(0, NITER),
                    design_vars_vals[dv][:, jj],
                    label=f"{dv}-{jj}",
                    color=cm[jj],
                    ls=linestyles[jj % len(linestyles)],
                )
            ax.legend(
                fontsize=10, labelcolor="linecolor", loc="upper left", frameon=False, ncol=design_vars_vals[dv].shape[1]
            )
        else:
            ax.plot(range(0, NITER), design_vars_vals[dv])

        ax.set_ylabel(f"{dv}", rotation="horizontal", ha="right", va="center")
        # ax.set_ylim(bottom=0.0)

    ax = axes[0, 1]
    ax.plot(range(0, NITER), objectives_vals[obj], label="Dtot")
    ax.set_ylabel(f"{obj}", rotation="horizontal", ha="right", va="center")

    for ii, con in enumerate(constraints_vals):
        ax = axes[1 + ii, 1]
        ax.plot(range(0, NITER), constraints_vals[con], label="CL")
        ax.set_ylabel(f"{con}", rotation="horizontal", ha="right", va="center")

    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)
        ax.set_xlabel("Iteration")
    if dosave:
        plt.savefig(plotname, format="pdf")
        print("Saved to:", plotname)
    plt.close()

    # ************************************************
    #     Check out history too
    # ************************************************
    dcfoil_cases = cr.list_cases("root.dcfoil", recurse=False)

    drag_vals = {}
    for case_num, case_id in enumerate(dcfoil_cases):
        dcfoil_case = cr.get_case(case_id)

        # dcfoil_case.inputs

        waveDrag = dcfoil_case.outputs["dcfoil.Dw"]
        profileDrag = dcfoil_case.outputs["dcfoil.Dpr"]
        inducedDrag = dcfoil_case.outputs["dcfoil.Fdrag"]
        spanwise_force_vector = dcfoil_case.outputs["dcfoil.forces_dist"]
        circ_dist = dcfoil_case.outputs["dcfoil.gammas"]
        aeroNodesXYZ = dcfoil_case.outputs["dcfoil.collocationPts"]
        spanwise_cl = dcfoil_case.outputs["dcfoil.cl"]
        displacements_col = dcfoil_case.outputs["dcfoil.displacements_col"]

        ventilationCon = dcfoil_case.outputs["dcfoil.ksvent"]

        sloc, Lprime = compute_elliptical(
            dcfoil_case.outputs["dcfoil.Flift"], Uinf, semispan + design_vars_vals["span"][case_num], 1025
        )

        # Create figure object
        fig, axes = plt.subplots(nrows=4, sharex=True, constrained_layout=True, figsize=(9, 9))

        ax = axes[0]
        rhoU = 1000 * 11.0  # rho * U
        ax.plot(sloc, Lprime, "-", c="k", alpha=0.5, label="Elliptical lift distribution")
        ax.plot(aeroNodesXYZ[1, :], circ_dist[10:] * 11.0 * rhoU, "-")
        ax.legend(fontsize=10, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

        ax.set_ylabel("Lift [N]", rotation="horizontal", ha="right", va="center")
        ax.set_ylabel("$\\Gamma$ [m$^2$/s]", rotation="horizontal", ha="right", va="center")
        ax.set_ylabel("$L'$ [N/m]", rotation="horizontal", ha="right", va="center")
        # ax.set_ylim(bottom=0.0, top=150)

        ax = axes[1]
        ax.plot(aeroNodesXYZ[1, :], spanwise_cl, "-")
        # plot horizontal line
        ax.axhline(np.max(spanwise_cl - ventilationCon), ls="--", label="cl Ventilation", color="magenta")
        ax.annotate("$c_{\ell_{in}}$", xy=(0.9, 0.99), xycoords="axes fraction", color="magenta")
        ax.set_ylabel("$c_\ell$", rotation="horizontal", ha="right", va="center")

        ax = axes[2]
        ax.plot(aeroNodesXYZ[1, :], np.rad2deg(displacements_col[4, 10:]), label="Deflected twist")
        spanY = np.linspace(
            0, semispan + design_vars_vals["span"][case_num], len(design_vars_vals["twist"][case_num]) + 1
        )
        twistDist = np.hstack((0.0, design_vars_vals["twist"][case_num]))
        ax.plot(spanY, twistDist, label="Pre-twist")
        ax.set_ylabel("Twist [deg]", rotation="horizontal", ha="right", va="center")
        ax.legend(fontsize=10, labelcolor="linecolor", loc="best", frameon=False, ncol=1)

        ax = axes[3]
        ax.plot(aeroNodesXYZ[1, :], displacements_col[2, 10:])
        ax.set_ylabel("OOP bending [m]", rotation="horizontal", ha="right", va="center")
        ax.axhline(semispan * 0.05, ls="--", color="magenta")
        ax.annotate("$\\delta_{max}$", xy=(0.9, 0.99), xycoords="axes fraction", color="magenta")

        for ax in axes.flatten():
            ax.set_xlabel("Spanwise position [m]")
            nplt.adjust_spines(ax, outward=True)

        plt.savefig(spanliftname + f"-{case_num}.pdf", format="pdf")
        print("Saved to:", spanliftname + f"-{case_num}.pdf")
        plt.close()
