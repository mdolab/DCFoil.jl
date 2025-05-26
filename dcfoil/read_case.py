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
    print("obj:\t",objectives["Dtot"])
    print("dv:\t",design_vars["alfa0"])
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

    dosave = not not plotname

    niceColors = sns.color_palette("tab10")
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
    cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create figure object
    fig, axes = plt.subplots(nrows=NDV+2, sharex=True, constrained_layout=True, figsize=(10, 10))

    for ii, dv in enumerate(design_vars_vals):
        ax = axes[ii]

        ax.plot(range(0, NITER), design_vars_vals[dv])

        ax.set_ylabel(f"{dv}", rotation="horizontal", ha="right", va="center")
        # ax.set_ylim(bottom=0.0)

    ax = axes[-2]
    ax.plot(range(0, NITER), objectives_vals[obj], label="Dtot")
    ax.set_ylabel(f"{obj}", rotation="horizontal", ha="right", va="center")

    ax = axes[-1]
    ax.plot(range(0, NITER), constraints_vals[con], label="CL")
    ax.set_ylabel(f"{con}", rotation="horizontal", ha="right", va="center")

    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)
        ax.set_xlabel("Iteration")
    if dosave:
        plt.savefig(plotname, format="pdf")
        print("Saved to:", plotname)
    plt.close()
