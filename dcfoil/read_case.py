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

fname = f"./run_OMDCfoil_out/dcfoil.sql"
plotname = f"opt_hist.pdf"
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_serif", help="use serif", action="store_true", default=False)
    args = parser.parse_args()

    cr = om.CaseReader(fname)

    driver_cases  = cr.list_cases("driver", recurse=False,out_stream=None)

    # ************************************************
    #     Last case
    # ************************************************
    last_case = cr.get_case(driver_cases[-1])

    objectives = last_case.get_objectives()
    design_vars = last_case.get_design_vars()
    constraints = last_case.get_constraints()
    print(objectives["Dtot"])
    print(design_vars["alfa0"])
    # print(constraints["CL"])

    # ************************************************
    #     Plot path of DVs
    # ************************************************
    design_vars_vals = {}
    for dv, val in design_vars.items():
        design_vars_vals[dv] = []
    
    NDV = len(design_vars_vals)
    for case in driver_cases:
        current_case = cr.get_case(case)
        case_design_vars = current_case.get_design_vars()
        for dv, val in case_design_vars.items():
            design_vars_vals[dv].append(val["value"])
    
    dosave = not not plotname
    
    niceColors = sns.color_palette("tab10")
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
    cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    # Create figure object
    fig, axes = plt.subplots(nrows=NDV, sharex=True, constrained_layout=True, figsize=(14, 10))
    
    ax = axes
    
    
    ax.set_xlabel("xlabel")
    ax.set_ylabel("ylabel", rotation="horizontal", ha="right", va="center")
    
    plt.show(block=(not dosave))
    # nplt.all()
    for ax in axes.flatten():
        nplt.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()