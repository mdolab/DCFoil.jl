"""
Utils functions for convenience specific to MDO lab framework
"""

import pickle as pkl
import os
import shutil
from baseclasses.utils import Error


# ==============================================================================
#                         MACH Framework helper functions
# ==============================================================================
# Load the data we need
def load_python_obj(filename):
    with open(filename, "rb") as f:
        return pkl.load(f)


def save_python_obj(obj, filename):
    with open(filename + ".pkl", "wb") as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)


def backupFiles(files, copyDir):
    # Create a folder called copyDir to save input files.
    os.system("mkdir -p %s" % copyDir)

    # Copy all the files
    for key in files:
        try:
            shutil.copy(files[key], copyDir)
        except FileNotFoundError:
            Error("Failed to copy file {0:s} to {1:s}".format(key, copyDir))


# ************************************************
#     JSON Numpy encoder for human readable output
# ************************************************
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.astype("d").tolist()
        return json.JSONEncoder.default(self, obj)


# ************************************************
#     Pretty print important information
# ************************************************
def print_AP_table(aps: list, clstars: dict, hydrofoil, Rlist: list):
    """
    Print out setup conditions for all problems
    Inputs
    ------
        aps: list of AeroProblem objects
        clstars: dict of clstar values
        hydrofoil: Hydrofoil object
        Rlist: list of gas constants
    """
    headers = [
        "apname",
        "alpha [deg]",
        "beta [deg]",
        "CLstar",
        "Ma",
        "V [m/s]",
        "Vb [kts]",
        "areaRef [m^2]",
        "Re",
        "gas R [J/(kg-K)]",
    ]
    table = []
    flowHeaders = ["rho [kg/m^3]", "Dyn visc. mu [kg/(m-s)]"]
    flowTable = []
    for ii, ap in enumerate(aps):
        table.append(
            [
                ap.name,
                ap.alpha,
                ap.beta,
                clstars[ap.name],
                ap.mach,
                ap.V,
                ap.V * 1.9438,
                hydrofoil.areaRef,
                ap.re * hydrofoil.chordRef,
                Rlist[ii],
            ]
        )
        flowTable.append([ap.rho, ap.mu])
    try:
        from tabulate import tabulate

        print(20 * "=")
        print("AeroProblem setup:")
        print(20 * "=")
        print(tabulate(flowTable, flowHeaders))
        print(tabulate(table, headers))
        print(20 * "=")
        print("Latex output:")
        print(20 * "=")
        print(tabulate(table, headers, tablefmt="latex"))
        print("\n")
    except:
        print("Install tabulate to print tables")
        print(flowHeaders)
        print(flowTable)
        print(headers)
        print(table)
        print("", flush=True)


def print_CP_results(cp, funcs: dict, is_asp=True):
    """
    Convenience print results as a table

    Parameters
    ----------
    cp : _type_
        _description_
    funcs : _type_
        _description_
    """
    CLtxt = funcs[f"{cp.name}_cl"]
    CDtxt = funcs[f"{cp.name}_cd"]
    Cpmintxt = funcs[f"{cp.name}_target_cpmin"]
    if is_asp:
        alphatxt = cp.AP.alpha
    else:
        alphatxt = cp.alpha
    headers = ["alpha [deg]", "CL", "CD", "Cpmin"]
    table = [[f"{alphatxt}", f"{CLtxt}", f"{CDtxt}", f"{Cpmintxt}"]]
    print(f"{cp.name:<5}")
    try:
        from tabulate import tabulate

        print(tabulate(table, headers))
    except ImportError:
        print("Install tabulate to print tables")


import h5py


def load_jld(filename: str):
    """
    Load data from a .jld or .jld2 file
    f is a dictionary
    """
    f = h5py.File(filename, "r")
    return f
