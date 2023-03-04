# --- Python 3.9 ---
"""
@File    :   helperFuncs.py
@Time    :   2023/02/03
@Author  :   Galen Ng
@Desc    :   Convenience functions for postprocessing
"""

# ==============================================================================
#                         Imports
# ==============================================================================
import h5py
import numpy as np
import pickle

# ==============================================================================
#                         Helper functions
# ==============================================================================
def load_jld(filename: str):
    """
    Load the data we need
    f is a dictionary
    """
    f = h5py.File(filename, "r")
    return f


def readlines(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def load_python_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_tecplot_file(filename):
    return np.loadtxt(filename, skiprows=3)


def get_bendingtwisting(states, nDOF=4):
    """
    Takes the structural 'u' vector and parses it into bending and twisting
    """
    w = states[::nDOF]
    psi = states[2::nDOF]  #  * 180 / np.pi

    return w, psi


def postprocess_flutterevals(iblankIn, rho, U, dynP, pvals_r, pvals_i):
    """
    Process the raw arrays with eigenvalues so they can be written out.
    Use the blanking array to create new modes

    dataDict = {"1":{
                        "rho"      : [],       # Fluid density [kg/m^3]
                        "U"        : [],       # Fluid velocity [m/s]
                        "dynP"     : [],       # Dynamic pressure [Pa]
                        "p_r"      : [],       # Damping (g, non-dimensional)
                        "p_i"      : [],       # Reduced Frequency (k, non-dimensional)
                        "pvals_r"  : [],       # Damping (gamma, dimensional)
                        "pvals_i"  : [],       # Frequency (omega, dimensional)
                        "pmG"      : [],       # If safety window is enabled then pmG = pvals_r - G else pmG = pvals_r
                        "N"        : [],       # Size of the arrays
                        "KS_pmG    : [],       # KS aggregated damping values
                        },
                "2": {},
                ...etc
            }
    """

    # Dictionary to store all results
    dataDict = {}

    # Turn integer 'iblank' into a boolean array for indexing
    iblank = np.array(iblankIn, dtype=bool)

    # Loop over all the modes
    for mm in range(pvals_r.shape[0]):
        key = f"{mm+1}"  # one-based indexing for key

        # Initialize mode dictionary
        dataDict[key] = {}
        # dataDict[key]["rho"] = rho[iblank[mm, :]]
        dataDict[key]["U"] = U[iblank[mm, :]]
        # dataDict[key]{"dynP"} = dynP[iblank[mm,:]]
        # dataDict[key]{"p_r"} = p_r[mm, iblank[mm,:]]
        # dataDict[key]{"p_i"} = p_i[mm, iblank[mm,:]]
        dataDict[key]["pvals_r"] = pvals_r[mm, iblank[mm, :]]
        dataDict[key]["pvals_i"] = pvals_i[mm, iblank[mm, :]]
        # dataDict[key]{"pmG"} = pmG[mm, iblank[mm,:]] # TODO: no idea what this is for
        dataDict[key]["N"] = iblank[mm, :].shape[0]

    return dataDict
