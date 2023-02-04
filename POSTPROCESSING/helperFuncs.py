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
    # Load the data we need
    f = h5py.File(filename, "r")
    return f


def readlines(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines


def get_bendingtwisting(states, nDOF=4):
    """
    Takes the structural 'u' vector and parses it into bending and twisting
    """
    w = states[::nDOF]
    psi = states[2::nDOF] #  * 180 / np.pi

    return w, psi


def load_python_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_tecplot_file(filename):
    return np.loadtxt(filename, skiprows=3)
