# --- Python 3.9 ---
"""
@File    :   helperFuncs.py
@Time    :   2023/02/03
@Author  :   Galen Ng with functions from Eirikur Jonsson
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
    Load data from a .jld file
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
    from Eirikur Jonsson with modifications by Galen Ng
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
        # dataDict[key]{"pmG"} = pmG[mm, iblank[mm,:]]
        dataDict[key]["N"] = iblank[mm, :].shape[0]

    return dataDict


def find_DivAndFlutterPoints(d: dict, xKey: str, yKey: str, debug=False):
    """
    Finds the velocity where damping crosses zero
    Input :
        d : dictionary with flutter solution
        xKey : The key that represents the known variable, usually "pvals_r" or "pmG"
        yKey : Key for the unknown we want to find, usually "dynp"
    Output :
        instabPts : list of tuples that contains the interpolated flutter
            and divergence points. The tuple is (x,y)=(dynp, damping)
    From Eirikur Jonsson with modifications by Galen Ng
    """

    instabPts = []
    print(120 * "=")
    print("Processing for: {0:s} and {1:s}".format(xKey, yKey))
    print(120 * "=")
    for k, v in d.items():
        print("Mode: {0:10s}".format(k), end="")

        x = d[k][xKey]  # This is either "pvals_r" or "pmG"
        y = d[k][yKey]  # This is "U"

        asign = np.sign(x)
        # Search for the first sign change since that indicates the flutter point
        for i in range(len(asign) - 1):
            if asign[i] != asign[i + 1]:

                if debug:
                    print("")
                    print(asign[i])
                    print(asign[i + 1])
                    print(x[i])
                    print(x[i + 1])

                # Interpolate the value at x=0 or zero damping since that is the accurate crossing
                xInt = 0.0
                yInt = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (xInt - x[i]) + y[i]

                # # Check what kind of processing is done
                # if xKey.lower() is "pmg" and useSW:
                #     print("HELLO")

                print(f"Found crossing between {i} and {i+1} at U = {yInt} m/s ({yInt*1.9438:.3f} kts)", end="")
                instabPts.append([yInt, xInt, k, i])
                break
        # Print new line to close
        print("")

    return instabPts
