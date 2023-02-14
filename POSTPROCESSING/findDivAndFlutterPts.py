from __future__ import print_function
import argparse
import pickle
import numpy as np
import os
from pprint import pprint as pp
import matplotlib.pyplot as plt

import niceplots

parser = argparse.ArgumentParser()
parser.add_argument("histFile", type=str, default="./hist.pkl")
# parser.add_argument("--writeTecFile", type=str, default='')
parser.add_argument("--writeTecFile", action="store_true", default=False)
parser.add_argument(
    "--basefn", type=str, default="divAndflutterPoints", help="Filename identifier appended the default file names"
)
parser.add_argument("--outDir", type=str, default="./")
args = parser.parse_args()

# ------------------------ HELPER FUNCTIONS ------------------------
# Load the data we need
def load_python_obj(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_tecplot_file(filename):
    return np.loadtxt(filename, skiprows=3)


# ------------------------ PARSE ARGUMENTS ------------------------
# Create the output folder
if not os.path.exists(args.outDir):
    os.makedirs(args.outDir)

# Load the exported data
d = load_python_obj(args.histFile)


# ------------------------ ANALYZE MODES --------------------------


def findDivAndFlutterPoints(d, xKey, yKey, debug=False):
    """
    Finds the velocity where damping crosses zero
    Input :
        d : dictionary with flutter solution
        xKey : The key that represents the known variable, usually "pvals_r" or "pmG"
        yKey : Key for the unknown we want to find, usually "dynp"
    Output :
        l : list of tuples that contains the interpolated flutter
            and divergence points. The tuple is (x,y)=(dynp,damping)
    """

    l = []
    print(120 * "=")
    print("Processing for: {0:s} and {1:s}".format(xKey, yKey))
    print(120 * "=")
    for k, v in d.iteritems():
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

                # Check what kind of processing is done
                if xKey.lower() is "pmg" and useSW:
                    print("HELLO")

                print("Found crossing between {0} and {1} at: {2}".format(i, i + 1, yInt), end="")
                l.append([yInt, xInt, k, i])
                break
        # Print new line to close
        print("")

    return l


def writeToTec(l, fName, zoneTxt):

    f = open(fName, "w")
    f.write('VARIABLES= "Dynamic Pressure" "Damping"\n')
    f.write('ZONE T="{0:s}"\n'.format(zoneTxt))
    f.write("DATAPACKING=POINT\n")
    for t in l:
        f.write("{0:4.2f} {1:4.2f}\n".format(t[0], t[1]))
    f.close()
    print("Saving to file: {0:s}".format(fName))


# Find the flutter and divergence points
# l = findDivAndFlutterPoints(d, "pvals_r", "U")
l = findDivAndFlutterPoints(d, "pvals_r", "dynp")

# Save to tec file of needed, check if filename was specified
if args.writeTecFile and len(l) > 0:
    zoneTxt = "Div/Flutter points no boundary"
    fName = args.basefn + ".dat"
    fName = os.path.join(args.outDir, fName)
    writeToTec(l, fName, zoneTxt)
else:
    print("No crossings found for pvals_r")


# Process the pmG (safety window) as well
# l = findDivAndFlutterPoints(d, "pmG", "U")
l = findDivAndFlutterPoints(d, "pmG", "dynp")

if args.writeTecFile and len(l) > 0:
    zoneTxt = "Div/Flutter points with boundary"
    fName = args.basefn + "_pmG.dat"
    fName = os.path.join(args.outDir, fName)
    writeToTec(l, fName, zoneTxt)
else:
    print("No crossings found for pmG")

# Also write out damping values as they would appear on the normal damping plot with the safetywindow
# Need to alter the damping value

for t in l:
    # Get the mode and interpolate the x value needed such that when plotted on the damping plot
    # with the safety window it will show the crossing
    xInt = t[0]
    x = d[t[2]]["dynp"]
    y = d[t[2]]["pvals_r"]
    i = t[3]
    t[1] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) * (xInt - x[i]) + y[i]


if args.writeTecFile and len(l) > 0:
    zoneTxt = "Div/Flutter points with boundary"
    fName = args.basefn + "_sw.dat"
    fName = os.path.join(args.outDir, fName)
    writeToTec(l, fName, zoneTxt)
