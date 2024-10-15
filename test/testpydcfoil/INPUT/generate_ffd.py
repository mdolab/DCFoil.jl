# --- Python 3.10 ---
"""
@File          :   generate_ffd.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   Make FFDs
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
from scipy import sparse

# from tabulate import tabulate


output_dir = f"./"

n_chord = 2
n_span = 5
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    # ************************************************
    #     Command line arguments
    # ************************************************
    parser = argparse.ArgumentParser()
    parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
    parser.add_argument("--semispan", type=float, default=0.333, help="semispan [m]")
    parser.add_argument("--chord", type=float, default=0.140, help="Chord [m]")
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    SPAN = args.semispan
    CHORD = args.chord

    # ************************************************
    #     Parent FFD
    # ************************************************
    # ---------------------------
    #   Determine FFD box
    # ---------------------------
    # --- FFD box ---
    FFDbox = np.zeros((n_chord, n_span, 2, 3))  # [chord, span, thick, 3]

    # --- Margins ---
    zmargin = 10e-3
    ymargin = 3e-3
    xmargin = 3e-3

    # --- Loop to populate ---
    for jj in range(n_span):
        s_dist = np.linspace(0, SPAN, n_span)[jj]
        for ii in range(n_chord):
            c_dist = np.linspace(-CHORD * 0.5, CHORD * 0.5, n_chord)[ii]

            lower = np.array([c_dist, s_dist, -zmargin])
            upper = np.array([c_dist, s_dist, zmargin])

            if jj == 0:
                lower[1] -= ymargin
                upper[1] -= ymargin
            elif jj == n_span - 1:
                lower[1] += ymargin
                upper[1] += ymargin

            if ii == 0:
                lower[0] -= xmargin
                upper[0] -= xmargin
            elif ii == 1:
                lower[0] += xmargin
                upper[0] += xmargin

            FFDbox[ii, jj, 0, :] = lower
            FFDbox[ii, jj, 1, :] = upper

    # ---------------------------
    #   Write to file
    # ---------------------------
    ffdfile = open(f"{output_dir}/{args.foil}_ffd.xyz", "w")
    ffdfile.write("1\n")
    ffdfile.write(f"{n_chord} {n_span} 2\n")

    for coordDir in range(3):  # coordinate index 0->x, 1->y
        for kk in range(2):
            for jj in range(n_span):
                for ii in range(n_chord):
                    ffdfile.write("%.15f " % (FFDbox[ii, jj, kk, coordDir]))
                ffdfile.write("\n")

    ffdfile.close()
