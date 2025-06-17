# --- Python 3.10 ---
"""
@File          :   generate_ffd.py
@Date created  :   2024/10/14
@Last modified :   2025/06/04
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
n_buffer = 2
# n_span = 5 + n_buffer # add 2 to root
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
    parser.add_argument("--rootChord", type=float, default=0.140, help="Root chord [m]")
    parser.add_argument("--tipChord", type=float, default=0.095, help="Tip chord [m]")
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    SPAN = args.semispan

    # ************************************************
    #     Parent FFD
    # ************************************************
    # ---------------------------
    #   Determine FFD box
    # ---------------------------
    # NSPANTOT = 2 * n_span - 1 + 2
    NSPANTOT = 2 * (n_span + n_buffer) - 1
    # --- FFD box ---
    FFDbox = np.zeros((n_chord, n_span, 2, 3))  # [chord, span, thick, 3]
    FFDbox = np.zeros((n_chord, 2 * (n_span + n_buffer) - 1, 2, 3))  # [chord, 2*span -1, thick, 3]
    # FFDbox = np.zeros((n_chord, NSPANTOT, 2, 3))  # [chord, 2*span, thick, 3]

    # --- Margins ---
    zmargin = 10e-3
    ymargin = 3e-3
    xmargin = 3e-3

    s_dist = np.linspace(0.0, SPAN, n_span)
    c_dist = np.linspace(args.rootChord, args.tipChord, n_span)

    # --- Loop to populate ---
    for jj in range(1, n_span):
        s_frac = -s_dist[-1 - jj + 1]
        for ii in range(n_chord):
            c_frac = np.linspace(-c_dist[-jj] * 0.5, c_dist[-jj] * 0.5, n_chord)[ii]

            lower = np.array([c_frac, s_frac, -zmargin])
            upper = np.array([c_frac, s_frac, zmargin])

            if jj == 1:
                lower[1] -= ymargin
                upper[1] -= ymargin

            if ii == 0:
                lower[0] -= xmargin
                upper[0] -= xmargin
            elif ii == 1:
                lower[0] += xmargin
                upper[0] += xmargin
            print(lower)

            FFDbox[ii, jj - 1, 0, :] = lower
            FFDbox[ii, jj - 1, 1, :] = upper

    # STBD
    for jj in range(n_span):
        s_frac = s_dist[jj]
        for ii in range(n_chord):
            c_frac = np.linspace(-c_dist[jj] * 0.5, c_dist[jj] * 0.5, n_chord)[ii]

            lower = np.array([c_frac, s_frac, -zmargin])
            upper = np.array([c_frac, s_frac, zmargin])

            # if jj == 0:
            #     lower[1] -= ymargin
            #     upper[1] -= ymargin
            #     FFDbox[ii, jj + n_span - 1, 0, :] = lower
            #     FFDbox[ii, jj + n_span - 1, 1, :] = upper
            #     print(lower)
            #     lower[1] += 2 * ymargin
            #     upper[1] += 2 * ymargin
            #     FFDbox[ii, jj + n_span + 1, 0, :] = lower
            #     FFDbox[ii, jj + n_span + 1, 1, :] = upper
            #     print(lower)
            #     # RESET
            #     lower[1] -= ymargin
            #     upper[1] -= ymargin

            #     lower[1] -= ymargin
            #     upper[1] -= ymargin
            if jj == n_span - 1:
                lower[1] += ymargin
                upper[1] += ymargin

            if ii == 0:
                lower[0] -= xmargin
                upper[0] -= xmargin
            elif ii == 1:
                lower[0] += xmargin
                upper[0] += xmargin

            print(lower)
            FFDbox[ii, jj + n_span + 2 * n_buffer - 1, 0, :] = lower
            FFDbox[ii, jj + n_span + 2 * n_buffer - 1, 1, :] = upper
            # FFDbox[ii, jj + n_span, 0, :] = lower
            # FFDbox[ii, jj + n_span, 1, :] = upper

    # --- Fill in buffer! ---
    spaces = np.linspace(-0.05 * SPAN, 0.05 * SPAN, len(np.arange(n_span - 1, n_span + n_buffer * 2)))
    for inum, jj in enumerate(np.arange(n_span - 1, n_span + n_buffer * 2)):
        FFDbox[0, jj, 0, :] = np.array([-args.rootChord * 0.5, spaces[inum], -zmargin])
        FFDbox[0, jj, 1, :] = np.array([-args.rootChord * 0.5, spaces[inum], zmargin])
        FFDbox[1, jj, 0, :] = np.array([args.rootChord * 0.5, spaces[inum], -zmargin])
        FFDbox[1, jj, 1, :] = np.array([args.rootChord * 0.5, spaces[inum], zmargin])

    # ---------------------------
    #   Write to file
    # ---------------------------
    ffdfile = open(f"{output_dir}/{args.foil}_ffd.xyz", "w")
    ffdfile.write("1\n")
    ffdfile.write(f"{n_chord} {NSPANTOT} 2\n")

    for coordDir in range(3):  # coordinate index 0->x, 1->y
        for kk in range(2):
            for jj in range(NSPANTOT):
                for ii in range(n_chord):
                    ffdfile.write("%.15f " % (FFDbox[ii, jj, kk, coordDir]))
                ffdfile.write("\n")

    ffdfile.close()

    # # ************************************************
    # #     Port FFD
    # # ************************************************
    # FFDbox[:,:,:,1] *= -1 # flip y
    # ffdfile = open(f"{output_dir}/{args.foil}_port_ffd.xyz", "w")
    # ffdfile.write("1\n")
    # ffdfile.write(f"{n_chord} {n_span} 2\n")

    # for coordDir in range(3):  # coordinate index 0->x, 1->y
    #     for kk in range(2):
    #         for jj in range(n_span):
    #             for ii in range(n_chord):
    #                 ffdfile.write("%.15f " % (FFDbox[ii, jj, kk, coordDir]))
    #             ffdfile.write("\n")

    # ffdfile.close()
