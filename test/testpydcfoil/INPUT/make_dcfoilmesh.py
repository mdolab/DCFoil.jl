# --- Python 3.10 ---
"""
@File          :   make_dcfoilmesh.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   Write mesh for dcfoil
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


n_span = 30
xMidchord = 0.0
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
    comp_name = "foil"

    # ************************************************
    #     Write mesh file
    # ************************************************
    f = open(f"{args.foil}_{comp_name}_mesh.dcf", "w")
    fTecplot = open(f"{args.foil}_{comp_name}_mesh.dat", "w")
    fTecplot.write('TITLE = "dcfoil mesh"\n')
    fTecplot.write('VARIABLES = "x", "y", "z"\n')

    f.write(f"{comp_name}\n")
    # ---------------------------
    #   Leading edge
    # ---------------------------
    fTecplot.write('ZONE T="LEADING EDGES"\nDATAPACKING=POINT\n')
    f.write("LE\n")

    s_dist = np.linspace(0.0, SPAN, n_span)
    c_dist = np.linspace(args.rootChord, args.tipChord, n_span)
    # --- Loop to populate ---
    for jj in range(n_span):
        # s_dist = SPAN * (jj + 1) / n_span
        s_frac = s_dist[jj]
        c_frac = c_dist[jj]
        xLE = xMidchord - c_frac * 0.5
        f.write(f"{xLE:.8f} {s_frac:.8f} {0.0:.8f}\n")
        fTecplot.write(f"{xLE:.8f} {s_frac:.8f} {0.0:.8f}\n")

    # ---------------------------
    #   Trailing Edge
    # ---------------------------
    f.write("TE\n")
    fTecplot.write('ZONE T="TRAILING EDGES"\nDATAPACKING=POINT\n')

    # --- Loop to populate ---
    for jj in range(n_span):
        # s_dist = SPAN * (jj + 1) / n_span
        s_frac = s_dist[jj]
        c_frac = c_dist[jj]
        xTE = xMidchord + c_frac * 0.5
        f.write(f"{xTE:.8f} {s_frac:.8f} {0.0:.8f}\n")
        fTecplot.write(f"{xTE:.8f} {s_frac:.8f} {0.0:.8f}\n")

    f.close()
    fTecplot.close()
