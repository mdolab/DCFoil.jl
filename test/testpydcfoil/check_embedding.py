# --- Python 3.10 ---
"""
@File          :   check_embedding.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   Play with FFDs
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import json
import argparse
import copy
from pathlib import Path
from pprint import pprint as pp

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
from baseclasses import AeroProblem
from pygeo import DVGeometry
# from pyspline.utils import (
#     openTecplot,
#     writeTecplot1D,
#     writeTecplot2D,
#     writeTecplot3D,
#     closeTecplot,
# )
from SETUP import setup_dcfoil

# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="INPUT")
    parser.add_argument("--output", type=str, default="OUTPUT")
    parser.add_argument("--animate", default=False, action="store_true")
    parser.add_argument("--deriv", default=False, action="store_true")
    parser.add_argument(
        "--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat"
    )
    parser.add_argument(
        "--geovar",
        type=str,
        default="trwpd",
        help="Geometry variables to test twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
    )
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    files = {}
    files["gridFile"] = f"{args.input}/{args.foil}_mesh.dcf"
    files["FFDFile"] = f"{args.input}/{args.foil}_ffd.xyz"

    outputDir = f"{args.output}/embedding/"
    Path(outputDir).mkdir(exist_ok=True, parents=True)
    # ==============================================================================
    #                         DCFoil setup
    # ==============================================================================
    V = 10 * 1.9438
    rho = 1025.0
    temp = 288.15
    mu = 1.22e-3  # dynamic viscosity [kg/m/s]

    evalFuncs = ["cl"]

    ap = AeroProblem(
        name="dcfoil",
        alpha=0.0,
        V=V,
        rho=rho,
        T=temp,
        areaRef=1.0,
        chordRef=1.0,
        xRef=0.25,
        evalFuncs=evalFuncs,
        muSuthDim=mu,
        TSuthDim=temp,
    )

    STICKSolver = setup_dcfoil.setup(args, None, files, evalFuncs)

    # ==============================================================================
    #                         DVGeometry setup
    # ==============================================================================
    DVGeo = DVGeometry(files["FFDFile"], kmax=4)

    nRefAxPts = DVGeo.addRefAxis("global", xFraction=0.5, alignIndex="j")

    # ==============================================================================
    #                         DVS
    # ==============================================================================
    # ---------------------------
    #   TWIST
    # ---------------------------
    if "t" in args.geovar:
        # if comm.rank == 0:
        nTwist = nRefAxPts
        print(f"{nTwist} foil twist vars", flush=True)

        twistAxis = "global"

        def twist_rottheta_func(val, geo):
            """
            val array has length of semi-span FFDs only. It's mirrored to the full config
            rottheta method is but the curve
            """

            for ii in range(nTwist):
                geo.rot_theta["c4_v0"].coef[ii + n_skip] = val[ii]

                # if args.config in ["full", "fs"]:
                #     geo.rot_theta["c4_v1"].coef[ii + n_skip] = val[ii]

        def twist_roty_func(val, geo):
            """
            val array has length of semi-span FFDs only. It's mirrored to the full config
            """
            nSkip = 0
            for ii in range(nTwist):
                geo.rot_y[twistAxis].coef[ii + nSkip] = val[ii]

                # if args.config in ["full", "fs"]:
                #     geo.rot_y["c4_v1"].coef[ii + nSkip] = val[ii]

        DVGeo.addGlobalDV(
            "foil_twist",
            value=np.zeros(nTwist),
            func=twist_roty_func,
            # func=twist_rottheta_func,
            lower=-15.0,
            upper=15.0,
            scale=1.0,
        )

    # ---------------------------
    #   SWEEP
    # ---------------------------
    if "w" in args.geovar:
        # Determine the number of sections that have sweep control
        nSweep = nRefAxPts
        sweepAxis = "global"

        def sweep_rot_func(val, geo):
            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef(sweepAxis)
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            sweep_ref_pt = C_orig[0, :]

            theta = -val[0] * np.pi / 180
            cc = np.cos(theta)
            ss = np.sin(theta)
            rot_mtx = np.array(  # rotation matrix about the z-axis
                [
                    [cc, -ss, 0.0],
                    [ss, cc, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            # modify the control points of the ref axis
            # by applying a rotation about the first point in the x-z plane
            for ii in range(nSweep):
                # get the vector from each ref axis point to the wing root
                vec = C[ii, :] - sweep_ref_pt
                # need to now rotate this by the sweep angle and add back the wing root loc
                C[ii, :] = sweep_ref_pt + rot_mtx @ vec

            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, sweepAxis)

            # if args.config in ["full", "fs"]:
            #     C2 = geo.extractCoef("c4_v1")
            #     C_orig = C2.copy()
            #     # we will sweep the wing about the first point in the ref axis
            #     sweep_ref_pt = C_orig[0, :]

            #     theta = val[0] * np.pi / 180
            #     cc = np.cos(theta)
            #     ss = np.sin(theta)

            #     for ii in range(nRefAxPts2):
            #         # get the vector from each ref axis point to the wing root
            #         vec = C2[ii, :] - sweep_ref_pt
            #         # need to now rotate this by the sweep angle and add back the wing root loc
            #         C2[ii, :] = sweep_ref_pt + rot_mtx @ vec

            #     # use the restoreCoef method to put the control points back in the right place
            #     geo.restoreCoef(C2, "c4_v1")

        DVGeo.addGlobalDV(
            dvName="sweep",
            value=0.0,
            func=sweep_rot_func,
            lower=-5.0,
            upper=5.0,
            scale=1,
        )

    # ---------------------------
    #   DIHEDRAL
    # ---------------------------
    if "d" in args.geovar:
        nDihedral = nRefAxPts
        dihedralAxis = "global"

        def dihedral_rot_func(val, geo):
            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef(dihedralAxis)
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            sweep_ref_pt = C_orig[0, :]

            theta = -val[0] * np.pi / 180
            cc = np.cos(theta)
            ss = np.sin(theta)
            rot_mtx = np.array(  # rotation matrix about the z-axis
                [
                    [1, 0.0, 0.0],
                    [0.0, cc, -ss],
                    [0.0, ss, cc],
                ]
            )

            # modify the control points of the ref axis
            # by applying a rotation about the first point in the x-z plane
            for ii in range(nDihedral):
                # get the vector from each ref axis point to the wing root
                vec = C[ii, :] - sweep_ref_pt
                # need to now rotate this by the sweep angle and add back the wing root loc
                C[ii, :] = sweep_ref_pt + rot_mtx @ vec
            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, dihedralAxis)

            # if args.config in ["full", "fs"]:
            #     C2 = geo.extractCoef("c4_v1")
            #     C_orig = C2.copy()
            #     # we will sweep the wing about the first point in the ref axis
            #     sweep_ref_pt = C_orig[0, :]

            #     theta = val[0] * np.pi / 180
            #     cc = np.cos(theta)
            #     ss = np.sin(theta)
            #     rot_mtx = np.array(  # rotation matrix about the z-axis
            #         [
            #             [1, 0.0, 0.0],
            #             [0.0, cc, -ss],
            #             [0.0, ss, cc],
            #         ]
            #     )

            #     # modify the control points of the ref axis
            #     # by applying a rotation about the first point in the x-z plane
            #     for ii in range(nRefAxPts2):
            #         # get the vector from each ref axis point to the wing root
            #         vec = C2[ii, :] - sweep_ref_pt
            #         # need to now rotate this by the sweep angle and add back the wing root loc
            #         C2[ii, :] = sweep_ref_pt + rot_mtx @ vec
            #     # use the restoreCoef method to put the control points back in the right place
            #     geo.restoreCoef(C2, "c4_v1")

        DVGeo.addGlobalDV(
            dvName="dihedral",
            value=0.0,
            func=dihedral_rot_func,
            lower=-5.0,
            upper=5.0,
            scale=1,
        )

    # ---------------------------
    #   CHORD
    # ---------------------------
    if "r" in args.geovar:

        nSkip = 4

        def chords(val, geo):
            # Set all the global chord values
            for ii in range(nSkip, nRefAxPts):
                geo.scale["global"].coef[ii] = val[ii - nSkip]

        DVGeo.addGlobalDV(
            "chord",
            value=np.ones(nRefAxPts - nSkip) * 0.1,
            func=chords,
            lower=0.01,
            upper=0.2,
            scale=1.0,
        )

        def taper(val, geo):
            s = geo.extractS("global")
            slope = (val[1] - val[0]) / (s[-1] - s[0])
            for ii in range(nRefAxPts):
                geo.scale_x["global"].coef[ii] = slope * (s[ii] - s[0]) + val[0]

        DVGeo.addGlobalDV(
            "taper",
            value=[0.1, 0.1],
            func=taper,
            lower=[0.01, 0.01],
            upper=[0.2, 0.2],
            scale=1.0,
        )

    # ---------------------------
    #   Span
    # ---------------------------
    if "p" in args.geovar:

        def span(val, geo):
            C = geo.extractCoef("global")
            s = geo.extractS("global")
            for ii in range(1, nRefAxPts):
                C[ii, 2] += val * s[ii]
            geo.restoreCoef(C, "global")

        DVGeo.addGlobalDV(
            dvName="span", value=0.0, func=span, lower=-10.0, upper=20.0, scale=0.1
        )

    # ************************************************
    #     Double check the embedding
    # ************************************************
    DVGeo.writeTecplot(fileName=f"./{outputDir}/dvgeo.dat", solutionTime=1)

    # ==============================================================================
    #                         Set it for solver
    # ==============================================================================
    STICKSolver.setDVGeo(DVGeo)

    # ==============================================================================
    #                         Do deformations
    # ==============================================================================
    dvDict = DVGeo.getValues()
    dvDict_base = copy.deepcopy(dvDict)

    print("Current DV dict:")
    pp(dvDict)

    DVGeo.setDesignVars(dvDict)
    ap.setDesignVars(dvDict)

    if args.animate:
        # ---------------------------
        #   TWIST
        # ---------------------------
        if "t" in args.geovar:
            print("+", 20 * "-", "Demo twist", 20 * "-", "+")
            dirName = f"{outputDir}/demo_twist/"
            # if comm.rank == 0:
            Path(dirName).mkdir(exist_ok=True, parents=True)
            STICKSolver.setOption("outputDir", dirName)
            n_twist = 40  # number of increments

            wave = np.sin(np.linspace(0, 2 * np.pi, n_twist)) * 15.0

            i_frame = 0
            ap.callCounter = 0

            twist_vals = np.array(
                [
                    [00.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                    [30.0, 0.0, -30.0, 0.0, 30.0, 30.0],
                    [0.0, 10.0, -30.0, 0.0, 30.0, 30.0],
                    [10.0, 0.0, -30.0, 0.0, 30.0, 30.0],
                    [-30.0, 0.0, 30.0, 0.0, -30.0, 30.0],
                    # [0.0, 30.0],
                    # [10.0, 30.0],
                    # [20.0, 30.0],
                ]
            )


            for ind, val in enumerate(twist_vals):

                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["twist"] = val
                DVGeo.setDesignVars(dvDict)

                # ap.name = "twist"
                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="twist")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/twist_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/twist_{i_frame:03d}_axes")

                i_frame += 1
            DVGeo.setDesignVars(dvDict_base)

        # ---------------------------
        #   SPAN
        # ---------------------------
        if "p" in args.geovar:
            print("+", 20 * "-", "Demo span", 20 * "-", "+")
            dirName = f"{outputDir}/demo_span/"
            # if comm.rank == 0:
            Path(dirName).mkdir(exist_ok=True, parents=True)
            STICKSolver.setOption("outputDir", dirName)

            n_span = 60
            wave = np.sin(np.linspace(0, 2 * np.pi, n_span)) * 0.4

            i_frame = 0
            # loop over wave
            for ind, val in enumerate(wave):
                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["span"] = val
                DVGeo.setDesignVars(dvDict)

                # ap.name = "span"
                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="span")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/span_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/span_{i_frame:03d}_axes")

                i_frame += 1
            DVGeo.setDesignVars(dvDict_base)

        # ---------------------------
        #   SWEEP
        # ---------------------------
        if "w" in args.geovar:
            dirName = f"{outputDir}/demo_sweep/"
            # if comm.rank == 0:
            print("+", 20 * "-", "Demo sweep", 20 * "-", "+")
            Path(dirName).mkdir(exist_ok=True, parents=True)

            STICKSolver.setOption("outputDirectory", dirName)

            n_sweep = 60
            mag = 30.0
            wave = mag * np.sin(np.linspace(0, 2 * np.pi, n_sweep))

            i_frame = 0

            # sweep_vals = np.array(
            #     [
            #         [0.0, 0.04],
            #         [0.01, 0.04],
            #         [0.02, 0.04],
            #     ]
            # )
            # sweep_vals = [0.0, 0.01, 0.02, 0.04, 0.08, 0.1]
            sweep_vals = wave

            for ind, val in enumerate(sweep_vals):

                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["sweep"] = val
                DVGeo.setDesignVars(dvDict)

                # ap.name = "sweep"
                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="sweep")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/sweep_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/sweep_{i_frame:03d}_axes")

                i_frame += 1

            DVGeo.setDesignVars(dvDict_base)

        # ---------------------------
        #   TAPER
        # ---------------------------
        if "r" in args.geovar:
            dirName = f"{outputDir}/demo_taper/"
            # if comm.rank == 0:
            print("+", 20 * "-", "Demo taper", 20 * "-", "+")
            Path(dirName).mkdir(exist_ok=True, parents=True)

            STICKSolver.setOption("outputDirectory", dirName)

            n_taper = 60
            wave = np.sin(np.linspace(0, 2 * np.pi, n_taper)) * 0.6

            i_frame = 0
            # loop over wave
            for ind, val in enumerate(wave):
                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["taper"][-1] = val
                DVGeo.setDesignVars(dvDict)

                # ap.name = "span"
                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="taper")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/taper_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/taper_{i_frame:03d}_axes")

                i_frame += 1
            DVGeo.setDesignVars(dvDict_base)
