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
from tabulate import tabulate

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
#                         FUNCTIONS
# ==============================================================================
def check_derivs(dvDict, funcSensAD, funcSensFD, dh):
    # now loop over the values and compare...
    for xs in dvDict:
        try:
            err = np.array(funcSensAD[xs].squeeze()) - np.array(funcSensFD[xs])
            print("Error:")
            print(err)
        except Exception:
            print(f"AD deriv for {xs} is", funcSensAD[xs])
            print(f"FD deriv for {xs} is", funcSensFD[xs])
            err = np.array(funcSensAD[xs]) - np.array(funcSensFD[xs])
            print("Error:")
            print(err)
            continue

        print(f"The FD step was {dh:.2e}")

        # get the L2 norm of error
        print("L2   error norm for DV %s is: " % xs, np.linalg.norm(err))

        # get the L_inf norm
        print("Linf error norm for DV %s is: " % xs, np.linalg.norm(err, ord=np.inf))


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="INPUT")
    parser.add_argument("--output", type=str, default="OUTPUT")
    parser.add_argument("--animate", default=False, action="store_true")
    parser.add_argument("--deriv", default=False, action="store_true")
    parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
    parser.add_argument(
        "--geovar",
        type=str,
        default="trwpd",
        help="Geometry variables to test twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
    )
    parser.add_argument("--is_dynamic", action="store_true", default=False)
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    files = {}
    # files["gridFile"] = f"{args.input}/{args.foil}_foil_mesh.dcf"
    files["gridFile"] = [
        f"./{args.input}/{args.foil}_foil_stbd_mesh.dcf",
        f"./{args.input}/{args.foil}_foil_port_mesh.dcf",
    ]
    files["FFDFile"] = f"{args.input}/{args.foil}_ffd.xyz"

    outputDir = f"{args.output}/embedding/"
    Path(outputDir).mkdir(exist_ok=True, parents=True)
    # ==============================================================================
    #                         DCFoil setup
    # ==============================================================================
    # V = 10 * 1.9438
    V = 18.0
    rho = 1025.0
    temp = 288.15
    mu = 1.22e-3  # dynamic viscosity [kg/m/s]

    evalFuncs = ["cl"]

    ap = AeroProblem(
        name="dcfoil",
        alpha=2.0,
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

    STICKSolver, solverOptions, _, _, _, _ = setup_dcfoil.setup(args, None, files, evalFuncs, outputDir, ap)

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
        # nTwist = nRefAxPts
        nTwist = nRefAxPts // 2
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
                # geo.rot_y[twistAxis].coef[ii] = val[ii]

                geo.rot_y[twistAxis].coef[nSkip - ii + nTwist - 1] = val[ii]
                geo.rot_y[twistAxis].coef[nTwist + ii + nSkip + 1] = val[ii]

        DVGeo.addGlobalDV(
            "twist",
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
        nSweep = nRefAxPts // 2

        print(f"{nSweep} foil sweep vars", flush=True)
        sweepAxis = "global"

        def sweep_rot_func(inval, geo):
            # REVERSE OF RH RULE FOR DCFOIL
            val = -inval

            # the extractCoef method gets the unperturbed ref axis control points
            C = geo.extractCoef(sweepAxis)
            C_orig = C.copy()
            # we will sweep the wing about the first point in the ref axis
            # sweep_ref_pt = C_orig[0, :]
            sweep_ref_pt = C_orig[nSweep, :]

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
            thetanew = val[0] * np.pi / 180
            ccnew = np.cos(thetanew)
            ssnew = np.sin(thetanew)
            rot_mtxnew = np.array(  # rotation matrix about the z-axis
                [
                    [ccnew, -ssnew, 0.0],
                    [ssnew, ccnew, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            # modify the control points of the ref axis
            # by applying a rotation about the first point in the x-z plane
            for ii in range(nSweep):
                # # get the vector from each ref axis point to the wing root
                # vec = C[ii + nSweep + 1, :] - sweep_ref_pt
                # # need to now rotate this by the sweep angle and add back the wing root loc
                # C[ii + nSweep + 1, :] = sweep_ref_pt + rot_mtx @ vec

                vec = C[-ii + nSweep - 1, :] - sweep_ref_pt
                C[-ii + nSweep - 1, :] = sweep_ref_pt + rot_mtx @ vec

                vec = C[nSweep + ii + 1, :] - sweep_ref_pt
                C[nSweep + ii + 1, :] = sweep_ref_pt + rot_mtxnew @ vec

            # use the restoreCoef method to put the control points back in the right place
            geo.restoreCoef(C, sweepAxis)

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

        nTaper = nRefAxPts // 2 + 1
        # def chords(val, geo):
        #     # Set all the global chord values
        #     for ii in range(nSkip, nRefAxPts):
        #         geo.scale["global"].coef[ii] = val[ii - nSkip]

        # DVGeo.addGlobalDV(
        #     "chord",
        #     value=np.ones(nRefAxPts - nSkip) * 0.1,
        #     func=chords,
        #     lower=0.01,
        #     upper=0.2,
        #     scale=1.0,
        # )

        def taper(val, geo):
            s = geo.extractS("global")
            # slope = (val[1] - val[0]) / (s[-1] - s[0])
            slope = (val[1] - val[0]) / (s[-1] - s[nTaper - 1])
            for ii in range(nTaper):
                # geo.scale_x["global"].coef[ii] = slope * (s[ii] - s[0]) + val[0]

                geo.scale_x["global"].coef[ii + nTaper - 1] = slope * (s[ii + nTaper - 1] - s[nTaper - 1]) + val[0]
                geo.scale_x["global"].coef[nTaper - ii - 1] = -slope * (s[nTaper - ii - 1] - s[nTaper - 1]) + val[0]

        DVGeo.addGlobalDV(
            "taper",
            value=np.ones(2) * 1.0,
            func=taper,
            lower=[0.1, 0.1],
            upper=[1.5, 1.5],
            scale=1.0,
        )

    # ---------------------------
    #   Span
    # ---------------------------
    if "p" in args.geovar:
        # nSpan = nRefAxPts
        nSpan = nRefAxPts // 2 + 1

        def span(val, geo):
            C = geo.extractCoef("global")
            s = geo.extractS("global")
            for ii in range(1, nSpan):
                # C[ii, 1] += val * s[ii]

                C[ii + nSpan - 1, 1] += val.item() * s[ii + nSpan - 1]
                C[nSpan - ii - 1, 1] += -val.item() * s[ii + nSpan - 1]
            geo.restoreCoef(C, "global")

        DVGeo.addGlobalDV(dvName="span", value=0.0, func=span, lower=-10.0, upper=20.0, scale=0.1)

    # ************************************************
    #     Double check the embedding
    # ************************************************
    DVGeo.writeTecplot(fileName=f"./{outputDir}/dvgeo.dat", solutionTime=1)

    # ==============================================================================
    #                         Set it for solver
    # ==============================================================================
    # STICKSolver.setDVGeo(DVGeo, debug=True)
    STICKSolver.setDVGeo(DVGeo)
    STICKSolver.addMesh(solverOptions["gridFile"])

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
            n_twist = 20  # number of increments

            mag = 15.0
            wave = mag * np.sin(np.linspace(0, 2 * np.pi, n_twist))

            i_frame = 0
            ap.callCounter = 0

            # twist_vals = np.array(
            #     [
            #         [00.0, 30.0, 30.0, 30.0, 30.0, 30.0],
            #         [30.0, 0.0, -30.0, 0.0, 30.0, 30.0],
            #         [0.0, 10.0, -30.0, 0.0, 30.0, 30.0],
            #         [10.0, 0.0, -30.0, 0.0, 30.0, 30.0],
            #         [-30.0, 0.0, 30.0, 0.0, -30.0, 30.0],
            #         # [0.0, 30.0],
            #         # [10.0, 30.0],
            #         # [20.0, 30.0],
            #     ]
            # )
            # for ii in range(1, nTwist):
            for ii in range(nTwist):
                # for ind, val in enumerate(twist_vals):
                for val in wave:
                    print(val)
                    dvDict = DVGeo.getValues()
                    dvDict["twist"][ii] = val
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
            wave = np.sin(np.linspace(0, np.pi, n_span)) * 0.1

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

            STICKSolver.setOption("outputDir", dirName)

            n_sweep = 30
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
            wave = np.sin(np.linspace(0, 2 * np.pi, n_taper)) * 0.5 + 1.0

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

        # ---------------------------
        #   All vars
        # ---------------------------
        if "t" and "r" and "w" and "p" in args.geovar:
            dirName = f"{outputDir}/demo_all/"
            print("+", 20 * "-", "Demo all", 20 * "-", "+")
            Path(dirName).mkdir(exist_ok=True, parents=True)

            STICKSolver.setOption("outputDir", dirName)

            n_all = 20
            wave = np.sin(np.linspace(0, 2 * np.pi, n_all))

            taper_mag = 0.5
            twist_mag = 15.0
            sweep_mag = 30.0
            span_mag = 0.1

            i_frame = 0

            # --- TAPER ---
            for ind, val in enumerate(wave):
                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["taper"][-1] = val * taper_mag + 1.0
                DVGeo.setDesignVars(dvDict)

                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="all")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                i_frame += 1
            DVGeo.setDesignVars(dvDict_base)

            # --- TWIST ---
            for ii in range(1, nTwist):
                for ind, val in enumerate(wave):
                    print(ind, val)
                    dvDict = DVGeo.getValues()
                    dvDict["twist"][ii] = val * twist_mag
                    DVGeo.setDesignVars(dvDict)

                    STICKSolver.setAeroProblem(ap)
                    STICKSolver.writeSolution(number=i_frame, baseName="all")

                    # Write deformed FFD
                    DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                    DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                    i_frame += 1

            DVGeo.setDesignVars(dvDict_base)

            # --- SWEEP ---
            for ind, val in enumerate(wave):
                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["sweep"] = val * sweep_mag
                DVGeo.setDesignVars(dvDict)

                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="all")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                i_frame += 1

            DVGeo.setDesignVars(dvDict_base)

            # --- SPAN ---
            wave = np.sin(np.linspace(0, np.pi, n_all))
            for ind, val in enumerate(wave):
                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["span"] = val * span_mag
                DVGeo.setDesignVars(dvDict)

                STICKSolver.setAeroProblem(ap)
                STICKSolver.writeSolution(number=i_frame, baseName="all")

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                i_frame += 1

            DVGeo.setDesignVars(dvDict_base)

    if args.deriv:
        print("Testing derivatives...")

        evalFuncs = ["lift", "cl", "wtip", "cd", "kscl"]
        # , "ksflutter"]

        stepsizes = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
        # DH = 1e-4

        funcSensAdj = {}
        funcSensFD = {}

        finalFDDict = {}

        funcs = {}
        funcsFD = {}

        DVGeo.setDesignVars(dvDict_base)
        ap.setDesignVars(dvDict_base)

        # Solve
        STICKSolver(ap)
        # Analytic sensitivities
        STICKSolver.evalFunctions(ap, funcs, evalFuncs=evalFuncs)
        print("Analytic funcs", funcs)
        STICKSolver.evalFunctionsSens(ap, funcSensAdj, evalFuncs=evalFuncs)
        print("Analytic funcs sens", funcSensAdj)

        # ---------------------------
        #   Finite difference
        # ---------------------------
        for DH in stepsizes:
            if "w" in args.geovar:
                funcSensFD["sweep"] = {}

                dvDict = DVGeo.getValues()
                dvDict["sweep"] += DH
                DVGeo.setDesignVars(dvDict)
                ap.setDesignVars(dvDict)

                STICKSolver(ap)
                STICKSolver.evalFunctions(ap, funcsFD, evalFuncs=evalFuncs)

                dvDict["sweep"] -= DH
                DVGeo.setDesignVars(dvDict)
                ap.setDesignVars(dvDict)

                for evalFunc in evalFuncs:
                    funcSensFD["sweep"][evalFunc] = np.divide(
                        funcsFD[f"{ap.name}_{evalFunc}"] - funcs[f"{ap.name}_{evalFunc}"], DH
                    )

            if "t" in args.geovar:
                funcSensFD["twist"] = np.zeros(nTwist)

                dvDict = DVGeo.getValues()
                dvDict["twist"] += np.ones(nTwist) * DH
                DVGeo.setDesignVars(dvDict)
                ap.setDesignVars(dvDict)

                STICKSolver(ap)
                STICKSolver.evalFunctions(ap, funcsFD, evalFuncs=evalFuncs)

                dvDict["twist"] -= np.ones(nTwist) * DH
                DVGeo.setDesignVars(dvDict)
                ap.setDesignVars(dvDict)

                for evalFunc in evalFuncs:
                    funcSensFD["twist"][evalFunc] = np.divide(
                        funcsFD[f"{ap.name}_{evalFunc}"] - funcs[f"{ap.name}_{evalFunc}"], DH
                    )

            if "r" in args.geovar:
                funcSensFD["taper"] = {}

                for evalFunc in evalFuncs:
                    funcSensFD["taper"][evalFunc] = np.zeros(2)

                for ii in range(2):
                    dvDict = DVGeo.getValues()
                    dvDict["taper"][ii] += DH
                    DVGeo.setDesignVars(dvDict)
                    ap.setDesignVars(dvDict)

                    STICKSolver(ap)
                    STICKSolver.evalFunctions(ap, funcsFD, evalFuncs=evalFuncs)

                    dvDict["taper"][ii] -= DH
                    DVGeo.setDesignVars(dvDict)
                    ap.setDesignVars(dvDict)

                    for evalFunc in evalFuncs:
                        funcSensFD["taper"][evalFunc][ii] = np.divide(
                            funcsFD[f"{ap.name}_{evalFunc}"] - funcs[f"{ap.name}_{evalFunc}"], DH
                        )

            if "p" in args.geovar:
                funcSensFD["span"] = {}

                dvDict = DVGeo.getValues()
                dvDict["span"] += DH
                DVGeo.setDesignVars(dvDict)
                ap.setDesignVars(dvDict)

                STICKSolver(ap)
                STICKSolver.evalFunctions(ap, funcsFD, evalFuncs=evalFuncs)

                dvDict["span"] -= DH
                DVGeo.setDesignVars(dvDict)
                ap.setDesignVars(dvDict)

                for evalFunc in evalFuncs:
                    funcSensFD["span"][evalFunc] = np.divide(
                        funcsFD[f"{ap.name}_{evalFunc}"] - funcs[f"{ap.name}_{evalFunc}"], DH
                    )

            # Angle of attack derivative
            funcSensFD["aoa"] = {}
            ap.alpha += DH
            STICKSolverNew, _, _, _, _, _ = setup_dcfoil.setup(args, None, files, evalFuncs, outputDir, ap)
            STICKSolverNew.setDVGeo(DVGeo)
            STICKSolverNew.addMesh(solverOptions["gridFile"])
            STICKSolverNew(ap)
            STICKSolverNew.evalFunctions(ap, funcsFD, evalFuncs=evalFuncs)
            ap.alpha -= DH

            for evalFunc in evalFuncs:
                funcSensFD["aoa"][evalFunc] = np.divide(
                    funcsFD[f"{ap.name}_{evalFunc}"] - funcs[f"{ap.name}_{evalFunc}"], DH
                )

            print(20 * "=")
            print(f"FD funcsens dh = {DH}")
            print(pp(funcSensFD))

            finalFDDict[f"{DH}"] = {}
            for evalFunc in evalFuncs:
                finalFDDict[f"{DH}"][evalFunc] = {}
                for dvkey, v in dvDict.items():
                    finalFDDict[f"{DH}"][evalFunc][dvkey] = copy.deepcopy(funcSensFD[dvkey][evalFunc])

                finalFDDict[f"{DH}"][evalFunc]["aoa"] = copy.deepcopy(funcSensFD["aoa"][evalFunc])

        print(20 * "=")
        print("Analytic funcs:")
        print(pp(funcs))
        print(20 * "=")
        print("Analytic funcsSens:")
        headers = ["Adjoint"] + stepsizes
        print(pp(funcSensAdj))
        for dhkey, v in finalFDDict.items():
            print(f"FD funcsSens dh = {dhkey}")
            print(pp(v))

        # --- Pretty print stuff ---
        for DV, value in dvDict.items():
            print(20 * "=")
            print(f"Checking DV: {DV}")
            print(20 * "=")
            for evalFunc in evalFuncs:
                print(f"{evalFunc} sens:")
                adjoint = funcSensAdj[f"dcfoil_{evalFunc}"][DV]

                fdvals = []
                for dh in stepsizes:
                    fdvals.append(finalFDDict[f"{dh}"][evalFunc][DV])

                print(
                    f" Adjoint |\tdh={stepsizes[0]:.1e} \tdh={stepsizes[1]:.1e}\tdh={stepsizes[2]:.1e}\tdh={stepsizes[3]:.1e}\tdh={stepsizes[4]:.1e}"
                )
                print(f"{adjoint}\t{fdvals[0]}\t{fdvals[1]}\t{fdvals[2]}\t{fdvals[3]}\t{fdvals[4]}")

                # row = [funcSensAdj[f"dcfoil_{evalFunc}"]["span"], finalFDDict[stepsizes[0]]["span"][evalFunc],finalFDDict[stepsizes[1]]["span"][evalFunc],finalFDDict[stepsizes[2]]["span"][evalFunc],finalFDDict[stepsizes[3]]["span"][evalFunc],finalFDDict[stepsizes[4]]["span"][evalFunc]]
                # tabulate(row, headers=headers)

        print(20 * "=")
        print(f"Checking DV: aoa")
        print(20 * "=")
        for evalFunc in evalFuncs:
            print(f"{evalFunc} sens:")

            fdvals = []
            for dh in stepsizes:
                fdvals.append(finalFDDict[f"{dh}"][evalFunc]["aoa"])

            print(
                f"\tdh={stepsizes[0]:.1e} \tdh={stepsizes[1]:.1e}\tdh={stepsizes[2]:.1e}\tdh={stepsizes[3]:.1e}\tdh={stepsizes[4]:.1e}"
            )
            print(f"{fdvals[0]}\t{fdvals[1]}\t{fdvals[2]}\t{fdvals[3]}\t{fdvals[4]}")
