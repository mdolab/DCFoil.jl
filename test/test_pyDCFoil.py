# --- Python 3.10 ---
"""
@File    :   test_pyDCFoil.py
@Time    :   2024/02/20
@Author  :   Galen Ng
@Desc    :   Test script for pyDCFoil optimization
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys
import json
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
# import niceplots
from pprint import pprint as pp
from pyoptsparse import Optimization, OPT, History
from dcfoil import DCFOIL  # make sure to pip install this code


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", "-o", help="type of optimizer", choices=["SLSQP", "SNOPT"], type=str, default="SNOPT"
    )
    args = parser.parse_args()
    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}:{getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    # ************************************************
    #     Create dynamic beam solver
    # ************************************************
    nNodes = 3
    nNodesStrut = 3
    mainFoilOptions = {
        "compName": "mainFoil",
        "config": "t-foil",
        "nNodes": nNodes,
        "nNodeStrut": nNodesStrut,
        "xMount": 3.355,
        "material": "cfrp",
        "strut_material": "cfrp",
    }
    appendageList = [mainFoilOptions]
    solverOptions = {
        # ---------------------------
        #   I/O
        # ---------------------------
        "name": "test",
        "debug": False,
        "writeTecplotSolution": True,
        "outputDir": "./OUTPUT/",
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "appendageList": appendageList,
        "gravityVector": [0.0, 0.0, -9.81],
        # ---------------------------
        #   Flow
        # ---------------------------
        "U∞": 5.0,  # free stream velocity [m/s]
        "ρ_f": 1000.0,  # fluid density [kg/m³]
        "use_freeSurface": False,
        "use_cavitation": False,
        "use_ventilation": False,
        "use_dwCorrection": False,
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static": True,
        "res_jacobian": "cs",
        # --- Forced solve ---
        "run_forced": True,
        "run_forced": False,
        "fRange": [0.1, 1000.0],
        "tipForceMag": 1.0,
        "run_body": False,
        # --- p-k (Eigen) solve ---
        "run_modal": False,
        "run_flutter": False,
        # "run_modal": True,
        # "run_flutter": True,
        "nModes": 4,
        "uRange": [170.0, 190.0],
        "maxQIter": 100,  # that didn't fix the slow run time...
        "rhoKS": 80.0,
    }
    DVDict = {  # THIS IS BASED OFF OF THE MOTH RUDDER
        "alfa0": 6.0,  # initial angle of attack [deg]
        "sweep": np.deg2rad(0.0),  # sweep angle [rad]
        "zeta": 0.04,  # modal damping ratio at first 2 modes
        "c": 0.1 * np.ones(nNodes),  # chord length [m]
        "s": 0.3,  # semispan [m]
        "ab": 0 * np.ones(nNodes),  # dist from midchord to EA [m]
        "toc": 0.12 * np.ones(nNodes),  # thickness-to-chord ratio
        "x_ab": 0 * np.ones(nNodes),  # static imbalance [m]
        "theta_f": np.deg2rad(15),  # fiber angle global [rad]
        # --- Strut vars ---
        "depth0": 0.5,  # submerged depth of strut [m] # from Yingqian
        "rake": 0.0,  # rake angle about top of strut [deg]
        "beta": 0.0,  # yaw angle wrt flow [deg]
        "s_strut": 1.0,  # [m]
        "c_strut": 0.14 * np.ones(nNodesStrut),  # chord length [m]
        "toc_strut": 0.095 * np.ones(nNodesStrut),  # thickness-to-chord ratio (mean)
        "ab_strut": 0.0 * np.ones(nNodesStrut),  # dist from midchord to EA [m]
        "x_ab_strut": 0.0 * np.ones(nNodesStrut),  # static imbalance [m]
        "theta_f_strut": np.deg2rad(0),  # fiber angle global [rad]
    }
    evalFuncs = [
        "wtip",
        "psitip",
        "cl",
        "cmy",
        "lift",
        "moment",
        "cdi",
        "cdj",
        "cdpr",
        "cds",
        # "ksflutter",
    ]

    debug = True
    STICKSolver = DCFOIL(DVDictList=[DVDict], evalFuncs=evalFuncs, options=solverOptions, debug=debug)

    # ************************************************
    #     Functions
    # ************************************************
    def cruiseFuncs(x):
        # FLUTTER FUNCS AT SPEED ENVELOPE BOUND ('flutterFuncs(x)' instead?)
        funcs = {}

        # Update design variables
        DVDict["c"] = x["chord"]
        DVDict["theta_f"] = x["fiberangle"].item()
        STICKSolver.setDesignVars(DVDict)

        # --- Solve ---
        STICKSolver.solve()

        # --- Grab cost funcs ---
        funcs = STICKSolver.evalFunctions(funcs, evalFuncs=evalFuncs)

        # --- Set objective ---
        funcs["obj"] = 0
        # funcs["obj"] = funcs["ksflutter"]
        for iapp in STICKSolver.solverOptions["appendageList"]:
            compName = iapp["compName"]
            funcs["obj"] += funcs[f"cdi-{compName}"] + funcs[f"cdj-{compName}"] + funcs[f"cdpr-{compName}"] + funcs[f"cds-{compName}"]

        print("These are the funcs: ")
        pp(funcs)

        return funcs

    def flutterFuncs(x):
        return funcs

    def cruiseFuncsSens(x, funcs):
        funcsSens = {}

        # Update design variables
        DVDict["c"] = x["chord"]
        DVDict["theta_f"] = x["fiberangle"].item()
        STICKSolver.setDesignVars(DVDict)

        # --- Solve sensitivity ---
        funcsSens = STICKSolver.evalFunctionsSens(funcsSens, evalFuncs=evalFuncs)

        # --- Set objective ---
        # funcsSens["obj"] = funcsSens["ksflutter"]
        funcs["obj"] = 0
        for iapp in STICKSolver.solverOptions["appendageList"]:
            compName = iapp["compName"]
            funcsSens["obj"] += funcsSens[f"cdi-{compName}"] + funcsSens[f"cdj-{compName}"] + funcsSens[f"cdpr-{compName}"] + funcsSens[f"cds-{compName}"]

        pp(funcsSens)

        return funcsSens

    # def objCon(funcs): # this part is only for multipoint

    #     funcs["obj"] = funcs["ksflutter"]

    #     if printOK:
    #         print("funcs in obj: ", funcs)

    #     return funcs

    # ************************************************
    #     Setup optimizer
    # ************************************************
    optProb = Optimization("opt", cruiseFuncs)

    optProb.addObj("obj")

    # ---------------------------
    #   DVs
    # ---------------------------
    optProb.addVarGroup(
        name="chord",
        nVars=nNodes,
        varType="c",
        value=DVDict["c"],
        lower=0.01,
        upper=0.2,
    )
    # optProb.addVarGroup(name="ab", nVars=nNodes,varType="c", value=DVDict["ab"], lower=-0.1, upper=0.1)
    optProb.addVarGroup(
        name="fiberangle",
        nVars=1,
        varType="c",
        value=DVDict["theta_f"],
        lower=-np.deg2rad(-45),
        upper=np.deg2rad(45),
        scale=1.0,
    )

    optProb.printSparsity()

    outputDir = "./pyDCFoilOUTPUT/"
    Path(outputDir).mkdir(parents=True, exist_ok=True)
    if args.optimizer == "SLSQP":
        optOptions = {
            "IFILE": "SLSQP.out",
        }
    elif args.optimizer == "SNOPT":
        optOptions = {
            "Major feasibility tolerance": 1e-4,
            "Major optimality tolerance": 1e-4,
            "Difference interval": 1e-4,
            "Hessian full memory": None,
            "Function precision": 1e-8,
            "Print file": os.path.join(outputDir, "SNOPT_print.out"),
            "Summary file": os.path.join(outputDir, "SNOPT_summary.out"),
            "Verify level": -1,  # NOTE: verify level 0 is pretty useless; just use level 1--3 when testing a new feature
            # "Linesearch tolerance": 0.99,  # all gradients are known so we can do less accurate LS
            # "Nonderivative linesearch": None,  # Comment out to specify yes nonderivative (nonlinear problem)
            # "Major Step Limit": 5e-3,
            # "Major iterations limit": 1,  # NOTE: for debugging; remove before runs if left active by accident
        }

    opt = OPT(args.optimizer, options=optOptions)

    sol = opt(optProb, sens=cruiseFuncsSens, storeHistory=f"{outputDir}/opt.hst")

    # print(sol)
