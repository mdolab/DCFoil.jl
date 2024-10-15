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
from SETUP import setup_dcfoil, setup_dvgeo
from mpi4py import MPI

comm = MPI.COMM_WORLD
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
    if comm.rank == 0:
        print(30 * "-")
        print("Arguments are", flush=True)
        for arg in vars(args):
            print(f"{arg:<20}:{getattr(args, arg)}", flush=True)
        print(30 * "-", flush=True)

    # ************************************************
    #     Input Files
    # ************************************************
    files = {}
    files["gridFile"] = ""
    filesSETUP = {
    "adflow": "SETUP/setup_adflow.py",
    "constants": "SETUP/setup_constants.py",
    "dvgeo": "SETUP/setup_dvgeo.py",
    "warping": "SETUP/setup_warp.py",
}

    # ************************************************
    #     Geometric design variables
    # ************************************************
    DVGeo = setup_dvgeo.setup(args, comm, files)

    # ************************************************
    #     Create dynamic beam solver
    # ************************************************
    STICKSolver = setup_dcfoil.setup(args, comm, files)

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
            funcs["obj"] += (
                funcs[f"cdi-{compName}"]
                + funcs[f"cdj-{compName}"]
                + funcs[f"cdpr-{compName}"]
                + funcs[f"cds-{compName}"]
            )

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

        print("all funcsSens:")
        pp(funcsSens)
        print("", flush=True)

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
