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
from SETUP import setup_dcfoil, setup_dvgeo, setup_opt
from mpi4py import MPI
from baseclasses import AeroProblem

comm = MPI.COMM_WORLD

# ==============================================================================
#                         MODELING FUNCTIONS
# ==============================================================================
def compute_clvent(Fnh:float):
    """
    Compute the critical c_l for ventilation

    Parameters
    ----------
    Fnh : float
        Depth-based Froude number

    Returns
    -------
    clvent : float

    TODO: eventually, this should be in the julia layer to get the fnh(XLoc) effect
    """

    sigmav = (PATM - PC) / (0.5*RHOF*UINF**2)
    clvent = (1 - np.exp(-sigmav*Fnh)) / np.sqrt(sigmav*Fnh)

    return clvent

# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="INPUT")
    parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
    parser.add_argument(
        "--geovar",
        type=str,
        default="trwpd",
        help="Geometry variables to test twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
    )
    parser.add_argument(
        "--optimizer", "-o", help="type of optimizer", choices=["SLSQP", "SNOPT"], type=str, default="SNOPT"
    )
    parser.add_argument(
        "--task",
        help="Check end of script for task type",
        type=str,
        default="run",
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
    #     Input Files/ IO
    # ************************************************
    outputDir = "./pyDCFoilOUTPUT/"
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    files = {}
    files["gridFile"] = [
        f"./{args.input}/{args.foil}_foil_stbd_mesh.dcf",
        f"./{args.input}/{args.foil}_foil_port_mesh.dcf",
    ]
    files["FFDFile"] = f"{args.input}/{args.foil}_ffd.xyz"
    # files["FFDFile-port"] = f"{args.input}/{args.foil}_port_ffd.xyz"

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

    evalFuncs = [
        "wtip",
        "psitip",
        "cl",
        "cmy",
        "lift",
        "moment",
        "cd",
        "ksflutter",
        "kscl",
    ]
    STICKSolver, solverOptions = setup_dcfoil.setup(args, comm, files, evalFuncs, outputDir)
    STICKSolver.setDVGeo(DVGeo)
    STICKSolver.addMesh(solverOptions["gridFile"])

    # ************************************************
    #     Functions
    # ************************************************
    def cruiseFuncs(x):
        # FLUTTER FUNCS AT SPEED ENVELOPE BOUND ('flutterFuncs(x)' instead?)
        funcs = {}

        # Set design variables
        ap.setDesignVars(x)
        DVGeo.setDesignVars(x)

        # --- Solve ---
        STICKSolver(ap)

        # --- Grab cost funcs ---
        funcs = STICKSolver.evalFunctions(ap, funcs, evalFuncs=evalFuncs)

        # for iapp in STICKSolver.solverOptions["appendageList"]:
            # compName = iapp["compName"]
        funcs["obj"] += funcs[f"{ap.name}_cd"]
            

        if comm.rank == 0:
            print("These are the funcs: ")
            pp(funcs)

        return funcs

    def cruiseFuncsSens(x, funcs):
        funcsSens = {}

        # --- Solve sensitivity ---
        funcsSens = STICKSolver.evalFunctionsSens(ap, funcsSens, evalFuncs=evalFuncs)

        # The span derivative is the only broken derivative in DCFoil. We FD it here.
        dh = 1e-5
        funcs_i = {}
        funcs_f = {}
        STICKSolver.evalFunctions(ap, funcs_i, evalFuncs=evalFuncs)
        x["span"] += dh
        ap.setDesignVars(x)
        STICKSolver.evalFunctions(ap, funcs_f, evalFuncs=evalFuncs)
        x["span"] -= dh
        for key, value in funcs_i.items():
            funcsSens["span"] = (funcs_f[key] - funcs_i[key]) / dh


        return funcsSens

    def objCon(funcs):  # this part is only needed for multipoint
        # Assemble the objective and any additional constraints:

        funcs["obj"] = funcs[f"{ap.name}_cd"]

        # ---------------------------
        #   Constraints
        # ---------------------------
        # --- Flutter ---
        funcs[f"ksflutter_con_{ap.name}"] = funcs[f"{ap.name}_ksflutter"]

        # --- Lift ---
        funcs["cl_con_" + ap.name] = funcs[f"{ap.name}_cl"] - mycl

        # --- Ventilation ---
        funcs[f"vent_con_{ap.name}"] = funcs[f"{ap.name}_kscl"] - myventcl

        funcs[f"wtip_con_{ap.name}"] = funcs[f"{ap.name}_wtip"] - mywtip

        if printOK:
            print("funcs in obj: ", funcs)

        return funcs

    # ************************************************
    #     Setup optimizer
    # ************************************************
    optProb = Optimization("opt", cruiseFuncs)

    optProb.addObj("obj")

    optProb.printSparsity()

    opt, optOptions = setup_opt.setup(args, outputDir)

    # ==============================================================================
    #                         TASKS
    # ==============================================================================
    # ---------------------------
    #   OPTIMIZATION
    # ---------------------------
    if args.task == "opt":
        sol = opt(optProb, sens=cruiseFuncsSens, storeHistory=f"{outputDir}/opt.hst")

        print(sol)
        print(f"final DVs:\n")
        print(sol.xStar)

    # ---------------------------
    #   ANALYSIS
    # ---------------------------
    if args.task in ["run", "opt"]:

        if args.task == "run":
            x_init = DVGeo.getValues()
            run_dvs = x_init
        if args.task == "opt":
            run_dvs = sol.xStar

        funcs = cruiseFuncs(run_dvs)

        # --- Pretty print output ---
        CLtxt = funcs[f"{ap.name}_cl"]
        CDtxt = funcs[f"{ap.name}_cd"]
        KSFluttertxt = funcs[f"{ap.name}_ksflutter"]
        table = [["CL", CLtxt], ["CD", CDtxt], ["KSFlutter", KSFluttertxt]]
        try:
            from tabulate import tabulate

            print(tabulate(table))
        except Exception:
            import warnings

            warnings.warn("tabulate not installed")
            print(table)
