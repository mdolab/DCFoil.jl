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
import matplotlib.pyplot as plt
from scipy import sparse
# from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
#import niceplots
from pprint import pprint as pp
from pyoptsparse import Optimization, OPT, History
# Add parent path to sys.path to import pyDCFoil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from src.wrappers.pyDCFoil import pyDCFOIL


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":


    # ************************************************
    #     Create dynamic beam solver
    # ************************************************
    nNodes = 3
    nNodesStrut = 3
    solverOptions = {
        # ---------------------------
        #   I/O
        # ---------------------------
        "name": "test",
        "debug": True,
        "writeTecplotSolution": True,
        "outputDir": "./OUTPUT/",
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "config": "wing",
        # "config": "t-foil",
        "nNodes": nNodes,  # number of nodes on foil half wing
        "nNodeStrut": nNodesStrut,  # nodes on strut
        "rotation": 0.0,  # deg
        "gravityVector": [0.0, 0.0, -9.81],
        "use_tipMass": False,
        # ---------------------------
        #   Flow
        # ---------------------------
        "U∞": 5.0,  # free stream velocity [m/s]
        "ρ_f": 1000.0,  # fluid density [kg/m³]
        "use_freeSurface": False,
        "use_cavitation": False,
        "use_ventilation": False,
        # ---------------------------
        #   Structure
        # ---------------------------
        "material": "cfrp",  # preselect from material library
        "strut_material": "cfrp",
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static": True,
        # --- Forced solve ---
        "run_forced": True,
        "fSweep": np.linspace(0.1, 1000.0, 1000),
        "tipForceMag": 1.0,
        # --- p-k (Eigen) solve ---
        "run_modal": True,
        "run_flutter": True,
        "nModes": 4,
        "uRange": [170.0, 190.0],
        "maxQIter": 100,  # that didn't fix the slow run time...
        "rhoKS": 80.0,
    }
    DVDict = {
        "α₀": 6.0,  # initial angle of attack [deg]
        "Λ": np.deg2rad(-15.0),  # sweep angle [rad]
        "zeta": 0.04,  # modal damping ratio at first 2 modes
        "c": 0.1 * np.ones(nNodes),  # chord length [m]
        "s": 0.3,  # semispan [m]
        "ab": 0 * np.ones(nNodes),  # dist from midchord to EA [m]
        "toc": 0.12 * np.ones(nNodes),  # thickness-to-chord ratio
        "x_αb": 0 * np.ones(nNodes),  # static imbalance [m]
        "θ": np.deg2rad(15),  # fiber angle global [rad]
        # --- Strut vars ---
        "beta": 0.0,  # yaw angle wrt flow [deg]
        "s_strut": 0.4,  # from Yingqian
        "c_strut": 0.1 * np.ones(nNodesStrut),  # chord length [m]
        "toc_strut": 0.12 * np.ones(nNodesStrut),  # thickness-to-chord ratio
        "ab_strut": 0 * np.ones(nNodesStrut),  # dist from midchord to EA [m]
        "x_αb_strut": 0 * np.ones(nNodesStrut),  # static imbalance [m]
        "θ_strut": np.deg2rad(15),  # fiber angle global [rad]
    }
    evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]

    DynamicSolver = pyDCFOIL(DVDict, evalFuncs, debug=True, options=solverOptions)

    # ************************************************
    #     Functions
    # ************************************************
    def cruiseFuncs(x):
        # FLUTTER FUNCS AT SPEED ENVELOPE BOUND ('flutterFuncs(x)' instead?)
        funcs = {}

        # Update design variables
        DVDict["c"] = x["chord"]
        DVDict["θ"] = x["fiberangle"].item()
        DynamicSolver.setDesignVars(DVDict)

        # --- Solve ---
        DynamicSolver.solve()

        # --- Grab cost funcs ---
        funcs = DynamicSolver.evalFunctions(funcs)

        # --- Set objective ---
        funcs["obj"] = funcs["ksflutter"]

        pp(funcs)

        return funcs

    def flutterFuncs(x):
        return funcs

    def cruiseFuncsSens(x, funcs):
        funcsSens = {}

        # Update design variables
        DVDict["c"] = x["chord"]
        DVDict["θ"] = x["fiberangle"].item()
        DynamicSolver.setDesignVars(DVDict)

        # --- Solve sensitivity ---
        funcsSens = DynamicSolver.evalFunctionsSens(funcsSens)

        # --- Set objective ---
        funcsSens["obj"] = funcsSens["ksflutter"]

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
        value=DVDict["θ"],
        lower=-np.deg2rad(-45),
        upper=np.deg2rad(45),
        scale=1.0,
    )

    optProb.printSparsity()

    optOptions = {
        "IFILE": "SLSQP.out",
    }

    opt = OPT("SLSQP", options=optOptions)

    sol = opt(optProb, sens=cruiseFuncsSens, storeHistory="opt.hst")

    # print(sol)
