# --- Python 3.10 ---
"""
@File          :   setup_dcfoil.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   Setup DCFoil solver
"""

import numpy as np
from dcfoil import DCFOIL  # make sure to pip install this code


def setup(args, comm, files, evalFuncs):
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
        "gridFile": files["gridFile"],
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
        "Uinf": 5.0,  # free stream velocity [m/s]
        "rhof": 1000.0,  # fluid density [kg/m³]
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

    debug = True

    # --- Instantiate it ---
    STICKSolver = DCFOIL(DVDictList=[DVDict], evalFuncs=evalFuncs, options=solverOptions, debug=debug)

    STICKSolver.addMesh(solverOptions["gridFile"])

    return STICKSolver
