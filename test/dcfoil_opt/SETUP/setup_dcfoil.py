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


def setup_old(args, comm, files, evalFuncs, outputDir: str, ap):
    nNodes = 5
    nNodesStrut = 3
    mainFoilOptions = {
        "compName": "rudder",
        # "config": "t-foil",
        "config": "full-wing",
        "nNodes": nNodes,
        "nNodeStrut": nNodesStrut,
        "use_tipMass": False,
        "xMount": 3.355,
        "material": "cfrp",
        "strut_material": "cfrp",
        "path_to_geom_props": "./INPUT/1DPROPS/",
        "path_to_struct_props": None,
        "path_to_geom_props": None,
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
        "outputDir": outputDir,
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "appendageList": appendageList,
        "gravityVector": [0.0, 0.0, -9.81],
        # ---------------------------
        #   Flow
        # ---------------------------
        "Uinf": 18.0,  # free stream velocity [m/s]
        "rhof": 1025.0,  # fluid density [kg/m³]
        "nu": 1.1892e-06,  # fluid kinematic viscosity [m²/s]
        "use_nlll": True,
        "use_freeSurface": False,
        "use_cavitation": False,
        "use_ventilation": False,
        "use_dwCorrection": False,
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static": True,
        "res_jacobian": "analytic",
        # --- Forced solve ---
        "run_forced": True,
        "run_forced": False,
        "fRange": [0.1, 1000.0],
        "tipForceMag": 1.0,
        "run_body": False,
        # --- p-k (Eigen) solve ---
        "run_modal": False,
        "run_flutter": False,
        "nModes": 4,
        "uRange": [10.0 / 1.9438, 50.0 / 1.9438],  # [kts -> m/s]
        "maxQIter": 100,  # that didn't fix the slow run time...
        "rhoKS": 500.0,
    }
    if args.is_dynamic:
        solverOptions.update(
            {
                "run_modal": True,
                "run_flutter": True,
            }
        )

    params = {  # THIS IS BASED OFF OF THE MOTH RUDDER
        "alfa0": ap.alpha,  # initial angle of attack [deg]
        # "sweep": np.deg2rad(0.0),  # sweep angle [rad]
        "zeta": 0.04,  # modal damping ratio at first 2 modes
        # "c": np.linspace(0.14, 0.095, nNodes),  # chord length [m]
        # "s": 0.333,  # semispan [m]
        "ab": 0 * np.ones(nNodes),  # dist from midchord to EA [m]
        "toc": 0.075 * np.ones(nNodes),  # thickness-to-chord ratio
        "x_ab": 0 * np.ones(nNodes),  # static imbalance [m]
        "theta_f": np.deg2rad(5.0),  # fiber angle global [rad]
        # --- Strut vars ---
        "depth0": 0.4,  # submerged depth of strut [m] # from Yingqian
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
    STICKSolver = DCFOIL(
        appendageParamsList=[params],
        evalFuncs=evalFuncs,
        options=solverOptions,
        debug=debug,
    )

    # --- Variables for DCFoil ---
    valDict = {
        "alfa0": ap.alpha,
        "theta_f": params["theta_f"],
        "toc": params["toc"],
    }
    lowerDict = {
        "alfa0": -5.0,
        "theta_f": np.deg2rad(0),  # rad
        "toc": 0.9 * params["toc"],
    }
    upperDict = {
        "alfa0": 5.0,
        "theta_f": np.deg2rad(30),  # rad
        "toc": 1.1 * params["toc"],
    }
    scaleDict = {
        "alfa0": 1 / (upperDict["alfa0"] - lowerDict["alfa0"]),
        "theta_f": 1 / (upperDict["theta_f"] - lowerDict["theta_f"]),
        "toc": 1 / (upperDict["toc"] - lowerDict["toc"]),
    }

    return STICKSolver, solverOptions, valDict, lowerDict, upperDict, scaleDict


def setup(args, comm, files, evalFuncs, outputDir: str):
    nNodes = 5
    nNodesStrut = 3
    mainFoilOptions = {
        "compName": "rudder",
        # "config": "t-foil",
        "config": "full-wing",
        "nNodes": nNodes,
        "nNodeStrut": nNodesStrut,
        "use_tipMass": False,
        "xMount": 3.355,
        "material": "cfrp",
        "strut_material": "cfrp",
        "path_to_geom_props": "./INPUT/1DPROPS/",
        "path_to_struct_props": None,
        "path_to_geom_props": None,
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
        "outputDir": outputDir,
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "appendageList": appendageList,
        "gravityVector": [0.0, 0.0, -9.81],
        # ---------------------------
        #   Flow
        # ---------------------------
        "Uinf": 18.0,  # free stream velocity [m/s]
        "rhof": 1025.0,  # fluid density [kg/m³]
        "nu": 1.1892e-06,  # fluid kinematic viscosity [m²/s]
        "use_nlll": True,
        "use_freeSurface": False,
        "use_cavitation": False,
        "use_ventilation": False,
        "use_dwCorrection": False,
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static": True,
        "res_jacobian": "analytic",
        # --- Forced solve ---
        "run_forced": True,
        "run_forced": False,
        "fRange": [0.1, 1000.0],
        "tipForceMag": 1.0,
        "run_body": False,
        # --- p-k (Eigen) solve ---
        "run_modal": False,
        "run_flutter": False,
        "nModes": 4,
        "uRange": [10.0 / 1.9438, 50.0 / 1.9438],  # [kts -> m/s]
        "maxQIter": 100,  # that didn't fix the slow run time...
        "rhoKS": 500.0,
    }
    if args.is_dynamic:
        solverOptions.update(
            {
                "run_modal": True,
                "run_flutter": True,
            }
        )

    appendageParams = {  # THIS IS BASED OFF OF THE MOTH RUDDER
        "alfa0": 2.0,  # initial angle of attack [deg]
        # "sweep": np.deg2rad(0.0),  # sweep angle [rad]
        "zeta": 0.04,  # modal damping ratio at first 2 modes
        # "c": np.linspace(0.14, 0.095, nNodes),  # chord length [m]
        # "s": 0.333,  # semispan [m]
        "ab": 0 * np.ones(nNodes),  # dist from midchord to EA [m]
        "toc": 0.075 * np.ones(nNodes),  # thickness-to-chord ratio
        "x_ab": 0 * np.ones(nNodes),  # static imbalance [m]
        "theta_f": np.deg2rad(5.0),  # fiber angle global [rad]
        # --- Strut vars ---
        "depth0": 0.4,  # submerged depth of strut [m] # from Yingqian
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

    # --- Variables for DCFoil ---
    valDict = {
        "alfa0": 6.0,
        "theta_f": appendageParams["theta_f"],
        "toc": appendageParams["toc"],
    }
    lowerDict = {
        "alfa0": -5.0,
        "theta_f": np.deg2rad(0),  # rad
        "toc": 0.9 * appendageParams["toc"],
    }
    upperDict = {
        "alfa0": 10.0,
        "theta_f": np.deg2rad(30),  # rad
        "toc": 1.1 * appendageParams["toc"],
    }
    scaleDict = {
        "alfa0": 1 / (upperDict["alfa0"] - lowerDict["alfa0"]),
        "theta_f": 1 / (upperDict["theta_f"] - lowerDict["theta_f"]),
        "toc": 1 / (upperDict["toc"] - lowerDict["toc"]),
    }

    return solverOptions, appendageParams, appendageList, valDict, lowerDict, upperDict, scaleDict
