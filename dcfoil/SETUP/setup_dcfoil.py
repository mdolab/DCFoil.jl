# --- Python 3.10 ---
"""
@File          :   setup_dcfoil.py
@Date created  :   2024/10/14
@Last modified :   2024/10/14
@Author        :   Galen Ng
@Desc          :   Setup DCFoil solver
"""

import numpy as np
import juliacall

jl = juliacall.newmodule("DCFoil")

jl.include("../src/io/MeshIO.jl")  # mesh I/O for reading inputs in
jl.include("../src/struct/beam_om.jl")  # discipline 1
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2


def setup(nNodes, nNodesStrut, args, comm, files, flutterSpeed, outputDir: str):
    Grid = jl.DCFoil.add_meshfiles(files["gridFile"], {"junction-first": True})
    # This chunk of code is just to initialize DCFoil properly. If you want to change DVs for the code, do it via OpenMDAO
    appendageOptions = {
        "compName": "rudder",
        # "config": "full-wing",
        "config": "wing",
        "nNodes": nNodes,
        "nNodeStrut": nNodesStrut,
        "use_tipMass": False,
        # "xMount": 3.355,
        "xMount": 0.0,
        "material": "cfrp",
        "strut_material": "cfrp",
        "path_to_geom_props": "./INPUT/1DPROPS/",
        "path_to_struct_props": None,
        "path_to_geom_props": None,
    }
    appendageList = [appendageOptions]
    solverOptions = {
        # ---------------------------
        #   I/O
        # ---------------------------
        "name": "test",
        # "gridFile": files["gridFile"],
        "debug": args.debug,
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
        "Uinf": 11.0,  # free stream velocity [m/s]
        "rhof": 1025.0,  # fluid density [kg/m³]
        "nu": 1.1892e-06,  # fluid kinematic viscosity [m²/s]
        "use_nlll": True,
        "use_freeSurface": args.freeSurf,
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
        "run_forced": False,
        "fRange": [0.1, 1000.0],
        # --- Great lakes ---
        "waveSpectrum": "ISSC",
        "Hsig": 1.5,  # significant wave height [m]
        "omegaz": 2 * np.pi / 3.0,  # zero-crossing frequency [rad/s]
        "headingAngle": np.deg2rad(180.0),  # heading angle of the waves [rad]
        "tipForceMag": 1.0,
        "run_body": False,
        # --- p-k (Eigen) solve ---
        "run_modal": False,
        "run_flutter": args.flutter,
        "nModes": 4,
        "uRange": flutterSpeed,  # [m/s] # just throttle this to get convergent results
        # "uRange": [10.0 / 1.9438, 15.0 / 1.9438],  # [kts -> m/s]
        "maxQIter": 100,  # that didn't fix the slow run time...
        "rhoKS": 500.0,
    }

    appendageParams = {
        "alfa0": 6.0,  # initial angle of attack [deg]
        "zeta": 0.04,  # modal damping ratio at first 2 modes
        "toc": 0.12 * np.ones(nNodes),  # thickness-to-chord ratio
        "abar": 0 * np.ones(nNodes),  # nondim dist from midchord to EA
        "x_a": 0 * np.ones(nNodes),  # nondim static imbalance
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

    # Need to set struct damping once at the beginning to avoid optimization taking advantage of changing beta
    ptVec, m, n = jl.FEMMethods.unpack_coords(Grid.LEMesh, Grid.TEMesh)
    nodeConn = np.array(Grid.nodeConn)
    solverOptions = jl.FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[0])

    # number of strips and FEM nodes
    if appendageOptions["config"] == "full-wing":
        npt_wing = jl.LiftingLine.NPT_WING
        npt_wing_full = jl.LiftingLine.NPT_WING
        n_node = nNodes * 2 - 1  # for full span
    else:
        npt_wing = jl.LiftingLine.NPT_WING / 2  # for half wing
        npt_wing_full = jl.LiftingLine.NPT_WING  # full span
        # check if npt_wing is integer
        if npt_wing % 1 != 0:
            raise ValueError("NPT_WING must be an even number for symmetric analysis")
        npt_wing = int(npt_wing)
        n_node = nNodes

    if args.foil == "amcfull":
        appendageParams["abar"] = -0.0464 * 2.0 * np.ones(nNodes)  # she nondimensionalized by chord, not semichord
        print("Setting elastic axis offset for CFRP NACA0009 from Julie's JFM part III paper")

    return ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, npt_wing, npt_wing_full, n_node
