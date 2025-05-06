# --- Python 3.10 ---
"""
@File          :   test_amc.py
@Date created  :   2025/02/04
@Last modified :   2025/05/02
@Desc          :   This is a test script for assembling parts of DCFoil's static solvers to test the Australia Maritime College hydrofoil geometry.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import time
import argparse

# ==============================================================================
# Extension modules
# ==============================================================================
import openmdao.api as om
import juliacall

jl = juliacall.newmodule("DCFoil")

jl.include("../src/io/MeshIO.jl")  # mesh I/O for reading inputs in
jl.include("../src/struct/beam_om.jl")  # discipline 1
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2
jl.include("../src/loadtransfer/ldtransfer_om.jl")  # coupling components
jl.include("../src/solvers/solveflutter_om.jl")  # discipline 4 flutter solver
jl.include("../src/solvers/solveforced_om.jl")  # discipline 5 forced solver

from omjlcomps import JuliaExplicitComp, JuliaImplicitComp

outputDir = "output"
files = {
    "gridFile": [
        "../INPUT/amc_foil_stbd_mesh.dcf",
        # "../INPUT/amc_foil_port_mesh.dcf", # only add this if config is full wing
    ]
}
Grid = jl.DCFoil.add_meshfiles(files["gridFile"], {"junction-first": True})
# Unpack for this code. Remember Julia is transposed from Python
LECoords = np.array(Grid.LEMesh).T
TECoords = np.array(Grid.TEMesh).T
nodeConn = np.array(Grid.nodeConn)
ptVec, m, n = jl.FEMMethods.unpack_coords(Grid.LEMesh, Grid.TEMesh)
nNodes = 5
nNodesStrut = 3
appendageOptions = {
    "compName": "rudder",
    # "config": "full-wing",
    "config": "wing",
    "nNodes": nNodes,
    "nNodeStrut": nNodesStrut,
    "use_tipMass": False,
    "xMount": 0.0,
    "material": "al6061",
    "strut_material": "al6061",
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
    # "uRange": [10.0 / 1.9438, 50.0 / 1.9438],  # [kts -> m/s]
    "uRange": [10.0 / 1.9438, 15.0 / 1.9438],  # [kts -> m/s]
    "maxQIter": 100,  # that didn't fix the slow run time...
    "rhoKS": 500.0,
}

appendageParams = {
    "alfa0": 6.0,  # initial angle of attack [deg]
    "zeta": 0.04,  # modal damping ratio at first 2 modes
    "ab": 0 * np.ones(nNodes),  # dist from midchord to EA [m]
    "toc": 0.06 * np.ones(nNodes),  # thickness-to-chord ratio
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

# Need to set struct damping once at the beginning to avoid optimization taking advantage of changing beta
solverOptions = jl.FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[0])


npt_wing = jl.LiftingLine.NPT_WING
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--run_struct", action="store_true", default=False)
    # parser.add_argument("--run_flow", action="store_true", default=False)
    # parser.add_argument("--run_flutter", action="store_true", default=False)
    # parser.add_argument("--test_partials", action="store_true", default=False)
    # args = parser.parse_args()

    # # --- Echo the args ---
    # print(30 * "-")
    # print("Arguments are", flush=True)
    # for arg in vars(args):
    #     print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    # print(30 * "-", flush=True)

    impcomp_struct_solver = JuliaImplicitComp(
        jlcomp=jl.OMFEBeam(nodeConn, appendageParams, appendageOptions, solverOptions)
    )
    expcomp_struct_func = JuliaExplicitComp(
        jlcomp=jl.OMFEBeamFuncs(nodeConn, appendageParams, appendageOptions, solverOptions)
    )
    impcomp_LL_solver = JuliaImplicitComp(
        jlcomp=jl.OMLiftingLine(nodeConn, appendageParams, appendageOptions, solverOptions)
    )
    expcomp_LL_func = JuliaExplicitComp(
        jlcomp=jl.OMLiftingLineFuncs(nodeConn, appendageParams, appendageOptions, solverOptions)
    )
    expcomp_load = JuliaExplicitComp(
        jlcomp=jl.OMLoadTransfer(nodeConn, appendageParams, appendageOptions, solverOptions)
    )
    expcomp_displacement = JuliaExplicitComp(
        jlcomp=jl.OMLoadTransfer(nodeConn, appendageParams, appendageOptions, solverOptions)
    )

    model = om.Group()

    # ************************************************
    #     Setup components
    # ************************************************
    # --- Combined hydroelastic ---
    model.add_subsystem(
        "beamstruct",
        impcomp_struct_solver,
        promotes_inputs=["ptVec"],
        promotes_outputs=["deflections"],
    )
    model.add_subsystem(
        "beamstruct_funcs",
        expcomp_struct_func,
        promotes_inputs=["ptVec", "deflections"],
        promotes_outputs=["*"],  # everything!
    )
    model.add_subsystem(
        "liftingline",
        impcomp_LL_solver,
        promotes_inputs=["ptVec", "alfa0", "displacements_col"],
        promotes_outputs=["gammas", "gammas_d"],
    )
    model.add_subsystem(
        "liftingline_funcs",
        expcomp_LL_func,
        promotes_inputs=[
            "gammas",
            "gammas_d",
            "ptVec",
            "alfa0",
            "displacements_col",
        ],  # promotion auto connects these variables
        promotes_outputs=["*"],  # everything!
    )
    # # --- Now add load transfer capabilities ---
    # model.add_subsystem("loadtransfer", expcomp_load, promotes_inputs=["*"], promotes_outputs=["*"])
    # model.add_subsystem("displtransfer", expcomp_load, promotes_inputs=["*"], promotes_outputs=["*"])

    # ************************************************
    #     Setup problem
    # ************************************************

    prob = om.Problem(model)

    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
    outputDir = "output"
    optOptions = {
        "Major feasibility tolerance": 1e-4,
        "Major optimality tolerance": 1e-4,
        "Difference interval": 1e-4,
        "Hessian full memory": None,
        "Function precision": 1e-8,
        "Print file": os.path.join(outputDir, "SNOPT_print.out"),
        "Summary file": os.path.join(outputDir, "SNOPT_summary.out"),
        "Verify level": -1,  # NOTE: verify level 0 is pretty useless; just use level 1--3 when testing a new feature
    }

    prob.model.add_design_var("ptVec")

    prob.setup()

    prob.set_val("ptVec", ptVec)


    # ************************************************
    #     Set starting values
    # ************************************************
    displacementsCol = np.zeros((6, npt_wing))

    prob.set_val("beamstruct.theta_f", np.deg2rad(15))
    prob.set_val("liftingline.displacements_col", displacementsCol)
    prob.set_val("alfa0", appendageParams["alfa0"])
    tractions = prob.get_val("beamstruct.traction_forces")
    tractions[-7] = 100.0
    prob.set_val("beamstruct.traction_forces", tractions)
    prob.set_val("liftingline.gammas", np.zeros(npt_wing))


    # ************************************************
    #     Evaluate model
    # ************************************************
    print("running model...\n" + "-" * 50)
    starttime = time.time()
    prob.final_setup()
    midtime = time.time()
    prob.run_model()
    endtime = time.time()
    print("model run complete\n" + "-" * 50)
    print(f"Time taken to run model: {endtime-midtime:.2f} s")


    # print("force distribution", prob.get_val("forces_dist"))
    print("bending deflections", prob.get_val("beamstruct.deflections")[2::9])
    print("twisting deflections", prob.get_val("beamstruct.deflections")[4::9])
    print("induced drag force", prob.get_val("F_x"))
    print("lift force", prob.get_val("F_z"))
