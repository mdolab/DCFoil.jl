# --- Python 3.10 ---
"""
@File          :   OMJLLiftingLine.py
@Date created  :   2025/02/04
@Last modified :   2025/02/04
@Author        :   Galen Ng
@Desc          :   OpenMDAO component for the Julia lifting line code
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

jl.include("../src/struct/beam_om.jl")  # discipline 1
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2
jl.include("../src/loadtransfer/ldtransfer_om.jl")  # coupling components

from omjlcomps import JuliaExplicitComp, JuliaImplicitComp

ptVec = np.array(
    [
        -0.07,
        0.0,
        0.0,
        -0.0675,
        0.037,
        0.0,
        -0.065,
        0.074,
        0.0,
        -0.0625,
        0.111,
        0.0,
        -0.06,
        0.148,
        0.0,
        -0.0575,
        0.185,
        0.0,
        -0.055,
        0.222,
        0.0,
        -0.0525,
        0.259,
        0.0,
        -0.05,
        0.296,
        0.0,
        -0.0475,
        0.333,
        0.0,
        -0.0675,
        -0.037,
        0.0,
        -0.065,
        -0.074,
        0.0,
        -0.0625,
        -0.111,
        0.0,
        -0.06,
        -0.148,
        0.0,
        -0.0575,
        -0.185,
        0.0,
        -0.055,
        -0.222,
        0.0,
        -0.0525,
        -0.259,
        0.0,
        -0.05,
        -0.296,
        0.0,
        -0.0475,
        -0.333,
        0.0,
        0.07,
        0.0,
        0.0,
        0.0675,
        0.037,
        0.0,
        0.065,
        0.074,
        0.0,
        0.0625,
        0.111,
        0.0,
        0.06,
        0.148,
        0.0,
        0.0575,
        0.185,
        0.0,
        0.055,
        0.222,
        0.0,
        0.0525,
        0.259,
        0.0,
        0.05,
        0.296,
        0.0,
        0.0475,
        0.333,
        0.0,
        0.0675,
        -0.037,
        0.0,
        0.065,
        -0.074,
        0.0,
        0.0625,
        -0.111,
        0.0,
        0.06,
        -0.148,
        0.0,
        0.0575,
        -0.185,
        0.0,
        0.055,
        -0.222,
        0.0,
        0.0525,
        -0.259,
        0.0,
        0.05,
        -0.296,
        0.0,
        0.0475,
        -0.333,
        0.0,
    ]
)
nodeConn = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 11, 12, 13, 14, 15, 16, 17, 18],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ]
)
nNodes = 5
nNodesStrut = 3
appendageOptions = {
    "compName": "rudder",
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
appendageList = [appendageOptions]
solverOptions = {
    # ---------------------------
    #   I/O
    # ---------------------------
    "name": "test",
    # "gridFile": files["gridFile"],
    "debug": False,
    "writeTecplotSolution": True,
    # "outputDir": outputDir,
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

appendageParams = {  # THIS IS BASED OFF OF THE MOTH RUDDER
    "alfa0": 6.0,  # initial angle of attack [deg]
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

# Need to set struct damping once at the beginning
solverOptions = jl.FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[0])
breakpoint()
# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--run_struct", action="store_true", default=False)
parser.add_argument("--run_flow", action="store_true", default=False)
parser.add_argument("--test_partials", action="store_true", default=False)
args = parser.parse_args()

# --- Echo the args ---
print(30 * "-")
print("Arguments are", flush=True)
for arg in vars(args):
    print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
print(30 * "-", flush=True)


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
    jlcomp=jl.OMLiftingLineFuncs(
        nodeConn, appendageParams, appendageOptions, solverOptions
    )
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
if args.run_struct:
    model.add_subsystem(
        "beamstruct",
        impcomp_struct_solver,
        promotes_inputs=["ptVec", "traction_forces"],
        promotes_outputs=["deflections"],
    )
    model.add_subsystem(
        "beamstruct_funcs",
        expcomp_struct_func,
        promotes_inputs=["ptVec","deflections"],
        promotes_outputs=["*"],  # everything!
    )
elif args.run_flow:

    # --- Do nonlinear liftingline ---
    model.add_subsystem(
        "liftingline",
        impcomp_LL_solver,
        promotes_inputs=["ptVec"],
        promotes_outputs=["gammas"],
    )
    model.add_subsystem(
        "liftingline_funcs",
        expcomp_LL_func,
        promotes_inputs=["gammas", "ptVec"],  # promotion auto connects these variables
        promotes_outputs=["*"],  # everything!
    )
else:
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
        promotes_inputs=["ptVec","deflections"],
        promotes_outputs=["*"],  # everything!
    )
    model.add_subsystem(
        "liftingline",
        impcomp_LL_solver,
        promotes_inputs=["ptVec"],
        promotes_outputs=["gammas"],
    )
    model.add_subsystem(
        "liftingline_funcs",
        expcomp_LL_func,
        promotes_inputs=["gammas", "ptVec"],  # promotion auto connects these variables
        promotes_outputs=["*"],  # everything!
    )
    # # --- Now add load transfer capabilities ---
    # model.add_subsystem("loadtransfer", expcomp_load, promotes_inputs=["*"], promotes_outputs=["*"])
    # model.add_subsystem("displtransfer", expcomp_load, promotes_inputs=["*"], promotes_outputs=["*"])


# ************************************************
#     Setup problem
# ************************************************

prob = om.Problem(model)

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options["optimizer"] = "SLSQP"
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
    # "Linesearch tolerance": 0.99,  # all gradients are known so we can do less accurate LS
    # "Nonderivative linesearch": None,  # Comment out to specify yes nonderivative (nonlinear problem)
    # "Major Step Limit": 5e-3,
    # "Major iterations limit": 1,  # NOTE: for debugging; remove before runs if left active by accident
}

prob.model.add_design_var("ptVec")
# prob.model.add_objective("liftingline.F_x")

# prob.model.nonlinear_solver = om.NewtonSolver(
#     solve_subsystems=True,
#     iprint=2,
# )
# prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS() # this is needed to get the system to converge but it sucks
# prob.model.nonlinear_solver.linesearch.options["maxiter"] = 10
# prob.model.nonlinear_solver.linesearch.options["iprint"] = 2

prob.setup()

prob.set_val("ptVec", ptVec)

if args.run_struct:
    tractions = prob.get_val("beamstruct.traction_forces")
    tractions[-6] = 10.0
    prob.set_val("beamstruct.traction_forces", tractions)
elif args.run_flow:
    prob.set_val("liftingline.gammas", np.zeros(40))
else:
    tractions = prob.get_val("beamstruct.traction_forces")
    tractions[-6] = 10.0
    prob.set_val("beamstruct.traction_forces", tractions)
    prob.set_val("liftingline.gammas", np.zeros(40))

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

print("running model again...\n" + "-" * 50)
starttime = time.time()
prob.run_model()
endtime = time.time()
print("model run complete\n" + "-" * 50)
print(f"Time taken to run model: {endtime-starttime:.2f} s")

if args.run_struct:
    print("bending deflections", prob.get_val("beamstruct.deflections")[2::9])
    print("twisting deflections", prob.get_val("beamstruct.deflections")[4::9])
elif args.run_flow:
    print("nondimensional gammas", prob.get_val("gammas"))
    print("CL", prob.get_val("CL"))
    print("force distribution", prob.get_val("forces_dist"))
else:
    print("nondimensional gammas", prob.get_val("liftingline.gammas"))
    print("CL", prob.get_val("CL"))
    # print("force distribution", prob.get_val("forces_dist"))
    print("bending deflections", prob.get_val("beamstruct.deflections")[2::9])
    print("twisting deflections", prob.get_val("beamstruct.deflections")[4::9])
    # print(prob["liftingline.f_xy"])  # Should print `[-15.]`
    print("induced drag force", prob.get_val("F_x"))
    print("lift force", prob.get_val("F_z"))
    print("spray drag:", prob.get_val("Ds"), f"\tprofile drag: {prob.get_val('Dpr')}\t wavedrag: {prob.get_val('Dw')}\t junctiondrag: {prob.get_val('Dj')}")
    print("spray drag coeff:", prob.get_val("CDs"), f"\tprofile drag coeff: {prob.get_val('CDpr')}\t wavedrag coeff: {prob.get_val('CDw')}\t junctiondrag coeff: {prob.get_val('CDj')}")

# --- Check partials after you've solve the system!! ---
if args.test_partials:
    fileName = "partials.out"
    f = open(fileName, "w")
    f.write("PARTIALS\n")
    # prob.set_check_partial_options(wrt=[""],)
    f.write("=" * 50)
    f.write(" liftingline partials ")
    f.write("=" * 50)
    # prob.check_partials(
    #     out_stream=f,
    #     includes=["liftingline"],
    #     method="fd",
    #     compact_print=True,
    # )
    prob.check_partials(
        out_stream=f,
        includes=["liftingline_funcs"],
        method="fd",
        step=1e-4,  # now we're cooking :)
        # compact_print=True,
    )
    f.write("=" * 50)
    f.write(" structural partials ")
    f.write("=" * 50)
    prob.check_partials(
        out_stream=f,
        includes=["beamstruct"],
        method="fd",
        compact_print=True,
    )
    prob.check_partials(
        out_stream=f,
        includes=["beamstruct_funcs"],
        method="fd",
        # compact_print=True,
    ) # THESE ARE GOOD
    # breakpoint()
    f.close()

# --- Testing other values ---
# prob.set_val("liftingline.x", 5.0)
# prob.set_val("liftingline.y", -2.0)

# prob.run_model()
# print(prob.get_val("liftingline.f_xy"))  # Should print `[-5.]`


# ************************************************
#     Do optimization
# ************************************************
# prob.run_driver()
# print(f"f_xy = {prob.get_val('liftingline.f_xy')}")  # Should print `[-27.33333333]`
# print(f"x = {prob.get_val('liftingline.x')}")  # Should print `[6.66666633]`
# print(f"y = {prob.get_val('liftingline.y')}")  # Should print `[-7.33333367]`
