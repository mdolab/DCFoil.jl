# --- Python 3.10 ---
"""
@File          :   OMJLLiftingLine.py
@Date created  :   2025/02/04
@Last modified :   2025/02/04
@Author        :   Galen Ng
@Desc          :   OpenMDAO component for the Julia lifting line code
                   This is a test script for assembling parts of DCFoil's static and dynamic solvers
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
import openmdao.visualization as omv
import juliacall

jl = juliacall.newmodule("DCFoil")

jl.include("../src/io/MeshIO.jl")  # mesh I/O for reading inputs in
jl.include("../src/struct/beam_om.jl")  # discipline 1
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2
jl.include("../src/loadtransfer/ldtransfer_om.jl")  # coupling components
jl.include("../src/solvers/solveflutter_om.jl")  # discipline 4 flutter solver
jl.include("../src/solvers/solveforced_om.jl")  # discipline 5 forced solver


from omjlcomps import JuliaExplicitComp, JuliaImplicitComp

# from transfer import DisplacementTransfer, LoadTransfer, CLaInterpolation
from transfer_FD import DisplacementTransfer, LoadTransfer, CLaInterpolation

files = {
    "gridFile": [
        "../INPUT/flagstaff_foil_stbd_mesh.dcf",
        "../INPUT/flagstaff_foil_port_mesh.dcf", # only add this if config is full wing
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
    "config": "full-wing",
    # "config": "wing",
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

# 2025-03-17 NOTE GN-SK: this dictionary is just to initialize DCFoil properly. If you want to change DVs for the code, do it via OpenMDAO
appendageParams = {  # THIS IS BASED OFF OF THE MOTH RUDDER
    "alfa0": 6.0,  # initial angle of attack [deg]
    "zeta": 0.04,  # modal damping ratio at first 2 modes
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

# Need to set struct damping once at the beginning to avoid optimization taking advantage of changing beta
solverOptions = jl.FEMMethods.set_structDamping(
    ptVec, nodeConn, appendageParams, solverOptions, appendageList[0]
)


# ==============================================================================
#                         Helper func
# ==============================================================================
def plot_cla():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import niceplots as nplt

    fname = "output/CLa.pdf"
    dosave = not not fname

    # plt.rcParams.update(myOptions)
    niceColors = sns.color_palette("tab10")
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", niceColors)
    cm = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Create figure object
    fig, axes = plt.subplots(
        nrows=1, sharex=True, constrained_layout=True, figsize=(14, 10)
    )

    ax = axes
    ax.plot(prob.get_val("collocationPts")[1, :], prob.get_val("cla"))

    ax.set_xlabel("spanwise location [m]")
    ax.set_ylabel("$c_{\ell_\\alpha}$", rotation="horizontal", ha="right", va="center")

    plt.show(block=(not dosave))
    # nplt.all()
    # for ax in axes.flatten():
    nplt.adjust_spines(ax, outward=True)
    if dosave:
        plt.savefig(fname, format="pdf")
        print("Saved to:", fname)
    plt.close()


# number of strips and FEM nodes
if appendageOptions["config"] == "full-wing":
    npt_wing = jl.LiftingLine.NPT_WING
    n_node_fullspan = nNodes * 2 - 1
else:
    npt_wing = jl.LiftingLine.NPT_WING / 2
    # check if npt_wing is integer
    if npt_wing % 1 != 0:
        raise ValueError("NPT_WING must be an even number for symmetric analysis")
    npt_wing = int(npt_wing)
    n_node_fullspan = nNodes


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":

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
        jlcomp=jl.OMFEBeamFuncs(
            nodeConn, appendageParams, appendageOptions, solverOptions
        )
    )
    impcomp_LL_solver = JuliaImplicitComp(
        jlcomp=jl.OMLiftingLine(
            nodeConn, appendageParams, appendageOptions, solverOptions
        )
    )
    expcomp_LL_func = JuliaExplicitComp(
        jlcomp=jl.OMLiftingLineFuncs(
            nodeConn, appendageParams, appendageOptions, solverOptions
        )
    )
    expcomp_load = JuliaExplicitComp(
        jlcomp=jl.OMLoadTransfer(
            nodeConn, appendageParams, appendageOptions, solverOptions
        )
    )
    expcomp_displacement = JuliaExplicitComp(
        jlcomp=jl.OMLoadTransfer(
            nodeConn, appendageParams, appendageOptions, solverOptions
        )
    )
    expcomp_flutter = JuliaExplicitComp(
        jlcomp=jl.OMFlutter(nodeConn, appendageParams, appendageOptions, solverOptions)
    )
    expcomp_forced = JuliaExplicitComp(
        jlcomp=jl.OMForced(nodeConn, appendageParams, appendageOptions, solverOptions)
    )

    model = om.Group()

    # ************************************************
    #     Setup components
    # ************************************************
    # --- geometry component ---
    # now ptVec is just an input, so use IVC as an placeholder. Later replace IVC with a geometry component
    indep = model.add_subsystem("input", om.IndepVarComp(), promotes=["*"])
    indep.add_output("ptVec", val=ptVec)

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
            promotes_inputs=["ptVec", "deflections"],
            promotes_outputs=["*"],  # everything!
        )
    elif args.run_flow:

        # --- Do nonlinear liftingline ---
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
    else:
        # --- Combined hydroelastic ---
        couple = model.add_subsystem("hydroelastic", om.Group(), promotes=["*"])

        # structure
        couple.add_subsystem(
            "beamstruct",
            impcomp_struct_solver,
            promotes_inputs=["ptVec", "traction_forces"],
            promotes_outputs=["deflections"],
        )
        couple.add_subsystem(
            "beamstruct_funcs",
            expcomp_struct_func,
            promotes_inputs=["ptVec", "deflections"],
            promotes_outputs=["*"],  # everything!
        )

        # displacement transfer
        couple.add_subsystem(
            "disp_transfer",
            DisplacementTransfer(n_node=n_node_fullspan, n_strips=npt_wing, xMount=appendageOptions["xMount"]),
            promotes_inputs=["nodes", "deflections", "collocationPts"],
            promotes_outputs=[("disp_colloc", "displacements_col")],
        )

        # hydrodynamics
        couple.add_subsystem(
            "liftingline",
            impcomp_LL_solver,
            promotes_inputs=["ptVec", "alfa0", "displacements_col"],
            promotes_outputs=["gammas", "gammas_d"],
        )
        couple.add_subsystem(
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

        # load transfer
        couple.add_subsystem(
            "load_transfer",
            LoadTransfer(n_node=n_node_fullspan, n_strips=npt_wing, xMount=appendageOptions["xMount"]),
            promotes_inputs=[("forces_hydro", "forces_dist"), "collocationPts", "nodes"],
            promotes_outputs=[("loads_str", "traction_forces")],
        )

        # hydroelastic coupled solver
        couple.nonlinear_solver = om.NonlinearBlockGS(use_aitken=False, maxiter=10, iprint=2, atol=1e-10, rtol=0)
        ### couple.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=50, iprint=2, atol=1e-7, rtol=0)
        couple.linear_solver = om.DirectSolver()   # for adjoint

        # CL_alpha mapping from flow points to FEM nodes (after hydroelestic loop)
        model.add_subsystem(
            "CLa_interp",
            CLaInterpolation(n_node=n_node_fullspan, n_strips=npt_wing),
            promotes_inputs=["collocationPts", "nodes", ("CL_alpha", "cla")],
            promotes_outputs=[("CL_alpha_node", "cla_node")],
        )

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
    prob.model.add_objective("CDi")

    # prob.model.nonlinear_solver = om.NewtonSolver(
    #     solve_subsystems=True,
    #     iprint=2,
    # )
    # prob.model.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS() # this is needed to get the system to converge but it sucks
    # prob.model.nonlinear_solver.linesearch.options["maxiter"] = 10
    # prob.model.nonlinear_solver.linesearch.options["iprint"] = 2

    prob.setup(check=False)

    # om.n2(prob)

    prob.set_val("ptVec", ptVec)

    if args.run_struct:
        tractions = prob.get_val("beamstruct.traction_forces")
        tractions[-7] = 100.0
        prob.set_val("beamstruct.traction_forces", tractions)
    elif args.run_flow:
        prob.set_val("liftingline.gammas", np.zeros(npt_wing))
        prob.set_val("displacements_col", np.zeros((6, npt_wing)))
        prob.set_val("alfa0", appendageParams["alfa0"])
    else:
        prob.set_val("displacements_col", np.zeros((6, npt_wing)))
        prob.set_val("alfa0", appendageParams["alfa0"])
        prob.set_val("gammas", np.zeros(npt_wing))

        # tip load test
        # loads = np.zeros(9 * n_node_fullspan)  # 9 forces per node
        # loads[4 * 9 + 1] = 1000
        # loads[4 * 9 + 2] = 1000   # tip vertical force (z direction)
        # tractions = prob.set_val("traction_forces", loads)

        # set fiber angle
        fiber_angle = np.deg2rad(0)
        prob.set_val('beamstruct.theta_f', fiber_angle)
        prob.set_val('beamstruct_funcs.theta_f', fiber_angle)
        prob.set_val('beamstruct.toc', 0.075 * np.ones(nNodes))
        prob.set_val('beamstruct_funcs.toc', 0.075 * np.ones(nNodes))

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

    # print("running model again...\n" + "-" * 50)
    # starttime = time.time()
    # prob.run_model()
    # endtime = time.time()
    # print("model run complete\n" + "-" * 50)
    # print(f"Time taken to run model: {endtime-starttime:.2f} s")

    # --- compute total derivatives ---
    if not args.run_struct:
        wrt = ['ptVec']
        of = ['CDw', 'CDpr', 'CDj', 'CDs']
        print('\ncomputing totals...')
        prob.compute_totals(of, wrt)
        print('done!\n')
        # NOTE: when using hydroelastic (with or without solver, with or without jax), compute_totals fails saying RAD for empirical drag partials is getting complex variables (it works if I set FIDI for empirical drag partials in liftingline_om.jl)
        #       it still fails even when I used transfer_FD.py (no Jax)
        #       compute_totals works fine if I do --run_flow 
    
    if args.run_struct:
        print("bending deflections", prob.get_val("beamstruct.deflections")[2::9])
        print("twisting deflections", prob.get_val("beamstruct.deflections")[4::9])
        print("all deflections", prob.get_val("beamstruct.deflections"))

        # Change fiber angle and rerun
        prob.set_val("beamstruct.theta_f", np.deg2rad(-15.0))
        prob.run_model()
        print("print again with theta_f = -15.0")
        print("bending deflections", prob.get_val("beamstruct.deflections")[2::9])
        print("twisting deflections", prob.get_val("beamstruct.deflections")[4::9])
        print("all deflections", prob.get_val("beamstruct.deflections"))

    elif args.run_flow:
        print("nondimensional gammas", prob.get_val("gammas"))
        print("CL", prob.get_val("CL"))  # should be around CL = 0.507 something
        print("CLa", prob.get_val("cla_col"))  #
        # print("force distribution", prob.get_val("forces_dist"))

    else:
        print("nondimensional gammas", prob.get_val("gammas"))
        print("nondimensional gammas_d", prob.get_val("gammas_d"))
        print("CL", prob.get_val("CL"))
        print("CLa", prob.get_val("cla_col"))
        print("mesh", prob.get_val("nodes"))
        print("elemConn", prob.get_val("elemConn"))
        print("collocationPts\n", prob.get_val("collocationPts")[0, :])
        print(prob.get_val("collocationPts")[1, :])
        print(prob.get_val("collocationPts")[2, :])

        print("fiber angle", prob.get_val("beamstruct.theta_f"), "rad")
        # print("force distribution", prob.get_val("forces_dist"))
        print("bending deflections", prob.get_val("deflections")[2::9])
        print("twisting deflections", prob.get_val("deflections")[4::9])
        print("Rx deflections", prob.get_val("deflections")[3::9])
        # print(prob["liftingline.f_xy"])  # Should print `[-15.]`
        print("induced drag force", prob.get_val("F_x"))
        print("lift force", prob.get_val("F_z"))
        print(
            "spray drag:",
            prob.get_val("Ds"),
            f"\nprofile drag: {prob.get_val('Dpr')}\nwavedrag: {prob.get_val('Dw')}\n junctiondrag: {prob.get_val('Dj')}",
        )
        print(
            "spray drag coeff:",
            prob.get_val("CDs"),
            f"\tprofile drag coeff: {prob.get_val('CDpr')}\t wavedrag coeff: {prob.get_val('CDw')}\t junctiondrag coeff: {prob.get_val('CDj')}",
        )

    om.n2(prob, show_browser=False)

    # --- plot ---
    ptVec = prob.get_val('ptVec').reshape(2, 20, 3)
    nodes = prob.get_val('nodes').swapaxes(0, 1)  # shape (3, n_nodes)
    collocationPts = prob.get_val('collocationPts')    # shape (3, n_strip)
    force_colloc = prob.get_val('forces_dist')   # shape (3, n_strip)
    force_FEM = prob.get_val('traction_forces').reshape(9, n_node_fullspan, order='F')   # shape (9, n_nodes)

    import matplotlib.pyplot as plt

    # --- 3D plot ---
    z_scaler = 10   # exaggerate vertical deflections
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.plot(ptVec[:, :, 0], ptVec[:, :, 1], ptVec[:, :, 2], 'o', color='k', ms=3)
    ax.plot(nodes[0, :], nodes[1, :], nodes[2, :], 'o', color='darkgray', ms=5)
    ax.plot(collocationPts[0, :] - appendageOptions['xMount'], collocationPts[1, :], collocationPts[2, :] * z_scaler, 'o-', color='C0', ms=3)
    ax.set_aspect('equal')

    # --- top view of planform ---
    fig, ax = plt.subplots()
    ax.plot(ptVec[:, :, 1], ptVec[:, :, 0], 'o', color='k', ms=3)
    ax.plot(nodes[1, :], nodes[0, :], 'o', color='darkgray', ms=5)
    ax.plot(collocationPts[1, :], collocationPts[0, :] - appendageOptions['xMount'], 'o-', color='C0', ms=3)
    ax.set_aspect('equal')

    # --- plot displacements ---
    disp_nodes = prob.get_val('deflections').reshape(nNodes * 2 - 1, 9)
    disp_colloc = prob.get_val('displacements_col')
    node_y = nodes[1, :]
    colloc_y = collocationPts[1, :]

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    fig.suptitle('Displacements')
    axs[0, 0].plot(node_y, disp_nodes[:, 0], 'o', color='darkgray', ms=5, label='FEM nodes')
    axs[0, 0].plot(colloc_y, disp_colloc[0, :], 'o-', color='C0', ms=3, label='Collocation points')
    axs[0, 0].set_ylabel('disp X')
    axs[0, 0].set_xticklabels([])
    axs[0, 0].legend()

    axs[1, 0].plot(node_y, disp_nodes[:, 1], 'o', color='darkgray', ms=5)
    axs[1, 0].plot(colloc_y, disp_colloc[1, :], 'o-', color='C0', ms=3)
    axs[1, 0].set_ylabel('disp Y')
    axs[1, 0].set_xticklabels([])

    axs[2, 0].plot(node_y, disp_nodes[:, 2], 'o', color='darkgray', ms=5)
    axs[2, 0].plot(colloc_y, disp_colloc[2, :], 'o-', color='C0', ms=3)
    axs[2, 0].set_ylabel('disp Z')
    axs[2, 0].set_xlabel('spanwise location')
    
    axs[0, 1].plot(node_y, disp_nodes[:, 3], 'o', color='darkgray', ms=5)
    axs[0, 1].plot(colloc_y, disp_colloc[3, :], 'o-', color='C0', ms=3)
    axs[0, 1].set_ylabel('disp Rx')
    axs[0, 1].set_xticklabels([])

    axs[1, 1].plot(node_y, disp_nodes[:, 4], 'o', color='darkgray', ms=5)
    axs[1, 1].plot(colloc_y, disp_colloc[4, :], 'o-', color='C0', ms=3)
    axs[1, 1].set_ylabel('disp Ry')
    axs[1, 1].set_xticklabels([])

    axs[2, 1].plot(node_y, disp_nodes[:, 5], 'o', color='darkgray', ms=5)
    axs[2, 1].plot(colloc_y, disp_colloc[5, :], 'o-', color='C0', ms=3)
    axs[2, 1].set_ylabel('disp Rz')
    axs[2, 1].set_xlabel('spanwise location')

    fig.tight_layout()
    fig.savefig('displacements.pdf', bbox_inches='tight')

    # --- plot forces ---
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    fig.suptitle('Forces')

    axs[0, 0].plot(node_y, force_FEM[0, :], 'o', color='darkgray', ms=5)
    axs[0, 0].plot(colloc_y, force_colloc[0, :], 'o-', color='C0', ms=3)
    axs[0, 0].set_ylabel('force X')
    axs[0, 0].set_xticklabels([])

    axs[1, 0].plot(node_y, force_FEM[1, :], 'o', color='darkgray', ms=5)
    axs[1, 0].plot(colloc_y, force_colloc[1, :], 'o-', color='C0', ms=3)
    axs[1, 0].set_ylabel('force Y')
    axs[1, 0].set_xticklabels([])

    axs[2, 0].plot(node_y, force_FEM[2, :], 'o', color='darkgray', ms=5)
    axs[2, 0].plot(colloc_y, force_colloc[2, :], 'o-', color='C0', ms=3)
    axs[2, 0].set_ylabel('force Z')
    axs[2, 0].set_xlabel('spanwise location')

    axs[0, 1].plot(node_y, force_FEM[3, :], 'o', color='darkgray', ms=5)
    axs[0, 1].set_ylabel('moment X')
    axs[0, 1].set_xticklabels([])

    axs[1, 1].plot(node_y, force_FEM[4, :], 'o', color='darkgray', ms=5)
    axs[1, 1].set_ylabel('moment Y')
    axs[1, 1].set_xticklabels([])

    axs[2, 1].plot(node_y, force_FEM[5, :], 'o', color='darkgray', ms=5)
    axs[2, 1].set_ylabel('moment Z')
    axs[2, 1].set_xlabel('spanwise location')
    
    fig.tight_layout()
    fig.savefig('forces.pdf', bbox_inches='tight')

    # --- plot CL_alpha ---
    cla_flow = prob.get_val("cla")
    cla_node = prob.get_val("cla_node")
    fig, ax = plt.subplots()
    ax.plot(colloc_y, cla_flow, 'o-', color='C0', ms=3, label='flow collocation points')
    ax.plot(node_y, cla_node, 'o', color='darkgray', ms=5, label='FEM nodes')
    ax.set_xlabel('spanwise location [m]')
    ax.set_ylabel("CL_alpha")
    ax.legend()
    fig.savefig('CLa.pdf', bbox_inches='tight')

    # total lift force
    loads = prob.get_val('traction_forces').reshape(9, n_node_fullspan, order='F')
    lift = np.sum(loads[2, :])  # sum of all lift forces
    print("Total lift force:", float(lift))

    plt.show()
    quit()

    # --- Check partials after you've solve the system!! ---
    starttime = time.time()
    if args.test_partials:

        # print("computing total derivatives...\n" + "-" * 50)
        # prob.compute_totals()

        np.set_printoptions(linewidth=1000, precision=4)

        fileName = "partials.out"
        f = open(fileName, "w")
        f.write("PARTIALS\n")
        # prob.set_check_partial_options(wrt=[""],)
        f.write("=" * 50)
        f.write("\n liftingline partials\n")
        f.write("=" * 50)
        prob.check_partials(
            out_stream=f,
            includes=["liftingline_funcs"],
            method="fd",
            step=1e-4,
            compact_print=True,
        )
        prob.check_partials(
            out_stream=f,
            includes=["liftingline_funcs"],
            method="fd",
            step=1e-4,
            # compact_print=True,
        )
        # prob.check_partials(
        #     out_stream=f,
        #     includes=["liftingline"],
        #     method="fd",
        #     step=1e-4,  # now we're cooking :)
        #     compact_print=False,
        # )
        # f.write("=" * 50)
        # f.write("\n structural partials \n")
        # f.write("=" * 50)
        # prob.check_partials(
        #     out_stream=f,
        #     includes=["beamstruct"],
        #     method="fd",
        #     # compact_print=True,
        # )
        # prob.check_partials(
        #     out_stream=f,
        #     includes=["beamstruct_funcs"],
        #     method="fd",
        #     # compact_print=True,
        # )  # THESE ARE GOOD
        # breakpoint()
        f.close()

        endtime = time.time()
        print(f"partials testing time: {endtime-starttime}")

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
