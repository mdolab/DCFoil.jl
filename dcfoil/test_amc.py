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

# import top-level OpenMDAO group that contains all components
from coupled_analysis import CoupledAnalysis


outputDir = "output"
files = {
    "gridFile": [
        "../INPUT/amc_foil_stbd_mesh.dcf",
        "../INPUT/amc_foil_port_mesh.dcf", # only add this if config is full wing
    ]
}
Grid = jl.DCFoil.add_meshfiles(files["gridFile"], {"junction-first": True})
# Unpack for this code. Remember Julia is transposed from Python
LECoords = np.array(Grid.LEMesh).T
TECoords = np.array(Grid.TEMesh).T
nodeConn = np.array(Grid.nodeConn)
ptVec, m, n = jl.FEMMethods.unpack_coords(Grid.LEMesh, Grid.TEMesh)
nNodes = 10
nNodesStrut = 3
appendageOptions = {
    "compName": "rudder",
    "config": "full-wing",
    # "config": "wing",
    "nNodes": nNodes,
    "nNodeStrut": nNodesStrut,
    "use_tipMass": False,
    "xMount": 0.0,
    # "material": "al6061",
    # "strut_material": "al6061",
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
    "outputDir": outputDir,
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList": appendageList,
    "gravityVector": [0.0, 0.0, -9.81],
    "correct_xsect": True,
    # ---------------------------
    #   Flow
    # ---------------------------
    "Uinf": 11.4,  # free stream velocity [m/s]
    "rhof": 998.0,  # fluid density [kg/m³]
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
    "toc": 0.09 * np.ones(nNodes),  # thickness-to-chord ratio (max t/c if using airfoil correction)
    "x_ab": 0 * np.ones(nNodes),  # static imbalance [m]
    "theta_f": np.deg2rad(5.0),  # fiber angle global [rad]
    # --- Strut vars ---
    "depth0": 0.4,  # submerged depth of strut [m] # from Yingqian
    "rake": 0.0,  # rake angle about top of strut [deg]
    "beta": 0.0,  # yaw angle wrt flow [deg]
    "s_strut": 1.0,  # [m]
    "c_strut": 0.14 * np.ones(nNodesStrut),  # chord length [m]
    "toc_strut": 0.095 * np.ones(nNodesStrut),  # thickness-to-chord ratio (max t/c if using airfoil correction)
    "ab_strut": 0.0 * np.ones(nNodesStrut),  # dist from midchord to EA [m]
    "x_ab_strut": 0.0 * np.ones(nNodesStrut),  # static imbalance [m]
    "theta_f_strut": np.deg2rad(0),  # fiber angle global [rad]
}

# Need to set struct damping once at the beginning to avoid optimization taking advantage of changing beta
solverOptions = jl.FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[0])

# number of strips and FEM nodes
if appendageOptions["config"] == "full-wing":
    npt_wing = jl.LiftingLine.NPT_WING
    npt_wing_full = jl.LiftingLine.NPT_WING
    n_node = nNodes * 2 - 1   # for full span
else:
    npt_wing = jl.LiftingLine.NPT_WING / 2   # for half wing
    npt_wing_full = jl.LiftingLine.NPT_WING   # full span
    # check if npt_wing is integer
    if npt_wing % 1 != 0:
        raise ValueError("NPT_WING must be an even number for symmetric analysis")
    npt_wing = int(npt_wing)
    n_node = nNodes


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
def main(theta_fiber, alfa0, initialize=True, plot=False):
    """
    Run static hydroelastic analysis and compute deflections

    Parameters
    ----------
    theta_fiber : float
        Fiber angle [deg]
    alfa0 : float
        Angle of attack [deg]
        NOTE: This will override the alfa0 in appendageParams
    initialize : bool
        If True, initialize displacement and gamma to zeros

    Returns
    -------
    dz_tip : float
        Tip out-of-plane bending deflection [m]
    tz_twist : float
        Tip twist deflection [deg]
    """

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

    # ************************************************
    #     Setup problem
    # ************************************************
    model = CoupledAnalysis(
        analysis_mode="coupled",
        include_flutter=False,
        ptVec_init=np.array(ptVec),
        npt_wing=npt_wing,
        n_node=n_node,
        appendageOptions=appendageOptions,
        appendageParams=appendageParams,
        nodeConn=nodeConn,
        solverOptions=solverOptions,
    )

    prob = om.Problem(model)

    # prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options["optimizer"] = "SLSQP"
    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
    prob.driver.options['print_results'] = True
    prob.driver.opt_settings["Major iterations limit"] = 100
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-4
    prob.driver.opt_settings["Major optimality tolerance"] = 1e-4
    prob.driver.opt_settings["Difference interval"] = 1e-4,
    prob.driver.opt_settings["Verify level"] = -1
    prob.driver.opt_settings["Function precision"] = 1e-8
    prob.driver.opt_settings["Hessian full memory"] = None
    prob.driver.opt_settings["Hessian frequency"] = 100
    outputDir = "output"
    prob.driver.opt_settings["Print file"] = os.path.join(outputDir, "SNOPT_print.out")
    prob.driver.opt_settings["Summary file"] = os.path.join(outputDir, "SNOPT_summary.out")
    # prob.driver.opt_settings["Linesearch tolerance"] = 0.99
    # prob.driver.opt_settings["Nonderivative linesearch"] = None
    # prob.driver.opt_settings["Major step limit"] = 5e-3

    # --- setup optimization problem ---
    prob.model.add_design_var("ptVec")

    prob.setup()

    prob.set_val("ptVec", ptVec)

    # ************************************************
    #     Set starting values
    # ************************************************
    # set sweep parameters
    prob.set_val("theta_f", np.deg2rad(theta_fiber))   # this is defined in [rad] in the julia wrapper layer
    prob.set_val("alfa0", alfa0)  # this is defined in [deg] in the julia wrapper layer

    # set thickness-to-chord (NACA0009)
    prob.set_val('toc', 0.09 * np.ones(nNodes))

    # initialization for solvers
    if initialize:
        displacementsCol = np.zeros((6, npt_wing_full))
        prob.set_val("displacements_col", displacementsCol)
        prob.set_val("gammas", np.zeros(npt_wing_full))

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
    bending = prob.get_val("deflections")[2::9]
    twist = prob.get_val("deflections")[4::9]
    CL = prob.get_val("CL")
    print('----------------------------------')
    print("alfa0", prob.get_val("alfa0"), "deg")
    print("fiber angle", prob.get_val("theta_f"), "rad")
    print("bending deflections", bending)
    print("twisting deflections", twist)
    print("induced drag force", prob.get_val("F_x"))
    print("lift force", prob.get_val("F_z"))
    print("lift coefficient", CL)
    print('----------------------------------')

    # --- plot ---
    if plot:
        om.n2(prob)

        ny = 39 if appendageOptions["config"] == "full-wing" else 20
        ptVec3D = np.array(ptVec).reshape(2, ny, 3)
        nodes = prob.get_val('nodes').swapaxes(0, 1)  # shape (3, n_nodes)
        collocationPts = prob.get_val('collocationPts')    # shape (3, n_strip)
        force_colloc = prob.get_val('forces_dist')   # shape (3, n_strip)
        force_FEM = prob.get_val('traction_forces').reshape(9, n_node, order='F')   # shape (9, n_nodes)

        import matplotlib.pyplot as plt

        # --- 3D plot ---
        z_scaler = 10   # exaggerate vertical deflections
        fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
        ax.plot(ptVec3D[:, :, 0], ptVec3D[:, :, 1], ptVec3D[:, :, 2], 'o', color='k', ms=3)
        ax.plot(nodes[0, :], nodes[1, :], nodes[2, :], 'o', color='darkgray', ms=5)
        ax.plot(collocationPts[0, :] - appendageOptions['xMount'], collocationPts[1, :], collocationPts[2, :] * z_scaler, 'o-', color='C0', ms=3)
        ax.set_aspect('equal')

        # --- top view of planform ---
        fig, ax = plt.subplots()
        ax.plot(ptVec3D[:, :, 1], ptVec3D[:, :, 0], 'o', color='k', ms=3)
        ax.plot(nodes[1, :], nodes[0, :], 'o', color='darkgray', ms=5)
        ax.plot(collocationPts[1, :], collocationPts[0, :] - appendageOptions['xMount'], 'o-', color='C0', ms=3)
        ax.set_aspect('equal')

        # --- plot displacements ---
        disp_nodes = prob.get_val('deflections').reshape(n_node, 9)
        disp_colloc = prob.get_val('displacements_col')
        if appendageOptions["config"] == "wing":
            # just use right wing
            disp_colloc = disp_colloc[:, npt_wing:]
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
        loads = prob.get_val('traction_forces').reshape(9, n_node, order='F')
        lift = np.sum(loads[2, :])  # sum of all lift forces
        print("Total lift force:", float(lift), "(not really total if config = wing)")

        plt.show()

    # return tip bending and twist deflections
    return bending[-1], twist[-1], CL


if __name__ == "__main__":
    # --- set fiber angle ---
    # NOTE: flow speed should be set in the solverOptions
    fiber_angle = 30

    dz_tip_list = []
    theta_tip_list = []
    CL_list = []
    
    alfa_list = [0, 2, 4, 6, 8, 10, 11, 12]
    for alfa0 in alfa_list:
        dz_tip, theta_tip, CL = main(fiber_angle, alfa0, initialize=True)
        ### initialize = False   # restart from previous solution. NOTE: This doesn't work!!
        dz_tip_list.append(dz_tip)
        theta_tip_list.append(theta_tip)
        CL_list.append(CL)

    # convert theta to degree
    theta_tip_list = np.rad2deg(np.array(theta_tip_list))

    # normalize dz
    dz_tip_normalized = np.array(dz_tip_list) * 2 / 0.09   # same normalization as Liao 2019. 0.09 = mean chord

    print('\n\n-----------------------------------')
    print("fiber angle [deg]", fiber_angle)
    print("alpha [deg]", alfa_list,)
    print("tip deflections (normalized)", dz_tip_normalized)
    print("tip twist [deg]", list(theta_tip_list))
    print('-----------------------------------')

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(alfa_list, dz_tip_normalized, "o-")
    axs[0].set_ylabel("tip delta z * 2 / c ")
    axs[0].grid()

    axs[1].plot(alfa_list, theta_tip_list, "o-")
    axs[1].set_xlabel("angle of attack [deg]")
    axs[1].set_ylabel("tip twist [deg]")
    axs[1].grid()

    axs[2].plot(alfa_list, CL_list, "o-")
    axs[2].set_ylabel("CL")
    axs[2].grid()

    plt.tight_layout()
    figname = f"tip_deflections_fiber{fiber_angle}deg_flow{solverOptions['Uinf']}ms.pdf"
    plt.savefig(figname, bbox_inches="tight")
    plt.show()