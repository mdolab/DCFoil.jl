# --- Python 3.11 ---
"""
@File          :   run_OMDCfoil.py
@Date created  :   2025/05/22
@Last modified :   2025/05/26
@Author        :   Galen Ng
@Desc          :   Run and optimize composite hydrofoil using DCFoil and OpenMDAO.
                   Requires pyGeo so you need to be in docker if on macOS
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import json
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import juliacall

from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
# import niceplots as nplt
import setup_OMdvgeo
from pygeo.mphys import OM_DVGEOCOMP
import openmdao.api as om
from multipoint import Multipoint

# import top-level OpenMDAO group that contains all components
from coupled_analysis import CoupledAnalysis

# ==============================================================================
#                         OMDCFoil setup
# ==============================================================================
jl = juliacall.newmodule("DCFoil")

jl.include("../src/io/MeshIO.jl")  # mesh I/O for reading inputs in
jl.include("../src/struct/beam_om.jl")  # discipline 1
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2
jl.include("../src/loadtransfer/ldtransfer_om.jl")  # coupling components
jl.include("../src/solvers/solveflutter_om.jl")  # discipline 4 flutter solver
jl.include("../src/solvers/solveforced_om.jl")  # discipline 5 forced solver

OUTPUTDIR = "test-opt"
files = {}
files["gridFile"] = [
    f"../INPUT/mothrudder_foil_stbd_mesh.dcf",
    f"../INPUT/mothrudder_foil_port_mesh.dcf",
]
FFDFile = "../test/dcfoil_opt/INPUT/mothrudder_ffd.xyz"
files["FFDFile"] = FFDFile

Grid = jl.DCFoil.add_meshfiles(files["gridFile"], {"junction-first": True})

# This chunk of code is just to initialize DCFoil properly. If you want to change DVs for the code, do it via OpenMDAO
nNodes = 5
nNodesStrut = 3
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
    "debug": False,
    "writeTecplotSolution": True,
    "outputDir": OUTPUTDIR,
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
    "toc": 0.12 * np.ones(nNodes),  # thickness-to-chord ratio
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
ptVec, m, n = jl.FEMMethods.unpack_coords(Grid.LEMesh, Grid.TEMesh)
nodeConn = np.array(Grid.nodeConn)
solverOptions = jl.FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[0])

dvDictInfo = {  # dictionary of design variable parameters
    "twist": {
        "lower": -15.0,
        "upper": 15.0,
        "scale": 1.0,
        "value": np.zeros(8 // 2),
    },
    "sweep": {
        "lower": 0.0,
        "upper": 30.0,
        "scale": 1,
        "value": 0.0,
    },
    # "dihedral": { # THIS DOES NOT WORK
    #     "lower": -5.0,
    #     "upper": 5.0,
    #     "scale": 1,
    #     "value": 0.0,
    # },
    "taper": { # the tip chord can change, but not the root
        "lower": [1.0, 0.5],
        "upper": [1.0, 1.1],
        "scale": 1.0,
        "value": np.ones(2) * 1.0,
    },
    "span": {
        "lower": -0.1,
        "upper": 0.1,
        "scale": 1,
        "value": 0.0,
    },
}
otherDVs = {
    "alfa0": {
        "lower": -10.0,
        "upper": 10.0,
        "scale": 1.0,  # the scale was messing with the DV bounds
        "value": appendageParams["alfa0"],
    },
    "toc": {
        "lower": 0.09,
        "upper": 0.18,
        "scale": 1.0,
        "value": appendageParams["toc"],
    },
    "theta_f": {
        "lower": np.deg2rad(-20),
        "upper": np.deg2rad(20),
        "scale": 1.0,
        "value": 0.0,
    },
}
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
# ==============================================================================
#                         Helper functions
# ==============================================================================
class Top(Multipoint):
    """
    This class does all the problem setup for the OpenMDAO model
    """

    def setup(self):

        # ************************************************
        #     Add subsystems
        # ************************************************
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.add_subsystem("mesh", om.IndepVarComp())

        self.add_subsystem(
            "geometry",
            OM_DVGEOCOMP(file=files["FFDFile"], type="ffd"),
            promotes=["twist", "sweep", "dihedral", "taper", "span"],
        )

        # --- Add core physics engine ---
        physModel = CoupledAnalysis(
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
        self.add_subsystem("dcfoil", physModel)
        self.mesh.add_output("x_ptVec0", val=ptVec, distributed=True)

        # --- Some post processing manipulation of cost functions ---
        adder = om.AddSubtractComp()
        cdComps = ["CDi", "CDpr", "CDw"]
        dragComps = ["Di", "Dpr", "Dw"]
        adder.add_equation("CD", input_names=cdComps)
        adder.add_equation("Dtot", input_names=dragComps)
        self.add_subsystem("objAdder", adder, promotes_outputs=["*"])

        # ************************************************
        #     Do connections
        # ************************************************
        self.connect("mesh.x_ptVec0", "geometry.x_ptVec_in")  # connect the mesh to the geometry parametrization
        self.connect("geometry.x_ptVec0", "dcfoil.ptVec")  # connect the geometry to the core physics engine
        # Connect nongeometry design variables to the core physics engine
        for dvName, value in otherDVs.items():
            self.connect(dvName, f"dcfoil.{dvName}")

        # Connect drag build up to total drag
        for cdName in cdComps:
            self.connect(f"dcfoil.{cdName}", f"objAdder.{cdName}")
        self.connect(f"dcfoil.Dpr", f"objAdder.Dpr")
        self.connect(f"dcfoil.Dw", f"objAdder.Dw")
        self.connect(f"dcfoil.Fdrag", f"objAdder.Di")

    def configure(self):
        """
        This method configures the Multipoint problem
        """

        self.geometry.nom_add_discipline_coords("ptVec", np.array(ptVec))  # add the ptset

        self = setup_OMdvgeo.setup(args, self, None, files)  # modify the model to have geometric design variables

        for dvName, value in dvDictInfo.items():
            self.dvs.add_output(dvName, val=value["value"])
            self.add_design_var(dvName, lower=value["lower"], upper=value["upper"], scaler=value["scale"])

        for dvName, value in otherDVs.items():
            self.dvs.add_output(dvName, val=value["value"])
            self.add_design_var(dvName, lower=value["lower"], upper=value["upper"], scaler=value["scale"])

        # --- Setup objectives and constraints ---
        # self.add_objective("CD")
        self.add_objective("Dtot")
        # self.add_constraint("dcfoil.CL", lower=0.5, upper=0.5)  # lift constraint
        self.add_constraint("dcfoil.Flift", lower=2500, upper=2550)  # lift constraint [N]
        if args.task != "trim":
            self.add_constraint("dcfoil.wtip", upper=0.05 * 0.333)  # tip deflection constraint (5% of baseline semispan)
            self.add_constraint("dcfoil.ksvent", upper=0.0)  # ventilation constraint
            # self.add_constraint("dcfoil.ksflutter", upper=0.0) # flutter constraint
            # self.add_constraint("dcfoil.vibareaw", upper=0.0) # bending vibration energy constraint


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        help="Check end of script for task type",
        type=str,
        default="run",
    )
    parser.add_argument(
        "--geovar",
        type=str,
        default="trwpd",
        help="Geometry variables to test twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
    )
    parser.add_argument("--name", type=str, default=None, help="Name of the problem to append to .sql recorder")
    parser.add_argument("--restart", type=str, default=None, help="Restart from a previous case's DVs (without .sql extension)")
    parser.add_argument("--freeSurf", action="store_true", default=False, help="Use free surface corrections")
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    # --- Trim only case ---
    if args.task == "trim":
        dvDictInfo = {  # dictionary of design variable parameters
            "sweep": {
                "lower": 0.0,
                "upper": 0.0,
                "scale": 1,
                "value": 0.0,
            },
        }
        otherDVs = {
            "alfa0": {
                "lower": -10.0,
                "upper": 10.0,
                "scale": 1.0,  # the scale was messing with the DV bounds
                "value": appendageParams["alfa0"],
            },
            "toc": {
                "lower": 0.12,
                "upper": 0.12,
                "scale": 1.0,
                "value": appendageParams["toc"],
            },
            "theta_f": {
                "lower": 0.0,
                "upper": 0.0,
                "scale": 1.0,
                "value": 0.0,
            },
        }

    if args.freeSurf:
        # --- Add free surface corrections to solver options ---
        solverOptions["use_freeSurface"] = True
        # solverOptions["use_cavitation"] = True
        # solverOptions["use_ventilation"] = True
        # solverOptions["use_dwCorrection"] = True

    prob = om.Problem()
    prob.model = Top()

    # ---------------------------
    #   Opt setup
    # ---------------------------
    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
    prob.driver.options["print_results"] = True
    prob.driver.opt_settings["Major iterations limit"] = 100
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-4
    prob.driver.opt_settings["Major optimality tolerance"] = 1e-4
    prob.driver.opt_settings["Difference interval"] = 1e-4
    prob.driver.opt_settings["Verify level"] = -1
    prob.driver.opt_settings["Function precision"] = 1e-8
    prob.driver.opt_settings["Hessian full memory"] = None
    prob.driver.opt_settings["Hessian frequency"] = 100
    prob.driver.opt_settings["Linesearch tolerance"] = 0.99
    prob.driver.opt_settings["Nonderivative linesearch"] = None
    prob.driver.opt_settings["Major Step Limit"] = 0.2
    outputDir = "output"
    prob.driver.opt_settings["Print file"] = os.path.join(outputDir, "SNOPT_print.out")
    prob.driver.opt_settings["Summary file"] = os.path.join(outputDir, "SNOPT_summary.out")

    # --- Some debug stuff ---
    prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]

    prob.setup()

    Path(OUTPUTDIR).mkdir(exist_ok=True, parents=True)

    # --- Recorder ---
    recorderName = "dcfoil.sql"
    if args.name is not None:
        recorderName = f"dcfoil-{args.name}.sql"
    print("=" * 60)
    print(f"Saving recorder to {recorderName}", flush=True)
    print("=" * 60)
    recorder = om.SqliteRecorder(recorderName)  # create recorder
    prob.add_recorder(recorder)  # attach recorder to the problem
    prob.driver.add_recorder(recorder)  # attach recorder to the driver
    model = prob.model
    model.dcfoil.add_recorder(recorder)  # attach recorder to the subsystem of interest
    # ************************************************
    #     Set starting values
    # ************************************************
    prob.set_val("theta_f", np.deg2rad(0.0))  # this is defined in [rad] in the julia wrapper layer
    prob.set_val("alfa0", 1.0)  # this is defined in [deg] in the julia wrapper layer

    # set thickness-to-chord (NACA0009)
    prob.set_val("toc", 0.12 * np.ones(nNodes))

    # initialization needed for solvers
    displacementsCol = np.zeros((6, npt_wing_full))
    prob.set_val("dcfoil.displacements_col", displacementsCol)
    prob.set_val("dcfoil.gammas", np.zeros(npt_wing_full))

    # ************************************************
    #     Other stuff
    # ************************************************
    om.n2(prob, outfile="n2.html", show_browser=False)
    # prob.set_val("sweep", 15.0) # there's a lot of lift here...maybe too much?

    # ==============================================================================
    #                         Restart from an old case
    # ==============================================================================
    if args.restart is not None:
        print("=" * 60)
        print(f"Restarting from {args.restart}", flush=True)
        print("=" * 60)
        datafname = f"./run_OMDCfoil_out/{args.restart}.sql"
        cr = om.CaseReader(datafname)

        driver_cases = cr.list_cases("driver", recurse=False, out_stream=None)

        # --- pickup last case ---
        last_case = cr.get_case(driver_cases[-1])

        objectives = last_case.get_objectives()
        design_vars = last_case.get_design_vars()
        constraints = last_case.get_constraints()
        print("obj:\t", objectives["Dtot"])

        # --- set all the design vars properly now ---
        for dv, val in design_vars.items():
            if dv in dvDictInfo:
                prob.set_val(dv, val)
            elif dv in otherDVs:
                prob.set_val(dv, val)
            else:
                print(f"WARNING: {dv} not found in dvDictInfo or otherDVs, skipping...")


    # ==============================================================================
    #                         TASKS
    # ==============================================================================

    # ************************************************
    #     OPTIMIZATION
    # ************************************************
    if args.task == "opt":
        print("=" * 60)
        print("Running optimization...", flush=True)
        print("=" * 60)
        prob.run_model()
        prob.run_driver()
        prob.record("final_state")

        print("=" * 20)
        print("Design variables:")
        print("=" * 20)
        dvs = prob.driver.get_design_var_values()
        for dvName, value in dvs.items():
            print(f"{dvName}: {value}")

        print("=" * 20)
        print("Objectives:")
        print("=" * 20)
        obj = prob.driver.get_objective_values()
        for objName, value in obj.items():
            print(f"{objName}: {value}")

        print("=" * 20)
        print("Constraints:")
        print("=" * 20)
        con = prob.driver.get_constraint_values()
        for conName, value in con.items():
            print(f"{conName}: {value}")

        # I also want to know the drag components
        print("=" * 20)
        print("Drag components:")
        print("=" * 20)
        print("CDi", prob.get_val("dcfoil.CDi"))
        print("CDpr", prob.get_val("dcfoil.CDpr"))
        print("CDw", prob.get_val("dcfoil.CDw"))
        print("Dtot", prob.get_val("Dtot"))
        print("Fdrag", prob.get_val("dcfoil.Fdrag"))
        print("Dpr", prob.get_val("dcfoil.Dpr"))
        print("Dw", prob.get_val("dcfoil.Dw"))

    if args.task in ["run"]:
        print("=" * 60)
        print("Running analysis...", flush=True)
        print("=" * 60)
        prob.run_model()

        print("=" * 20)
        print("Objectives:")
        print("=" * 20)
        obj = prob.driver.get_objective_values()
        for objName, value in obj.items():
            print(f"{objName}: {value}")

        print("=" * 20)
        print("Constraints:")
        print("=" * 20)
        con = prob.driver.get_constraint_values()
        for conName, value in con.items():
            print(f"{conName}: {value}")

        # I also want to know the drag components
        print("=" * 20)
        print("Drag components:")
        print("=" * 20)
        print("CDi", prob.get_val("dcfoil.CDi"))
        print("CDpr", prob.get_val("dcfoil.CDpr"))
        print("CDw", prob.get_val("dcfoil.CDw"))
        print("Dtot", prob.get_val("Dtot"))
        print("Fdrag", prob.get_val("dcfoil.Fdrag"))
        print("Dpr", prob.get_val("dcfoil.Dpr"))
        print("Dw", prob.get_val("dcfoil.Dw"))

    if args.task == "deriv":
        prob.run_model()
        fileName = "derivative-check-full.out"
        f = open(fileName, "w")
        # print("Checking partials...")
        # f.write("PARTIALS\n")
        # prob.check_partials(out_stream=f, method="fd", includes=,compact_print=True)
        # prob.check_partials(out_stream=f, method="fd")

        print("Checking totals...")
        f.write("TOTALS\n")
        prob.check_totals(out_stream=f, method="fd", compact_print=True)
        prob.check_totals(out_stream=f, method="fd", step=1e-4)
