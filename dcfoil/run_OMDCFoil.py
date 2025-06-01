# --- Python 3.11 ---
"""
@File          :   run_OMDCFoil.py
@Date created  :   2025-05-22
@Last modified :   2025-05-30
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
from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
from SETUP import setup_OMdvgeo, setup_dcfoil, setup_opt
from pygeo.mphys import OM_DVGEOCOMP
import openmdao.api as om
from multipoint import Multipoint

# import top-level OpenMDAO group that contains all components
from coupled_analysis import CoupledAnalysis

# ==============================================================================
#                         Settings
# ==============================================================================

DCFoilOutputDir = "run_OMDCFoil_out/dcfoil"
files = {}
files["gridFile"] = [
    f"../INPUT/mothrudder_foil_stbd_mesh.dcf",
    f"../INPUT/mothrudder_foil_port_mesh.dcf",
]
FFDFile = "../test/dcfoil_opt/INPUT/mothrudder_ffd.xyz"
files["FFDFile"] = FFDFile

nNodes = 10
nNodesStrut = 3

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
    "taper": {  # the tip chord can change, but not the root
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
        "value": 4.0,
    },
    "toc": {
        "lower": 0.09,
        "upper": 0.18,
        "scale": 1.0,
        "value": 0.10 * np.ones(nNodes),
    },
    "theta_f": {
        "lower": np.deg2rad(-30),
        "upper": np.deg2rad(30),
        "scale": 1.0,
        "value": 0.0,
    },
}

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
            include_flutter=args.flutter,
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
            self.add_constraint("dcfoil.wtip", upper=0.05 * 0.333)  # tip defl con (5% of baseline semispan)
            self.add_constraint("dcfoil.ksvent", upper=0.0)  # ventilation constraint
        if args.flutter:
            self.add_constraint("dcfoil.ksflutter", upper=0.0)  # flutter constraint

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
    parser.add_argument("--optimizer", type=str, default="SNOPT", help="What type of optimizer?")
    parser.add_argument(
        "--restart", type=str, default=None, help="Restart from a previous case's DVs (without .sql extension)"
    )
    parser.add_argument("--freeSurf", action="store_true", default=False, help="Use free surface corrections")
    parser.add_argument("--flutter", action="store_true", default=False, help="Run flutter analysis")
    parser.add_argument("--fixStruct", action="store_true", default=False, help="Fix the structure design variables")
    parser.add_argument("--fixHydro", action="store_true", default=False, help="Fix the hydro design variables")
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    # ==============================================================================
    #                         DCFoil setup
    # ==============================================================================
    if args.name is not None:
        DCFoilOutputDir += "-" + args.name
    (
        ptVec,
        nodeConn,
        appendageParams,
        appendageOptions,
        solverOptions,
        npt_wing,
        npt_wing_full,
        n_node,
    ) = setup_dcfoil.setup(nNodes, nNodesStrut, args, None, files, DCFoilOutputDir)

    # --- Trim only case ---
    if args.task == "trim":
        thetaStart = np.deg2rad(0.0)
        dvDictInfo = {  # dictionary of design variable parameters
            "sweep": {
                "lower": 0.0,
                "upper": 0.0,
                "scale": 1,
                "value": 0.0,
            },
            "twist": {
                "lower": 0.0,
                "upper": 0.0,
                "scale": 1.0,
                "value": np.zeros(8 // 2),
            },
            "taper": {  # the tip chord can change, but not the root
                "lower": [1.0, 1.0],
                "upper": [1.0, 1.0],
                "scale": 1.0,
                "value": np.ones(2) * 1.0,
            },
            "span": {
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
                "lower": appendageParams["toc"],
                "upper": appendageParams["toc"],
                "scale": 1.0,
                "value": appendageParams["toc"],
            },
            "theta_f": {
                "lower": thetaStart,
                "upper": thetaStart,
                "scale": 1.0,
                "value": thetaStart,
            },
        }
    if args.fixStruct:
        thetaStart = np.deg2rad(0.0)
        otherDVs = {
            "alfa0": {
                "lower": -10.0,
                "upper": 10.0,
                "scale": 1.0,  # the scale was messing with the DV bounds
                "value": appendageParams["alfa0"],
            },
            "toc": {
                "lower": appendageParams["toc"],
                "upper": appendageParams["toc"],
                "scale": 1.0,
                "value": appendageParams["toc"],
            },
            "theta_f": {
                "lower": np.deg2rad(-30),
                "upper": np.deg2rad(30),
                "scale": 1.0,
                "value": thetaStart,
            },
        }
    if args.fixHydro:
        # Free the DVs that seem to be giving problems for the flutter optimization
        print("WARNING: Fixing hydro design variables")
        dvDictInfo = {
            "twist": {
                "lower": 0.0,
                "upper": 0.0,
                "scale": 1.0,
                "value": np.zeros(8 // 2),
            },
            "sweep": {
                "lower": 0.0,
                "upper": 30.0,
                "scale": 1,
                "value": 0.0,
            },
            "taper": {  # the tip chord can change, but not the root
                "lower": [1.0, 1.0],
                "upper": [1.0, 1.0],
                "scale": 1.0,
                "value": np.ones(2) * 1.0,
            },
            "span": {
                "lower": 0.0,
                "upper": 0.0,
                "scale": 1,
                "value": 0.0,
            },
        }

    prob = om.Problem()
    prob.model = Top()

    # ---------------------------
    #   Opt setup
    # ---------------------------
    prob.driver = om.pyOptSparseDriver(
        title=f"{args.name}",
        optimizer="SNOPT",
        print_results=True,
        print_opt_prob=True,
    )
    prob.driver.options["hist_file"] = "dcfoil.hst"
    prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs"]
    outputDir = "output"
    optOptions = setup_opt.setup(args, outputDir)
    for key, val in optOptions.items():
        prob.driver.opt_settings[key] = val

    prob.setup()

    Path(DCFoilOutputDir).mkdir(exist_ok=True, parents=True)

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
    model.geometry.add_recorder(recorder)  # attach recorder to the geometry subsystem
    # ************************************************
    #     Set starting values
    # ************************************************
    prob.set_val("theta_f", np.deg2rad(0.0))  # this is defined in [rad] in the julia wrapper layer
    prob.set_val("alfa0", 2.0)  # this is defined in [deg] in the julia wrapper layer

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
    if args.task in ["opt", "trim"]:
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
        print("Checking totals...")
        f.write("TOTALS\n")
        prob.check_totals(out_stream=f, method="fd", compact_print=True)
        prob.check_totals(out_stream=f, method="fd", step=1e-4)

        print("Checking partials...")
        f.write("PARTIALS\n")
        prob.model.dcfoil.hydroelastic.liftingline_funcs.set_check_partial_options(wrt=["toc"])
        prob.check_partials(method="fd", includes=["liftingline_funcs"], compact_print=True)
        prob.check_partials(method="fd", includes=["liftingline_funcs"])
