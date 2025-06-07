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
from datetime import date

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from tabulate import tabulate

# ==============================================================================
# Extension modules
# ==============================================================================
from SETUP import setup_OMdvgeo, setup_dcfoil, setup_opt
from SPECS.point_specs import boatSpds, Fliftstars, opdepths, alfa0
from pygeo.mphys import OM_DVGEOCOMP
import openmdao.api as om
from multipoint import Multipoint

# import top-level OpenMDAO group that contains all components
from coupled_analysis import CoupledAnalysis

# ==============================================================================
#                         Command line arguments
# ==============================================================================
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
parser.add_argument("--debug", action="store_true", default=False, help="Debug the flutter runs")
parser.add_argument("--pts", type=str, default="3", help="Performance point IDs to run, e.g., 3 is p3")
parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
args = parser.parse_args()

# --- Echo the args ---
print(30 * "-")
print("Arguments are", flush=True)
for arg in vars(args):
    print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
print(30 * "-", flush=True)

# ==============================================================================
#                         Settings
# ==============================================================================

files = {}
files["gridFile"] = [
    f"../INPUT/{args.foil}_foil_stbd_mesh.dcf",
    f"../INPUT/{args.foil}_foil_port_mesh.dcf",
]
FFDFile = f"../test/dcfoil_opt/INPUT/{args.foil}_ffd.xyz"
files["FFDFile"] = FFDFile

nNodes = 10
nNodesStrut = 3

dvDictInfo = {  # dictionary of design variable parameters
    "twist": {
        "lower": -15.0,
        "upper": 15.0,
        "scale": 1.0/30,
        "value": np.zeros(8 // 2),
    },
    "sweep": {
        "lower": 0.0,
        "upper": 30.0,
        "scale": 1/30.0,
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
        "upper": [1.0, 1.4],
        "scale": 1.0,
        "value": np.ones(2) * 1.0,
    },
    "span": {
        "lower": -0.2,
        "upper": 0.2,
        "scale": 1/(0.4),
        "value": 0.0,
    },
}
otherDVs = {
    "alfa0": {
        "lower": -10.0,
        "upper": 10.0,
        "scale": 1.0/(20),  # the scale was messing with the DV bounds
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
        "scale": 1.0/(np.deg2rad(60)),
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

        # --- Add core physics engine for each flight point ---
        for ptName in probList:
            print(f"Adding point {ptName} to the problem", flush=True)

            run_flutter = args.flutter and ptName == "p3"  # only run flutter for p3
            
            Uinf = boatSpds[ptName]
            depth = opdepths[ptName]
            solverOptions["Uinf"] = Uinf
            appendageParams["depth0"] = depth
            solverOptions["outputDir"] += f"/{ptName}"  # add the point name to the output directory

            # print(f"Uinf: {Uinf}, depth: {depth}", flush=True)
            flowOptions = {
                "Uinf": Uinf,
                "depth0": depth,
                "alfa0_flutter": alfa0,  # angle of attack for flutter analysis [deg]
            }


            physModel = CoupledAnalysis(
                analysis_mode="coupled",
                include_flutter=run_flutter,
                ptVec_init=np.array(ptVec),
                npt_wing=npt_wing,
                n_node=n_node,
                appendageOptions=appendageOptions,
                appendageParams=appendageParams,
                nodeConn=nodeConn,
                solverOptions=solverOptions,
                flowOptions=flowOptions,
            )
            self.add_subsystem(f"dcfoil_{ptName}", physModel)


        self.mesh.add_output("x_ptVec0", val=ptVec, distributed=True)

        # --- Some post processing manipulation of cost functions ---
        adder = om.AddSubtractComp()
        cdCompsPt = ["CDi", "CDpr", "CDw"]
        dragCompsPt = ["Di", "Dpr", "Dw"]
        cdComps = []
        dragComps = []
        for ptName in probList: # loop through all the points so all drags are added
            for cd in cdCompsPt:
                cdComps.append(f"{cd}_{ptName}")
            for drag in dragCompsPt:
                dragComps.append(f"{drag}_{ptName}")
        adder.add_equation("CD", input_names=cdComps)
        adder.add_equation("Dtot", input_names=dragComps)
        self.add_subsystem("objAdder", adder, promotes_outputs=["*"])

        # ************************************************
        #     Do connections
        # ************************************************
        self.connect("mesh.x_ptVec0", "geometry.x_ptVec_in")  # connect the mesh to the geometry parametrization

        for ptName in probList:
            self.connect("geometry.x_ptVec0", f"dcfoil_{ptName}.ptVec")  # connect the geometry to the core physics engine

            # Connect nongeometry design variables to the core physics engine
            for dvName, value in otherDVs.items():
                if dvName == "alfa0":
                    self.connect(f"{dvName}_{ptName}", f"dcfoil_{ptName}.{dvName}")
                else:
                    self.connect(dvName, f"dcfoil_{ptName}.{dvName}")

            # Connect drag build up to total drag
            for cdName in cdCompsPt:
                self.connect(f"dcfoil_{ptName}.{cdName}", f"objAdder.{cdName}_{ptName}")
            self.connect(f"dcfoil_{ptName}.Dpr", f"objAdder.Dpr_{ptName}")
            self.connect(f"dcfoil_{ptName}.Dw", f"objAdder.Dw_{ptName}")
            self.connect(f"dcfoil_{ptName}.Fdrag", f"objAdder.Di_{ptName}")

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
            if dvName == "alfa0":
                for ptName in probList:
                    self.dvs.add_output(f"{dvName}_{ptName}", val=value["value"])
                    self.add_design_var(f"{dvName}_{ptName}", lower=value["lower"], upper=value["upper"], scaler=value["scale"])
            else:
                self.dvs.add_output(dvName, val=value["value"])
                self.add_design_var(dvName, lower=value["lower"], upper=value["upper"], scaler=value["scale"])

        # ************************************************
        #     Set flow conditions
        # ************************************************
        if "p1" in probList:
            for key, value in self.dcfoil_p1.options["flowOptions"].items():
                self.dcfoil_p1.options["solverOptions"][key] = value
        if "p2" in probList:
            for key, value in self.dcfoil_p2.options["flowOptions"].items():
                self.dcfoil_p2.options["solverOptions"][key] = value
        if "p3" in probList:
            for key, value in self.dcfoil_p3.options["flowOptions"].items():
                self.dcfoil_p3.options["solverOptions"][key] = value

        # ************************************************
        #     Objectives
        # ************************************************
        # self.add_objective("CD")
        self.add_objective("Dtot")

        # ************************************************
        #     Constraints
        # ************************************************
        for ptName in probList:
            # self.add_constraint("dcfoil.CL", lower=0.5, upper=0.5)  # lift constraint
            self.add_constraint(f"dcfoil_{ptName}.Flift", lower=2500, upper=2550, scaler=1/2500)  # lift constraint [N]

            if args.task != "trim":
                self.add_constraint(f"dcfoil_{ptName}.wtip", upper=0.05 * 0.333)  # tip defl con (5% of baseline semispan)
                self.add_constraint(f"dcfoil_{ptName}.ksvent", upper=0.0)  # ventilation constraint

            # self.add_constraint("dcfoil.vibareaw", upper=0.0) # bending vibration energy constraint
            
            if args.flutter and ptName == "p3":
                self.add_constraint(f"dcfoil_{ptName}.ksflutter", upper=0.0)  # flutter constraint only for p3

def print_drags():
    for ptName in probList:
        print(f"Point {ptName}:")
        print("-" * 20)
        print("CDi", prob.get_val(f"dcfoil_{ptName}.CDi"))
        print("CDpr", prob.get_val(f"dcfoil_{ptName}.CDpr"))
        print("CDw", prob.get_val(f"dcfoil_{ptName}.CDw"))
        print("Dtot", prob.get_val("Dtot"))
        print("Fdrag", prob.get_val(f"dcfoil_{ptName}.Fdrag"))
        print("Dpr", prob.get_val(f"dcfoil_{ptName}.Dpr"))
        print("Dw", prob.get_val(f"dcfoil_{ptName}.Dw"))

# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    probList = []
    for pt in args.pts:
        ptName = f"p{pt}"
        probList.append(ptName)

    # ==============================================================================
    #                         DCFoil setup
    # ==============================================================================
    case_name = f"OUTPUT/{date.today().strftime('%Y-%m-%d')}-{args.task}-p{args.pts}"
    Path(case_name).mkdir(exist_ok=True, parents=True)

    if args.name is not None:
        case_name += "-" + args.name
    (
        ptVec,
        nodeConn,
        appendageParams,
        appendageOptions,
        solverOptions,
        npt_wing,
        npt_wing_full,
        n_node,
    ) = setup_dcfoil.setup(nNodes, nNodesStrut, args, None, files, case_name)

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
        output_dir=case_name,
    )
    prob.driver.options["hist_file"] = "dcfoil.hst"
    prob.driver.options["debug_print"] = ["desvars", "ln_cons", "nl_cons", "objs", "totals"]

    outputDir = case_name
    optOptions = setup_opt.setup(args, outputDir)
    for key, val in optOptions.items():
        prob.driver.opt_settings[key] = val


    prob.setup()
    print("Problem setup complete!", flush=True)


    # --- Recorder ---
    recorderName = f"{Path(__file__).parent.resolve()}/dcfoil.sql" # weird bug that OUTPUT can't be written into, but whatever
    if args.name is not None:
        recorderName = f"{Path(__file__).parent.resolve()}/dcfoil-{args.name}.sql" # weird bug that OUTPUT can't be written into, but whatever
    print("=" * 60)
    print(f"Saving recorder to {recorderName}", flush=True)
    print("=" * 60)
    recorder = om.SqliteRecorder(recorderName)  # create recorder
    prob.add_recorder(recorder)  # attach recorder to the problem
    prob.driver.add_recorder(recorder)  # attach recorder to the driver
    model = prob.model
    model.geometry.add_recorder(recorder)  # attach recorder to the geometry subsystem


    # attach recorder to the subsystem of interest
    if "p1" in probList:
        model.dcfoil_p1.add_recorder(recorder)  
    if "p2" in probList:
        model.dcfoil_p2.add_recorder(recorder)
    if "p3" in probList:
        model.dcfoil_p3.add_recorder(recorder)

    # ************************************************
    #     Set starting values
    # ************************************************
    prob.set_val("theta_f", np.deg2rad(0.0))  # this is defined in [rad] in the julia wrapper layer
    for ptName in probList:
        prob.set_val(f"alfa0_{ptName}", alfa0)  # this is defined in [deg] in the julia wrapper layer

    # set thickness-to-chord (NACA0009)
    prob.set_val("toc", 0.12 * np.ones(nNodes))

    # initialization needed for solvers
    displacementsCol = np.zeros((6, npt_wing_full))
    for ptName in probList:
        prob.set_val(f"dcfoil_{ptName}.displacements_col", displacementsCol)
        prob.set_val(f"dcfoil_{ptName}.gammas", np.zeros(npt_wing_full))

    # ************************************************
    #     Other stuff
    # ************************************************
    om.n2(prob, outfile=f"n2-{args.name}.html", show_browser=False)

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
        # Don't forget scales
        for dv, val in design_vars.items():
            if dv in dvDictInfo:
                print(f"Setting {dv} to {val} but scaled")
                scale = dvDictInfo[dv]["scale"]
                prob.set_val(dv, val/scale)
            elif dv in otherDVs:
                try:
                    print(f"Setting {dv} to {val} but scaled")
                    scale = otherDVs[dv]["scale"]
                    prob.set_val(dv, val/scale)
                except KeyError:
                    breakpoint()
                    print(f"WARNING: {dv} not found in prob, skipping...")
            elif dv.startswith("alfa0_"):
                print(f"Setting {dv} to {val} but scaled")
                scale = otherDVs["alfa0"]["scale"]
                prob.set_val(dv, val/scale)
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
        print_drags()

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
        print_drags()

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
