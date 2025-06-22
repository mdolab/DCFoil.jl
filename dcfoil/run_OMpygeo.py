# --- Python 3.10 ---
"""
@File          :   test_OMpygeo.py
@Date created  :   2025/03/04
@Last modified :   2025/05/22
@Author        :   Galen Ng
@Desc          :   Test pygeo as an openmdao component. This is similar to a check embedding script
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import argparse
from pathlib import Path
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
# import niceplots as nplt
from SETUP import setup_OMdvgeo, setup_dcfoil
from pygeo.mphys import OM_DVGEOCOMP
import openmdao.api as om
from multipoint import Multipoint

parser = argparse.ArgumentParser()
parser.add_argument(
    "--geovar",
    type=str,
    default="trwpd",
    help="Geometry variables to test twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
)
parser.add_argument("--animate", default=False, action="store_true")
parser.add_argument("--freeSurf", action="store_true", default=False, help="Use free surface corrections")
parser.add_argument("--flutter", action="store_true", default=False, help="Run flutter analysis")
parser.add_argument("--forced", action="store_true", default=False, help="Run forced vibration analysis")
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

outputDir = "embedding"
files = {}
files["gridFile"] = [
    f"../INPUT/{args.foil}_foil_stbd_mesh.dcf",
    f"../INPUT/{args.foil}_foil_port_mesh.dcf",
]
if "moth" in args.foil:
    FFDFile = f"../test/dcfoil_opt/INPUT/{args.foil}_ffd.xyz"
else:
    FFDFile = f"../test/dcfoil_opt/INPUT/{args.foil}-1buffer_ffd.xyz"
files["FFDFile"] = FFDFile


nNodes = 10
nNodesStrut = 5
case_name = "embedding"
(
    ptVec,
    nodeConn,
    appendageParams,
    appendageOptions,
    solverOptions,
    npt_wing,
    npt_wing_full,
    n_node,
) = setup_dcfoil.setup(nNodes, nNodesStrut, args, None, files, [0.01, 1.0], case_name)

# nnodes = len(ptVec) // 3 // 2
# XCoords = ptVec.reshape(-1, 3)
# LEcoords = XCoords[:nnodes]
# TEcoords = XCoords[nnodes:]

nTwist = 8 // 2 + 2
nTwist = 8 // 2 # with 1 buffer 
dvDictInfo = {  # dictionary of design variable parameters
    "twist": {
        "lower": -15.0,
        "upper": 15.0,
        "scale": 1.0,
        "value": np.zeros(nTwist),
    },
    "sweep": {
        "lower": 0.0,
        "upper": 30.0,
        "scale": 1,
        "value": 0.0,
    },
    "dihedral": {
        "lower": -5.0,
        "upper": 5.0,
        "scale": 1,
        "value": 0.0,
    },
    "taper": {
        "lower": [-0.1, -0.1],
        "upper": [0.1, 0.1],
        "scale": 1.0,
        "value": np.ones(2) * 1.0,
    },
    "span": {
        "lower": -0.1,
        "upper": 0.1,
        "scale": 1 / 0.2,
        "value": 0.0,
    },
}


class Top(Multipoint):
    """
    This class does geometry stuff
    """

    def setup(self):
        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.add_subsystem("mesh", om.IndepVarComp())

        self.mesh.add_output("x_ptVec0", val=ptVec, distributed=True)

        self.add_subsystem(
            "geometry",
            OM_DVGEOCOMP(file=files["FFDFile"], type="ffd"),
            promotes=["twist", "sweep", "dihedral", "taper", "span"],
        )

        self.connect("mesh.x_ptVec0", "geometry.x_ptVec_in")

    def configure(self):
        self.geometry.nom_add_discipline_coords("ptVec", np.array(ptVec))

        self = setup_OMdvgeo.setup(args, self, None, files)

        for dvName, value in dvDictInfo.items():
            self.dvs.add_output(dvName, val=value["value"])
            self.add_design_var(dvName, lower=value["lower"], upper=value["upper"], scaler=value["scale"])

        # self.add_objective("x_leptVec0")


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    prob = om.Problem()
    prob.model = Top()

    # ---------------------------
    #   Opt setup
    # ---------------------------
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

    prob.setup()

    Path(outputDir).mkdir(exist_ok=True, parents=True)

    om.n2(prob, outfile="n2.html", show_browser=False)

    # --- Run this model with no sweep ---
    prob.set_val("sweep", 0.0)
    prob.run_model()
    # print("ptVec_in")
    # print("LE coords")
    # print(prob.get_val("geometry.x_leptVec0").reshape(-1, 3))
    # print("TE coords")
    # print(prob.get_val("geometry.x_teptVec0").reshape(-1, 3))
    # om.n2(prob, outfile=f"n2_before.html", show_browser=False)

    # --- run model again with sweep ---
    prob.set_val("sweep", 0.0)
    # print("sweep angle:\t", prob.get_val("sweep"))

    prob.final_setup()
    prob.run_model()

    # print("ptVec_after")
    # print("LE coords")
    # print(prob.get_val("geometry.x_leptVec0").reshape(-1, 3))
    # print("TE coords")
    # print(prob.get_val("geometry.x_teptVec0").reshape(-1, 3))
    # om.n2(prob, outfile=f"n2_after.html", show_browser=False)
    # print("ptVec_0")
    # print(prob.get_val("ptVec_0"))

    if args.animate:
        outputDir = "embedding"

        # ---------------------------
        #   TWIST
        # ---------------------------
        if "t" in args.geovar:
            print("+", 20 * "-", "Demo twist", 20 * "-", "+")
            dirName = f"{outputDir}/demo_twist/"
            # if comm.rank == 0:
            Path(dirName).mkdir(exist_ok=True, parents=True)
            n_twist = 20  # number of increments

            mag = 15.0
            wave = mag * np.sin(np.linspace(0, 2 * np.pi, n_twist))

            i_frame = 0
            for ii in range(nTwist):
                print("index: ", ii)
                # for ind, val in enumerate(twist_vals):
                for val in wave:
                    prob.set_val("twist", indices=ii, val=val)
                    prob.run_model()
                    DVGeo = prob.model.geometry.nom_getDVGeo()
                    # dvDict = DVGeo.getValues()
                    # dvDict["twist"][ii] = val
                    # DVGeo.setDesignVars(dvDict)

                    # Write deformed FFD
                    DVGeo.writeTecplot(f"{dirName}/twist_{i_frame:03d}_ffd.dat")
                    DVGeo.writeRefAxes(f"{dirName}/twist_{i_frame:03d}_axes")

                    print("Writing ptSets to tecplot...")

                    ptSetName = "x_ptVec0"
                    DVGeo.writePointSet(ptSetName, f"{dirName}/twist_{i_frame:03d}", solutionTime=i_frame)
                    # ptSetName = "x_leptVec0"
                    # DVGeo.writePointSet(ptSetName, f"{dirName}/twist_{i_frame:03d}", solutionTime=i_frame)
                    # ptSetName = "x_teptVec0"
                    # DVGeo.writePointSet(ptSetName, f"{dirName}/twist_{i_frame:03d}", solutionTime=i_frame)

                    i_frame += 1

        # ---------------------------
        #   SPAN
        # ---------------------------
        if "p" in args.geovar:
            print("+", 20 * "-", "Demo span", 20 * "-", "+")
            dirName = f"{outputDir}/demo_span/"
            # if comm.rank == 0:
            Path(dirName).mkdir(exist_ok=True, parents=True)

            n_span = 60
            wave = np.sin(np.linspace(0, np.pi, n_span)) * 0.1

            i_frame = 0
            # loop over wave
            for ind, val in enumerate(wave):
                print(ind, val)

                prob.set_val("span", val)
                prob.run_model()
                DVGeo = prob.model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/span_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/span_{i_frame:03d}_axes")

                print("Writing ptSets to tecplot...")

                ptSetName = "x_ptVec0"
                DVGeo.writePointSet(ptSetName, f"{dirName}/span_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_leptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/span_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_teptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/span_{i_frame:03d}", solutionTime=i_frame)

                i_frame += 1

        # ---------------------------
        #   SWEEP
        # ---------------------------
        if "w" in args.geovar:
            dirName = f"{outputDir}/demo_sweep/"
            # if comm.rank == 0:
            print("+", 20 * "-", "Demo sweep", 20 * "-", "+")
            Path(dirName).mkdir(exist_ok=True, parents=True)

            n_sweep = 30
            mag = 30.0
            wave = mag * np.sin(np.linspace(0, 2 * np.pi, n_sweep))

            i_frame = 0

            sweep_vals = wave

            for ind, val in enumerate(sweep_vals):
                print("index:", ind, val)
                prob.set_val("sweep", val)
                prob.run_model()
                DVGeo = prob.model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/sweep_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/sweep_{i_frame:03d}_axes")

                print("Writing ptSets to tecplot...")

                ptSetName = "x_ptVec0"
                DVGeo.writePointSet(ptSetName, f"{dirName}/sweep_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_leptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/sweep_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_teptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/sweep_{i_frame:03d}", solutionTime=i_frame)

                i_frame += 1

        # ---------------------------
        #   TAPER
        # ---------------------------
        if "r" in args.geovar:
            dirName = f"{outputDir}/demo_taper/"
            # if comm.rank == 0:
            print("+", 20 * "-", "Demo taper", 20 * "-", "+")
            Path(dirName).mkdir(exist_ok=True, parents=True)

            n_taper = 60
            wave = np.sin(np.linspace(0, 2 * np.pi, n_taper)) * 0.5 + 1.0

            i_frame = 0
            # loop over wave
            for ind, val in enumerate(wave):
                print(ind, val)
                prob.set_val("taper", val=val, indices=[-1])
                prob.run_model()
                DVGeo = prob.model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/taper_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/taper_{i_frame:03d}_axes")

                print("Writing ptSets to tecplot...")

                ptSetName = "x_ptVec0"
                DVGeo.writePointSet(ptSetName, f"{dirName}/taper_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_leptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/taper_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_teptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/taper_{i_frame:03d}", solutionTime=i_frame)

                i_frame += 1

        # ---------------------------
        #   All vars
        # ---------------------------
        if "t" and "r" and "w" and "p" in args.geovar:
            dirName = f"{outputDir}/demo_all/"
            print("+", 20 * "-", "Demo all", 20 * "-", "+")
            Path(dirName).mkdir(exist_ok=True, parents=True)

            n_all = 20
            wave = np.sin(np.linspace(0, np.pi / 2, n_all))

            taper_mag = 0.5
            twist_mag = 15.0
            sweep_mag = 30.0
            span_mag = 0.2

            i_frame = 0

            # --- SPAN ---
            # wave = np.sin(np.linspace(0, np.pi, n_all))
            for ind, val in enumerate(wave):
                print(ind, val)

                prob.set_val("span", val=val * span_mag)
                prob.run_model()
                DVGeo = prob.model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                print("Writing ptSets to tecplot...")

                ptSetName = "x_ptVec0"
                DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_leptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_teptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)

                i_frame += 1

            # --- TAPER ---
            for ind, val in enumerate(wave):
                print(ind, val)
                dvDict = DVGeo.getValues()
                dvDict["taper"][-1] = val * taper_mag + 1.0
                DVGeo.setDesignVars(dvDict)
                prob.set_val("taper", val=val * taper_mag + 1.0, indices=[-1])
                prob.run_model()

                DVGeo = prob.model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                print("Writing ptSets to tecplot...")

                ptSetName = "x_ptVec0"
                DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_leptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_teptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)

                i_frame += 1

            # --- TWIST ---
            for ii in range(nTwist):
                for ind, val in enumerate(wave):
                    print(ind, val)

                    prob.set_val("twist", indices=ii, val=val * twist_mag)
                    prob.run_model()
                    DVGeo = prob.model.geometry.nom_getDVGeo()

                    # Write deformed FFD
                    DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                    DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")
                    print("Writing ptSets to tecplot...")

                    ptSetName = "x_ptVec0"
                    DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                    # ptSetName = "x_leptVec0"
                    # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                    # ptSetName = "x_teptVec0"
                    # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)

                    i_frame += 1

            # --- SWEEP ---
            for ind, val in enumerate(wave):
                print(ind, val)

                prob.set_val("sweep", val=val * sweep_mag)
                prob.run_model()
                DVGeo = prob.model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/all_{i_frame:03d}_ffd.dat")
                DVGeo.writeRefAxes(f"{dirName}/all_{i_frame:03d}_axes")

                print("Writing ptSets to tecplot...")

                ptSetName = "x_ptVec0"
                DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_leptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)
                # ptSetName = "x_teptVec0"
                # DVGeo.writePointSet(ptSetName, f"{dirName}/all_{i_frame:03d}", solutionTime=i_frame)

                i_frame += 1

    # ************************************************
    #     Check partials
    # ************************************************

    fileName = "partials-geo.out"
    f = open(fileName, "w")
    print("=" * 20)
    print("Checking partials...")
    print("=" * 20)
    prob.check_partials(out_stream=f, method="fd", compact_print=True)
    prob.check_partials(out_stream=f, method="fd")
    # prob.check_totals()
