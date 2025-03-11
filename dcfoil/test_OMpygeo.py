# --- Python 3.10 ---
"""
@File          :   test_OMpygeo.py
@Date created  :   2025/03/04
@Last modified :   2025/03/04
@Author        :   Galen Ng
@Desc          :   Test pygeo as an openmdao component
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
import setup_OMdvgeo
from pygeo.mphys import OM_DVGEOCOMP
import openmdao.api as om
from mphys.multipoint import Multipoint  # TODO: just copy this file in so there's no dependency on mphys

outputDir = "embedding"
files = {}
# files["gridFile"] = [
#         f"../test/dcfoil_opt/mothrudder_foil_stbd_mesh.dcf",
#         f"../test/dcfoil_opt/mothrudder_foil_port_mesh.dcf",
#     ]
FFDFile = "../test/dcfoil_opt/INPUT/mothrudder_ffd.xyz"
files["FFDFile"] = FFDFile

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
nnodes = len(ptVec) // 3 // 2
XCoords = ptVec.reshape(-1, 3)
LEcoords = XCoords[:nnodes]
TEcoords = XCoords[nnodes:]

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

        self.mesh.add_output("x_leptVec0", val=LEcoords.flatten(), distributed=True)
        self.mesh.add_output("x_teptVec0", val=TEcoords.flatten(), distributed=True)

        self.add_subsystem(
            "geometry",
            OM_DVGEOCOMP(file=files["FFDFile"], type="ffd"),
            promotes=["twist", "sweep", "dihedral", "taper", "span"],
        )

        self.connect("mesh.x_leptVec0", "geometry.x_leptVec_in")
        self.connect("mesh.x_teptVec0", "geometry.x_teptVec_in")

    def configure(self):

        self.geometry.nom_add_discipline_coords("leptVec", LEcoords.flatten())
        self.geometry.nom_add_discipline_coords("teptVec", TEcoords.flatten())

        self = setup_OMdvgeo.setup(args, self, None, files)

        for dvName, value in dvDictInfo.items():
            self.dvs.add_output(dvName, val=value["value"])
            self.add_design_var(dvName, lower=value["lower"], upper=value["upper"], scaler=value["scale"])


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geovar",
        type=str,
        default="trwpd",
        help="Geometry variables to test twist (t), shape (s), taper/chord (r), sweep (w), span (p), dihedral (d)",
    )
    parser.add_argument("--animate", default=False, action="store_true")
    args = parser.parse_args()

    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    # ************************************************
    #     DVGeo setup
    # ************************************************

    # model.geometry.nom_addPointSet(ptVec, ptSetName, add_output=False, DVGeoName="defaultDVGeo")
    # model.geometry.add_input(f"{ptSetName}_in", distributed=True, val=ptVec)
    # model.geometry.add_output(f"{ptSetName}_0", distributed=True, val=ptVec)

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
        # "Linesearch tolerance": 0.99,  # all gradients are known so we can do less accurate LS
        # "Nonderivative linesearch": None,  # Comment out to specify yes nonderivative (nonlinear problem)
        # "Major Step Limit": 5e-3,
        # "Major iterations limit": 1,  # NOTE: for debugging; remove before runs if left active by accident
    }

    # model.add_subsystem("geometry", OM_DVGEOCOMP(file=files["FFDFile"], type="ffd"), promotes=["*"])

    prob.setup()

    Path(outputDir).mkdir(exist_ok=True, parents=True)

    om.n2(prob, outfile="n2.html", show_browser=False)

    # --- Run this model with no sweep ---
    prob.set_val("sweep", 0.0)
    prob.run_model()
    print("ptVec_in")
    print("LE coords")
    print(prob.get_val("geometry.x_leptVec0").reshape(-1, 3))
    print("TE coords")
    print(prob.get_val("geometry.x_teptVec0").reshape(-1, 3))
    om.n2(prob, outfile=f"n2_before.html", show_browser=False)

    # --- run model again with sweep ---
    prob.set_val("sweep", 10.0)
    print("sweep angle:\t", prob.get_val("sweep"))

    prob.final_setup()
    prob.run_model()

    print("ptVec_after")
    print("LE coords")
    print(prob.get_val("geometry.x_leptVec0").reshape(-1, 3))
    print("TE coords")
    print(prob.get_val("geometry.x_teptVec0").reshape(-1, 3))
    om.n2(prob, outfile=f"n2_after.html", show_browser=False)
    # print("ptVec_0")
    # print(prob.get_val("ptVec_0"))

    if args.animate:
        # ---------------------------
        #   TWIST
        # ---------------------------
        if "t" in args.geovar:
            nTwist = 8 // 2
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
                # DVGeo.writeRefAxes(f"{dirName}/sweep_{i_frame:03d}_axes")

                i_frame += 1

    # # ************************************************
    # #     Check partials
    # # ************************************************
    # prob.check_partials(method="fd")
    # prob.check_totals()
