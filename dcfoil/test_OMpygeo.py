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
import openmdao.api as om

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

    model = om.Group()
    model.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
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
        # "Linesearch tolerance": 0.99,  # all gradients are known so we can do less accurate LS
        # "Nonderivative linesearch": None,  # Comment out to specify yes nonderivative (nonlinear problem)
        # "Major Step Limit": 5e-3,
        # "Major iterations limit": 1,  # NOTE: for debugging; remove before runs if left active by accident
    }



    # ************************************************
    #     DVGeo setup
    # ************************************************
    model, dvDictInfo = setup_OMdvgeo.setup(args, model, None, files)
    DVGeo = model.geometry.nom_getDVGeo()
    Path(outputDir).mkdir(exist_ok=True, parents=True)
    DVGeo.writeTecplot(fileName=f"./{outputDir}/dvgeo.dat", solutionTime=1)

    ptSetName = "ptVec"
    model.geometry.nom_addPointSet(ptVec, ptSetName, add_output=False, DVGeoName="defaultDVGeo")
    model.geometry.add_input(f"{ptSetName}_in", distributed=True, val=ptVec)
    model.geometry.add_output(f"{ptSetName}_0", distributed=True, val=ptVec)

    # ************************************************
    #     Add design vars
    # ************************************************
    for key, value in dvDictInfo.items():
        print(f"{key} design variable info:")
        print(f"lower: {value['lower']}")
        print(f"upper: {value['upper']}")
        print(f"scale: {value['scale']}")
        prob.model.add_design_var(key, lower=value["lower"], upper=value["upper"], scaler=value["scale"])
        
        prob.model.dvs.add_output(key, val=value["value"])

    # TODO: PICKUP HERE, WHY ARE THE pts NOT MOVING WHEN I CHANGE THE SWEEP?? Look at mphys examples more
    # MAYBE IT IS BECAUSE OF THE DV SEPARATION
    prob.setup()

    prob.set_val("sweep", 10.0)

    prob.final_setup()
    prob.run_model()

    print("ptVec_in")
    print(prob.get_val("ptVec_in"))
    print("ptVec_0")
    print(prob.get_val("ptVec_0"))


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

                # for ind, val in enumerate(twist_vals):
                for val in wave:

                    print("twist: ", val)

                    dvDict = DVGeo.getValues()
                    dvDict["twist"][ii] = val
                    DVGeo.setDesignVars(dvDict)

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
                DVGeo = model.geometry.nom_getDVGeo()

                # Write deformed FFD
                DVGeo.writeTecplot(f"{dirName}/sweep_{i_frame:03d}_ffd.dat")
                # DVGeo.writeRefAxes(f"{dirName}/sweep_{i_frame:03d}_axes")

                i_frame += 1

    # # ************************************************
    # #     Check partials
    # # ************************************************
    # prob.check_partials(method="fd")
    # prob.check_totals()
