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

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
# import niceplots as nplt
import setup_OMdvgeo
import openmdao.api as om

files = {}
# files["gridFile"] = [
#         f"../test/dcfoil_opt/mothrudder_foil_stbd_mesh.dcf",
#         f"../test/dcfoil_opt/mothrudder_foil_port_mesh.dcf",
#     ]
FFDFile = "../test/dcfoil_opt/INPUT/mothrudder_ffd.xyz"
files["FFDFile"] = FFDFile
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
    args = parser.parse_args()
    # --- Echo the args ---
    print(30 * "-")
    print("Arguments are", flush=True)
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}", flush=True)
    print(30 * "-", flush=True)

    model = om.Group()
    prob = om.Problem(model)

    model, dvDictInfo = setup_OMdvgeo.setup(args, model, None, files)
