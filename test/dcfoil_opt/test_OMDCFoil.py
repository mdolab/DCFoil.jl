# --- Python 3.11 ---
"""
@File          :   test_OMDCFoil.py
@Date created  :   2025/03/12
@Last modified :   2025/03/12
@Author        :   Galen Ng
@Desc          :   Full scale test of DCFoil with OpenMDAO and pygeo for DVs


example run:
python test_OMDCFoil.py --input INPUT --foil mothrudder
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import argparse
from pathlib import Path

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from pygeo.mphys import OM_DVGEOCOMP
import openmdao.api as om
from multipoint import Multipoint
from SETUP import setup_OMdvgeo, setup_dcfoil

import juliacall

jl = juliacall.newmodule("DCFoil")
jl.include("../../src/struct/beam_om.jl")  # discipline 1
jl.include("../../src/hydro/liftingline_om.jl")  # discipline 2
jl.include("../../src/loadtransfer/ldtransfer_om.jl")  # coupling components
jl.include("../../src/io/MeshIO.jl")  # mesh I/O for reading inputs in

from omjlcomps import JuliaExplicitComp, JuliaImplicitComp

# ==============================================================================
#                         Top level variables
# ==============================================================================
outputDir = "output"

# ==============================================================================
#                         OpenMDAO model
# ==============================================================================
class Top(Multipoint):
    """
    OM model for DCFoil with pygeo
    """

    def setup(self):

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.add_subsystem("mesh", om.IndepVarComp())

        self.mesh.add_output("x_ptVec0", val=np.array(ptVec), distributed=True)

        self.add_subsystem(
            "geometry",
            OM_DVGEOCOMP(file=files["FFDFile"], type="ffd"),
            promotes=["twist", "sweep", "dihedral", "taper", "span"],
        )

        self.connect("mesh.x_ptVec0", "geometry.x_ptVec_in")

        # ---------------------------
        #   Julia specific components
        # ---------------------------
        appendageOptions = appendageList[0]
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
            jlcomp=jl.OMLiftingLineFuncs(nodeConn, appendageParams, appendageOptions, solverOptions)
        )
        self.add_subsystem(
            "beamstruct",
            impcomp_struct_solver,
            promotes_inputs=["ptVec", "theta_f", "toc"],
            promotes_outputs=["deflections"],
        )
        self.add_subsystem(
            "beamstruct_funcs",
            expcomp_struct_func,
            promotes_inputs=["ptVec", "deflections", "theta_f", "toc"],
            promotes_outputs=["*"],  # everything!
        )
        self.add_subsystem(
            "liftingline",
            impcomp_LL_solver,
            promotes_inputs=["ptVec", "alfa0", "displacements_col"],
            promotes_outputs=["gammas", "gammas_d"],
        )
        self.add_subsystem(
            "liftingline_funcs",
            expcomp_LL_func,
            promotes_inputs=["gammas", "gammas_d", "ptVec", "alfa0", "displacements_col"],  # promotion auto connects these variables
            promotes_outputs=["*"],  # everything!
        )

        self.connect("geometry.x_ptVec0", "ptVec") # connect the geometry to the beam and lifting line solver

    def configure(self):

        # ************************************************
        #     Geometry
        # ************************************************
        self.geometry.nom_add_discipline_coords("ptVec", np.array(ptVec))

        self, geoDVDictInfo = setup_OMdvgeo.setup(args, self, None, files)

        for dvName, value in geoDVDictInfo.items():
            self.dvs.add_output(dvName, val=value["value"])
            self.add_design_var(dvName, lower=value["lower"], upper=value["upper"], scaler=value["scale"])

        # ************************************************
        #     DCFoil DVs
        # ************************************************
        for key, value in valDict.items():
            self.dvs.add_output(key, val=value)
            self.add_design_var(key, lower=lowerDict[key], upper=upperDict[key], scaler=scaleDict[key])

        self.add_objective("CDi")


# ==============================================================================
#                         MAIN DRIVER
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="INPUT")
    parser.add_argument("--foil", type=str, default=None, help="Foil .dat coord file name w/o .dat")
    parser.add_argument("--is_dynamic", action="store_true", default=False)
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

    Path(outputDir).mkdir(exist_ok=True, parents=True)

    # ************************************************
    #     Input Files/ IO
    # ************************************************
    files = {}
    files["gridFile"] = [
        f"./{args.input}/{args.foil}_foil_stbd_mesh.dcf",
        f"./{args.input}/{args.foil}_foil_port_mesh.dcf",
    ]
    files["FFDFile"] = f"{args.input}/{args.foil}_ffd.xyz"

    Grid = jl.DCFoil.add_meshfiles(files["gridFile"], {"junction-first": True})
    # Unpack for this code. Remember Julia is transposed from Python
    LECoords = np.array(Grid.LEMesh).T
    TECoords = np.array(Grid.TEMesh).T
    nodeConn = np.array(Grid.nodeConn)
    ptVec, m, n = jl.DCFoil.FEMMethods.unpack_coords(Grid.LEMesh, Grid.TEMesh)
    # ************************************************
    #     DCFoil options setup
    # ************************************************
    solverOptions, appendageParams, appendageList, valDict, lowerDict, upperDict, scaleDict = setup_dcfoil.setup(
        args, None, files, None, outputDir
    )
    # Need to set struct damping once at the beginning to avoid optimization taking advantage of changing beta
    solverOptions = jl.FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[0])

    prob = om.Problem()
    prob.model = Top()

    # ---------------------------
    #   Opt setup
    # ---------------------------
    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
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

    print("Running setup...")
    prob.setup()

    # ************************************************
    #     Execution
    # ************************************************
    print("Running model...")
    prob.run_model()

    # print("ptVec", prob.get_val("geometry.x_ptVec0").reshape(-1, 3))
    print("deflections", prob.get_val("deflections"))
    print("CL", prob.get_val("CL"))
    print("CDi", prob.get_val("CDi"))

    print(20 * "=")
    print("setting sweep to 45 deg")
    print(20 * "=")
    prob.set_val("sweep", 45.0)

    prob.run_model()
    # print("ptVec", prob.get_val("geometry.x_ptVec0").reshape(-1, 3))
    print("deflections", prob.get_val("deflections"))
    print("CL", prob.get_val("CL"))
    print("CDi", prob.get_val("CDi"))

    breakpoint()
