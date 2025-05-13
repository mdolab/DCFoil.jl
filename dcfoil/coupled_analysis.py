"""
Top-level OpenMDAO group for structural/hydrodynamic/hydroelastic static analysis + flutter analysis
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
import openmdao.api as om
import juliacall

jl = juliacall.newmodule("DCFoil")

jl.include("../src/io/MeshIO.jl")  # mesh I/O for reading inputs in
jl.include("../src/struct/beam_om.jl")  # discipline 1
jl.include("../src/hydro/liftingline_om.jl")  # discipline 2
### jl.include("../src/loadtransfer/ldtransfer_om.jl")  # coupling components
jl.include("../src/solvers/solveflutter_om.jl")  # discipline 4 flutter solver
jl.include("../src/solvers/solveforced_om.jl")  # discipline 5 forced solver

from omjlcomps import JuliaExplicitComp, JuliaImplicitComp

from transfer import DisplacementTransfer, LoadTransfer, CLaInterpolation


class CoupledAnalysis(om.Group):

    def initialize(self):
        # define all options and defaults
        self.options.declare("analysis_mode", values=["struct", "flow", "coupled"], desc="Analysis mode")
        self.options.declare("ptVec_init", types=np.ndarray, desc="Initial value of the point vector")
        self.options.declare("npt_wing", default=0, desc="Number of flow collocation points")
        self.options.declare("n_node_fullspan", default=0, desc="Number of points on the full span")

        self.options.declare("appendageOptions", types=dict, desc="Appendage options")
        self.options.declare("appendageParams", types=dict, desc="Appendage parameters")
        self.options.declare("nodeConn", types=np.ndarray, desc="Node connectivity")
        self.options.declare("solverOptions", types=dict, desc="Solver options")

    def setup(self):
        npt_wing = self.options["npt_wing"]
        n_node_fullspan = self.options["n_node_fullspan"]

        appendageOptions = self.options["appendageOptions"]
        appendageParams = self.options["appendageParams"]
        nodeConn = self.options["nodeConn"]
        solverOptions = self.options["solverOptions"]

        # ************************************************
        #     Setup OM components
        # ************************************************
        # TODO: move these into setup method?

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
        # expcomp_load = JuliaExplicitComp(
        #     jlcomp=jl.OMLoadTransfer(
        #         nodeConn, appendageParams, appendageOptions, solverOptions
        #     )
        # )
        # expcomp_displacement = JuliaExplicitComp(
        #     jlcomp=jl.OMLoadTransfer(
        #         nodeConn, appendageParams, appendageOptions, solverOptions
        #     )
        # )
        expcomp_flutter = JuliaExplicitComp(
            jlcomp=jl.OMFlutter(nodeConn, appendageParams, appendageOptions, solverOptions)
        )
        expcomp_forced = JuliaExplicitComp(
            jlcomp=jl.OMForced(nodeConn, appendageParams, appendageOptions, solverOptions)
        )

        # ************************************************
        #     Assemble components into an OM group
        # ************************************************
        # --- geometry component ---
        # now ptVec is just an input, so use IVC as an placeholder. Later replace IVC with a geometry component
        indep = self.add_subsystem("input", om.IndepVarComp(), promotes=["*"])
        indep.add_output("ptVec", val=self.options["ptVec_init"])   # TODO: set units

        if self.options["analysis_mode"] == "struct":
            # --- Structural analysis only ---
            self.add_subsystem(
                "beamstruct",
                impcomp_struct_solver,
                promotes_inputs=["ptVec", "traction_forces"],
                promotes_outputs=["deflections"],
            )
            self.add_subsystem(
                "beamstruct_funcs",
                expcomp_struct_func,
                promotes_inputs=["ptVec", "deflections"],
                promotes_outputs=["*"],  # everything!
            )

        elif self.options["analysis_mode"] == "flow":
            # --- Do nonlinear liftingline ---
            self.add_subsystem(
                "liftingline",
                impcomp_LL_solver,
                promotes_inputs=["ptVec", "alfa0", "displacements_col"],
                promotes_outputs=["gammas", "gammas_d"],
            )
            self.add_subsystem(
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

        elif self.options["analysis_mode"] == "coupled":
            # --- Combined hydroelastic ---
            couple = self.add_subsystem("hydroelastic", om.Group(), promotes=["*"])

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
            couple.nonlinear_solver = om.NonlinearBlockGS(use_aitken=True, maxiter=200, iprint=2, atol=1e-6, rtol=0)
            ### couple.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=50, iprint=2, atol=1e-7, rtol=0)
            couple.linear_solver = om.DirectSolver()   # for adjoint

            # CL_alpha mapping from flow points to FEM nodes (after hydroelestic loop)
            self.add_subsystem(
                "CLa_interp",
                CLaInterpolation(n_node=n_node_fullspan, n_strips=npt_wing),
                promotes_inputs=["collocationPts", "nodes", ("CL_alpha", "cla")],
                promotes_outputs=[("CL_alpha_node", "cla_node")],
            )

            # --- Dynamic solvers ---
            # TODO: need to fix the shape of cla here
            # self.add_subsystem("flutter_funcs", expcomp_flutter, promotes_inputs=["*"], promotes_outputs=["*"])
            # model.add_subsystem("forced_funcs", expcomp_forced, promotes_inputs=["*"], promotes_outputs=["*"])

        else:
            raise ValueError("Invalid analysis mode. Choose 'struct', 'flow', or 'coupled'.")


            