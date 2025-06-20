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
        self.options.declare("include_flutter", default=True, desc="Include flutter analysis")
        self.options.declare("include_forced", default=True, desc="Include forced vibration analysis")
        self.options.declare("ptVec_init", types=np.ndarray, desc="Initial value of the point vector")
        self.options.declare("npt_wing", default=0, desc="Number of flow collocation points for full or half wing")
        self.options.declare("n_node", default=0, desc="Number of FEM nodes for full or half wing")

        self.options.declare("appendageOptions", types=dict, desc="Appendage options")
        self.options.declare("appendageParams", types=dict, desc="Appendage parameters")
        self.options.declare("nodeConn", types=np.ndarray, desc="Node connectivity")
        self.options.declare("solverOptions", types=dict, desc="Solver options")
        self.options.declare(
            "flowOptions", types=dict, desc="Flow options"
        )  # had to hack this to give point specific options

    def setup(self):
        npt_wing = self.options["npt_wing"]
        n_node = self.options["n_node"]

        appendageOptions = self.options["appendageOptions"]
        appendageParams = self.options["appendageParams"]
        nodeConn = self.options["nodeConn"]
        flowOptions = self.options["flowOptions"]
        for key, value in flowOptions.items():
            self.options["solverOptions"][key] = value
        solverOptions = self.options["solverOptions"]
        # There was a weird multipoint quirk hear so pass the flow properties directly into the julia components

        # ************************************************
        #     Setup OM components
        # ************************************************
        # TODO: move these into setup method?

        impcomp_struct_solver = JuliaImplicitComp(
            jlcomp=jl.OMFEBeam(nodeConn, appendageParams, appendageOptions, solverOptions)
        )
        expcomp_struct_func = JuliaExplicitComp(
            jlcomp=jl.OMFEBeamFuncs(nodeConn, appendageParams, appendageOptions, solverOptions)
        )
        impcomp_LL_solver = JuliaImplicitComp(
            jlcomp=jl.OMLiftingLine(
                nodeConn, appendageParams, appendageOptions, solverOptions, flowOptions["Uinf"], flowOptions["depth0"]
            )
        )
        expcomp_LL_func = JuliaExplicitComp(
            jlcomp=jl.OMLiftingLineFuncs(
                nodeConn, appendageParams, appendageOptions, solverOptions, flowOptions["Uinf"], flowOptions["depth0"]
            )
        )
        # --- Make it so AoA is set for flutter solver ---
        appendageParamsCopy = appendageParams.copy()
        appendageParamsCopy["alfa0"] = flowOptions["alfa0_flutter"]
        solverOptionsCopy = solverOptions.copy()
        solverOptionsCopy["outputDir"] = flowOptions["outputDir"]  # make sure output dir is set for dynamic solvers
        expcomp_flutter = JuliaExplicitComp(
            jlcomp=jl.OMFlutter(nodeConn, appendageParamsCopy, appendageOptions, solverOptionsCopy)
        )

        expcomp_forced = JuliaExplicitComp(
            jlcomp=jl.OMForced(nodeConn, appendageParams, appendageOptions, solverOptionsCopy)
        )

        # ************************************************
        #     Assemble components into an OM group
        # ************************************************
        # --- input variables ---
        # now ptVec is just an input, so use IVC as an placeholder. Later replace IVC with a geometry component
        # indep = self.add_subsystem("input", om.IndepVarComp(), promotes=["*"])
        # indep.add_output("ptVec", val=self.options["ptVec_init"])  # TODO: set units
        # # other constant setup. If we want to connect these values from upstream component, we need to commend these lines out
        # # TODO: double check unit consistency to Julia layer
        # indep.add_output("alfa0", val=appendageParams["alfa0"], units="deg")
        # indep.add_output("theta_f", val=appendageParams["theta_f"], units="rad")
        # indep.add_output("toc", val=appendageParams["toc"])

        if self.options["analysis_mode"] == "struct":
            # --- Structural analysis only ---
            self.add_subsystem(
                "beamstruct",
                impcomp_struct_solver,
                promotes_inputs=["ptVec", "traction_forces", "theta_f", "toc"],
                promotes_outputs=["deflections"],
            )
            self.add_subsystem(
                "beamstruct_funcs",
                expcomp_struct_func,
                promotes_inputs=["ptVec", "deflections", "theta_f", "toc"],
                promotes_outputs=["*"],  # everything!
            )

        elif self.options["analysis_mode"] == "flow":
            # --- Do nonlinear liftingline ---
            self.add_subsystem(
                "liftingline",
                impcomp_LL_solver,
                promotes_inputs=["ptVec", "alfa0", "displacements_col"],
                promotes_outputs=[
                    "gammas",
                    "gammas_d",
                    # "jigtwist",
                ],
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
                    "toc",
                ],  # promotion auto connects these variables
                promotes_outputs=["*"],  # everything!
            )

        elif self.options["analysis_mode"] == "coupled":
            # --- Combined hydroelastic ---
            couple = self.add_subsystem(f"hydroelastic", om.Group(), promotes=["*"])

            # structure
            couple.add_subsystem(
                "beamstruct",
                impcomp_struct_solver,
                promotes_inputs=["ptVec", "traction_forces", "theta_f", "toc"],
                promotes_outputs=["deflections"],
            )
            couple.add_subsystem(
                "beamstruct_funcs",
                expcomp_struct_func,
                promotes_inputs=["ptVec", "deflections", "theta_f", "toc"],
                promotes_outputs=["*"],  # everything!
            )

            # displacement transfer
            couple.add_subsystem(
                "disp_transfer",
                DisplacementTransfer(
                    n_node=n_node,
                    n_strips=npt_wing,
                    xMount=appendageOptions["xMount"],
                    config=appendageOptions["config"],
                ),
                promotes_inputs=["nodes", "deflections", "collocationPts"],
                promotes_outputs=[("disp_colloc", "displacements_col")],
            )

            # hydrodynamics
            couple.add_subsystem(
                "liftingline",
                impcomp_LL_solver,
                promotes_inputs=["ptVec", "alfa0", "displacements_col"],
                promotes_outputs=[
                    "gammas",
                    "gammas_d",
                    # "jigtwist",
                ],
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
                    "toc",
                ],  # promotion auto connects these variables
                promotes_outputs=["*"],  # everything!
            )

            # load transfer
            couple.add_subsystem(
                "load_transfer",
                LoadTransfer(
                    n_node=n_node,
                    n_strips=npt_wing,
                    xMount=appendageOptions["xMount"],
                    config=appendageOptions["config"],
                ),
                promotes_inputs=[("forces_hydro", "forces_dist"), "collocationPts", "nodes"],
                promotes_outputs=[("loads_str", "traction_forces")],
            )

            # hydroelastic coupled solver
            couple.nonlinear_solver = om.NonlinearBlockGS(use_aitken=True, maxiter=200, iprint=2, atol=1e-6, rtol=0)
            # couple.nonlinear_solver = om.NonlinearBlockGS(use_aitken=False, maxiter=200, iprint=2, atol=1e-6, rtol=0, err_on_non_converge=True)
            ### couple.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=50, iprint=2, atol=1e-7, rtol=0)
            couple.linear_solver = om.DirectSolver()  # for adjoint

            # CL_alpha mapping from flow points to FEM nodes (after hydroelestic loop)
            self.add_subsystem(
                "CLa_interp",
                CLaInterpolation(n_node=n_node, n_strips=npt_wing),
                promotes_inputs=["collocationPts", "nodes", ("CL_alpha", "cla_col")],
                promotes_outputs=[("CL_alpha_node", "cla")],
            )

            # --- Dynamic solvers ---
            if self.options["include_flutter"]:
                self.add_subsystem("flutter_funcs", expcomp_flutter, promotes_inputs=["*"], promotes_outputs=["*"])
            if self.options["include_forced"]:
                self.add_subsystem("forced_funcs", expcomp_forced, promotes_inputs=["*"], promotes_outputs=["*"])

        else:
            raise ValueError("Invalid analysis mode. Choose 'struct', 'flow', or 'coupled'.")
