# --- Julia ---
"""
@File          :   runtests.jl
@Date created  :   2022/07/15
@Last modified :   2025/01/23
@Author        :   Galen Ng
@Desc          :   Run unit tests
Some big picture notes:
"""

using Test
using ChainRulesCore
include("test_struct.jl")
include("test_hydro.jl")
include("test_solvers.jl")


# ==============================================================================
#                         Common input for unit analysis tests
# ==============================================================================
nNodes = 40
nNodesStrut = 2

DVDict1 = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "sweep" => 0.0 * π / 180, # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12 * ones(nNodes), # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(15), # fiber angle global [rad]
    # --- Strut vars ---
    "rake" => 0.0, # rake angle wrt flow [deg]
    "depth0" => 0.1,
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 0.4, # from Yingqian
    "c_strut" => 0.1 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.12 * ones(nNodesStrut), # thickness-to-chord ratio
    "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(15), # fiber angle global [rad]
)
wingOptions1 = Dict(
    "compName" => "test-comp",
    "material" => "cfrp", # preselect from material library
    "config" => "wing",
    "nNodes" => nNodes, # number of nodes on foil half wing
    "nNodeStrut" => nNodesStrut, # nodes on strut
    "use_tipMass" => false,
    "xMount" => 0.0,
)
solverOptions1 = Dict(
    "rhof" => 1000.0, # fluid density [kg/m³]
    "Uinf" => 6.0, # free stream velocity [m/s]
    # --- I/O ---
    "name" => "akcabay",
    "debug" => false,
    "outputDir" => "test_out/",
    "appendageList" => [wingOptions1],
    # --------------------------------
    #   Flow
    # --------------------------------
    "use_cavitation" => false,
    "use_freeSurface" => false,
    # --- Static solve ---
    "run_static" => true,
    # --- Forced solve ---
    "run_forced" => false,
    "fRange" => [0, 10],
    "tipForceMag" => 0.0,
    # --- Eigen solve ---
    "run_modal" => false,
    "run_flutter" => false,
    "nModes" => 5,
    "uRange" => [0.1, 1.0],
)

@testset "Test solver" begin
    # ************************************************
    #     Structural tests
    # ************************************************
    @test test_struct() <= 1e-5 # constitutive relations

    # --- FiniteElement tests ---
    @test test_FECOMP2() <= 1e-1

    # ************************************************
    #     Hydrodynamic tests
    # ************************************************
    @test test_stiffness() <= 1e-10
    @test test_damping() <= 1e-10
    @test test_mass() <= 1e-10
    # @test test_FSeffect() <= 1e-5 # not ready yet
    # @test test_dwWake() <= 1e-5
    @test test_45degwingLL() <= 1.5e-2

    # ************************************************
    #     Solver tests
    # ************************************************
    DVDict2 = Dict(
        "alfa0" => 6.0, # initial angle of attack [deg]
        "sweep" => 0.0 * π / 180, # sweep angle [rad]
        "c" => 0.0925 * ones(nNodes), # chord length [m]
        "s" => 0.2438, # semispan [m]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.03459 * ones(nNodes), # thickness-to-chord ratio
        "x_ab" => 0 * ones(nNodes), # static imbalance [m]
        "theta_f" => deg2rad(0), # fiber angle global [rad]
        # --- Strut vars ---
        "rake" => 0.0, # rake angle wrt flow [deg]
        "depth0" => 0.1,
        "beta" => 0.0, # yaw angle wrt flow [deg]
        "s_strut" => 0.4, # from Yingqian
        "c_strut" => 0.1 * ones(nNodesStrut), # chord length [m]
        "toc_strut" => 0.12 * ones(nNodesStrut), # thickness-to-chord ratio
        "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
        "x_ab_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
        "theta_f_strut" => deg2rad(15), # fiber angle global [rad]
    )
    wingOptions2 = Dict(
        "compName" => "test-comp",
        "material" => "cfrp", # preselect from material library
        "config" => "wing",
        "nNodes" => nNodes,
        "nNodeStrut" => nNodesStrut, # nodes on strut
        "use_tipMass" => false,
        "xMount" => 0.0,
    )
    solverOptions2 = Dict(
        # --- I/O ---
        "debug" => false,
        "outputDir" => "",
        # --- General solver options ---
        "appendageList" => [wingOptions2],
        "use_cavitation" => false,
        "use_freeSurface" => false,
        "Uinf" => 5.0, # free stream velocity [m/s]
        "rhof" => 1000.0, # fluid density [kg/m³]
        "use_nlll" => false,
        # --- Static solve ---
        "run_static" => false,
        # --- Forced solve ---
        "run_forced" => false,
        "fRange" => [0.0, 10.0],
        "tipForceMag" => 0.0,
        # --- Eigen solve ---
        "run_modal" => true,
        "run_flutter" => false,
        "nModes" => 5,
        "uRange" => [0.1, 1.0],
    )

    # A lot of these tolerances are loosened because of the @ffast math decorator
    # @test test_flutter_staticDiv() <= 1e-2 # flutter analysis of cfrp that statically diverges
    # --- Mesh convergence tests ---
    # @test test_SolveStaticRigid() <= 1e-2 # rigid hydrofoil solve
    # @test test_SolveStaticIso() <= 1e-2 # ss hydrofoil solve
    # @test test_SolveStaticComp(DVDict1, solverOptions1) <= 5.7e-2 # cfrp hydrofoil (kind of loose)
    # @test test_hydroLoads() <= 1e-2
    # @test test_SolveForcedComp() <= 1e-12 # not ready yet
    @test test_modal(DVDict2, solverOptions2) <= 1e-2 # dry and wet modal analysis of cfrp
    # @test test_flutter() <= 1e-5 # flutter analysis of cfrp
    # @test test_forced() <= 1e-3# forced vibration of the hydrofoils

end

# ==============================================================================
#                         Verification cases
# ==============================================================================
# These are verification studies published in the 2024 Composite Structures journal paper
include("../validations/akcabay_src/akcabay_staticdiv.jl")
include("../validations/akcabay_src/akcabay_flutter.jl")

# ==============================================================================
#                         Sensitivity tests
# ==============================================================================
include("test_sensitivities.jl")

include("test_partials.jl")
@testset "Test sensitivities" begin
    # ************************************************
    #     Common input
    # ************************************************
    nNodes = 4
    nNodesStrut = 2
    NPT_WING = 6

    appendageParams = Dict(
        "alfa0" => 6.0, # initial angle of attack [deg] (angle of flow vector)
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "ab" => 0.0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.06 * ones(nNodes), # thickness-to-chord ratio (mean) #FLAGSTAFF
        "x_ab" => 0.0 * ones(nNodes), # static imbalance [m]
        "theta_f" => deg2rad(0), # fiber angle global [rad]
        # --- Strut vars ---
        "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
        "rake" => 0.0, # rake angle about top of strut [deg]
        "beta" => 0.0, # yaw angle wrt flow [deg]
        "s_strut" => 1.0, # [m]
        "c_strut" => 0.14 * collect(LinRange(1.0, 1.0, nNodesStrut)), # chord length [m]
        "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
        "ab_strut" => 0.0 * ones(nNodesStrut), # dist from midchord to EA [m]
        "x_ab_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
        "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
    )
    appendageOptions = Dict(
        "compName" => "rudder",
        "config" => "full-wing",
        "nNodes" => nNodes,
        "nNodeStrut" => nNodesStrut,
        "use_tipMass" => false,
        "xMount" => 0.0,
        "material" => "cfrp", # preselect from material library
        "strut_material" => "cfrp",
        "path_to_geom_props" => "./INPUT/1DPROPS/",
        "path_to_struct_props" => nothing,
        "path_to_geom_props" => nothing,
    )
    solverOptions = Dict(
        # ---------------------------
        #   I/O
        # ---------------------------
        "name" => "test",
        "debug" => false,
        "writeTecplotSolution" => true,
        # ---------------------------
        #   General appendage options
        # ---------------------------
        "appendageList" => [appendageOptions],
        "gravityVector" => [0.0, 0.0, -9.81],
        # ---------------------------
        #   Flow
        # ---------------------------
        "Uinf" => 18.0, # free stream velocity [m/s]
        # "Uinf" => 11.0, # free stream velocity [m/s]
        # "Uinf" => 1.0, # free stream velocity [m/s]
        "rhof" => 1025.0, # fluid density [kg/m³]
        "nu" => 1.1892E-06,
        "use_nlll" => true, # use non-linear lifting line code
        "use_freeSurface" => false,
        "use_cavitation" => false,
        "use_ventilation" => false,
        # "use_dwCorrection" => true,
        # ---------------------------
        #   Solver modes
        # ---------------------------
        # --- Static solve ---
        "run_static" => true,
        "res_jacobian" => "analytic",
        # --- Forced solve ---
        "run_forced" => true,
        "fRange" => [0.01, 2.0], # forcing frequency sweep [Hz]
        # "df" => 0.05, # frequency step size
        "df" => 0.005, # frequency step size
        "tipForceMag" => 1.0,
        # --- p-k (Eigen) solve ---
        "run_modal" => true,
        "run_flutter" => true,
        "nModes" => 4,
        "uRange" => [29.0, 30],
        "maxQIter" => 100, # that didn't fix the slow run time...
        "rhoKS" => 500.0,
    )
    displacementsCol = zeros(6, NPT_WING)
    # --- Lifting line tests ---
    @test test_LLcostFuncJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol) <= 1e-10
    @test test_LLresidualJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol) <= 1e-10

    # --- Flutter derivative test ---
    # TODO PICKUP HERER

    # --- Structural tests ---
    # Run these tests last because they introduce a complex data type bug
    @test test_BeamCostFuncJacobians() <= 1e-10
    @test test_BeamResidualJacobians(appendageParams, appendageOptions, solverOptions) <= 1e-10

end
