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
    # @test test_45degwingLL() <= 2e-2

    # ************************************************
    #     Solver tests
    # ************************************************
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

end

# ==============================================================================
#                         Verification cases
# ==============================================================================
# These are verification studies published in the 2024 Composite Structures journal paper
include("../validations/akcabay_src/akcabay_staticdiv.jl")
include("../validations/akcabay_src/akcabay_flutter.jl")

# ==============================================================================
#                         Common input for sensitivity tests
# ==============================================================================
include("test_sensitivities.jl")
