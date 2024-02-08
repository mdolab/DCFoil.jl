# --- Julia ---
"""
@File    :   run_tests.jl
@Time    :   2022/07/15
@Author  :   Galen Ng, Sicheng He
@Desc    :   Run the test files
Some big picture notes:
"""

using Test

include("test_struct.jl")
include("test_hydro.jl")
include("test_solvers.jl")
include("test_sensitivities.jl")


# ==============================================================================
#                         Common input for unit analysis tests
# ==============================================================================
nNodes = 40
nNodesStrut = 2
DVDict = Dict(
		"α₀" => 6.0, # initial angle of attack [deg]
		"Λ" => 0.0 * π / 180, # sweep angle [rad]
		"zeta" => 0.04, # modal damping ratio at first 2 modes
		"c" => 1 * ones(nNodes), # chord length [m]
		"s" => 1.0, # semispan [m]
		"ab" => zeros(nNodes), # dist from midchord to EA [m]
		"toc" => 1, # thickness-to-chord ratio
		"x_αb" => zeros(nNodes), # static imbalance [m]
		"θ" => deg2rad(15), # fiber angle global [rad]
        # --- Strut vars ---
        "beta" => 0.0, # yaw angle wrt flow [deg]
        "s_strut" => 0.4, # from Yingqian
        "c_strut" => 0.1 * ones(nNodesStrut), # chord length [m]
        "toc_strut" => 0.12 * ones(nNodesStrut), # thickness-to-chord ratio
        "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
        "x_αb_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
        "θ_strut" => deg2rad(15), # fiber angle global [rad]
	)
solverOptions = Dict(
    "material" => "test-comp", # preselect from material library
    "nNodes" => nNodes,
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "config" => "wing",
)
DVDict1 = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "c" => 0.1 * ones(nNodes), # chord length [m]
        "s" => 0.3, # semispan [m]
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.12 * ones(nNodes), # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(15), # fiber angle global [rad]
        # --- Strut vars ---
        "beta" => 0.0, # yaw angle wrt flow [deg]
        "s_strut" => 0.4, # from Yingqian
        "c_strut" => 0.1 * ones(nNodesStrut), # chord length [m]
        "toc_strut" => 0.12 * ones(nNodesStrut), # thickness-to-chord ratio
        "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
        "x_αb_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
        "θ_strut" => deg2rad(15), # fiber angle global [rad]
    )
solverOptions1 = Dict(
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "U∞" => 6.0, # free stream velocity [m/s]
    # --- I/O ---
    "name" => "akcabay",
    "debug" => false,
    "outputDir" => "test_out/",
    # --- General solver options ---
    "config" => "wing",
    "nNodes" => nNodes, # number of nodes on foil half wing
    "nNodeStrut" => nNodesStrut, # nodes on strut
    "use_tipMass" => false,
    "material" => "cfrp", # preselect from material library
    "rotation" => 0.0, # deg
    # --------------------------------
    #   Flow
    # --------------------------------
    "use_cavitation" => false,
    "use_freesurface" => false,
    # --- Static solve ---
    "run_static" => true,
    # --- Forced solve ---
    "run_forced" => false,
    "fSweep" => 0:0.1:10,
    "tipForceMag" => 0.0,
    # --- Eigen solve ---
    "run_modal" => false,
    "run_flutter" => false,
    "nModes" => 5,
    "uRange" => nothing,
)
DVDict2 = Dict(
        "α₀" => 6.0, # initial angle of attack [deg]
        "Λ" => 0.0 * π / 180, # sweep angle [rad]
        "c" => 0.0925 * ones(nNodes), # chord length [m]
        "s" => 0.2438, # semispan [m]
        "zeta" => 0.04, # modal damping ratio at first 2 modes
        "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
        "toc" => 0.03459 * ones(nNodes), # thickness-to-chord ratio
        "x_αb" => 0 * ones(nNodes), # static imbalance [m]
        "θ" => deg2rad(0), # fiber angle global [rad]
        # --- Strut vars ---
        "beta" => 0.0, # yaw angle wrt flow [deg]
        "s_strut" => 0.4, # from Yingqian
        "c_strut" => 0.1 * ones(nNodesStrut), # chord length [m]
        "toc_strut" => 0.12 * ones(nNodesStrut), # thickness-to-chord ratio
        "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
        "x_αb_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
        "θ_strut" => deg2rad(15), # fiber angle global [rad]
    )
solverOptions2 = Dict(
    # --- I/O ---
    "debug" => false,
    "outputDir" => "",
    "config" => "wing",
    # --- General solver options ---
    "nNodes" => nNodes,
    "use_tipMass" => false,
    "use_cavitation" => false,
    "use_freesurface" => false,
    "material" => "cfrp", # preselect from material library
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    # --- Static solve ---
    "run_static" => false,
    # --- Forced solve ---
    "run_forced" => false,
    "fSweep" => 0:0.1:10,
    "tipForceMag" => 0.0,
    # --- Eigen solve ---
    "run_modal" => true,
    "run_flutter" => false,
    "nModes" => 5,
    "uRange" => nothing,
)
@testset "Test solver" begin
    # Write your tests here.
    # ************************************************
    #     Structural tests
    # ************************************************
    @test test_struct() <= 1e-5 # constitutive relations

    # --- FiniteElement tests ---
    # These are old element types that we don't use anymore
    # @test test_FiniteElementIso(DVDict, solverOptions) <= 1e-10
    # solverOptions["material"] = "test-comp"
    # @test test_FiniteElementComp(DVDict, solverOptions) <= 1e-6
    # @test test_BT2_stiff() <= 1e-5
    # @test test_BT2_mass() <= 1e-4
    # @test test_FEBT3() <= 1e-5
    @test test_FECOMP2(DVDict, solverOptions) <= 1e-1

    # ************************************************
    #     Hydrodynamic tests
    # ************************************************
    @test test_stiffness() <= 1e-10
    @test test_damping() <= 1e-10
    @test test_mass() <= 1e-10
    # @test test_FSeffect() <= 1e-5 # not ready yet

    # ************************************************
    #     Solver tests
    # ************************************************
    # A lot of these tolerances are loosened because of the @ffast math decorator
    # @test test_flutter_staticDiv() <= 1e-2 # flutter analysis of cfrp that statically diverges
    # --- Mesh convergence tests ---
    # @test test_SolveStaticRigid() <= 1e-2 # rigid hydrofoil solve
    # @test test_SolveStaticIso() <= 1e-2 # ss hydrofoil solve
    @test test_SolveStaticComp(DVDict1, solverOptions1) <= 1.7e-2 # cfrp hydrofoil (kind of loose)
    # @test test_hydroLoads() <= 1e-2
    # @test test_SolveForcedComp() <= 1e-12 # not ready yet
    @test test_modal(DVDict2, solverOptions2) <= 1e-2 # dry and wet modal analysis of cfrp
    # @test test_flutter() <= 1e-5 # flutter analysis of cfrp

end


# ==============================================================================
#                         Common input for sensitivity tests
# ==============================================================================
nNodes = 4
DVDict = Dict(
    "α₀" => 6.0, # initial angle of attack [deg]
    "Λ" => deg2rad(-15.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(15), # fiber angle global [rad]
)

solverOptions = Dict(
    # --- I/O ---
    "name" => "test",
    "debug" => false,
    "outputDir" => "./test_out/",
    # --- General solver options ---
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "nNodes" => nNodes,
    "config" => "wing",
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_tipMass" => false,
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => false,
    # --- Forced solve ---
    "run_forced" => false,
    "fSweep" => range(0.1, 1000.0, 1000),
    "tipForceMag" => 0.5 * 0.5 * 1000 * 100 * 0.03,
    # --- Eigen solve ---
    "run_modal" => false,
    "run_flutter" => true,
    "nModes" => 4,
    "uRange" => [187.0, 190.0],
    "maxQIter" => 100,
    "rhoKS" => 80.0,
)

@testset "Test sensitivities" begin
    # Write your tests here.
    # ************************************************
    #     Unit test derivative tests
    # ************************************************
    # @test test_hydromass() <=1e-4 # hydrodynamic mass
    # @test test_hydrodamp() <= 1e-4
    @test test_eigenvalueAD() <= 1e-5 # eigenvalue dot product
    @test test_interp() <= 1e-1
    # @test test_hydroderiv(DVDict, solverOptions) <= 1e-4
end

# @testset "Larger scale local test" begin
#     # Write your tests here.
#     @test test_pkflutterderiv(DVDict, solverOptions) <= 1e-4
# end