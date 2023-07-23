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
#                         COMMON INPUT
# ==============================================================================
nNodes = 4
DVDict = Dict(
    "α₀" => 6.0, # initial angle of attack [deg]
    "Λ" => deg2rad(-15.0), # sweep angle [rad]
    "g" => 0.04, # structural damping percentage
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
    "tipMass" => false,
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

# ==============================================================================
#                         Test sets
# ==============================================================================
@testset "Test solver" begin
    # Write your tests here.
    # @test test_BVP() <= 1e-3
    # ************************************************
    #     Structural tests
    # ************************************************
    @test test_struct() <= 1e-5 # constitutive relations

    # --- FiniteElement tests ---
    @test test_FiniteElementIso() <= 1e-10
    @test test_FiniteElementComp() <= 1e-6
    @test test_BT2_stiff() <= 1e-5
    @test test_BT2_mass() <= 1e-4
    @test testFEBT3() <= 1e-5

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
    # @test test_flutter_staticDiv() <= 1e-2 # flutter analysis of cfrp that statically diverges
    # --- Mesh convergence tests ---
    @test test_SolveStaticRigid() <= 1e-2 # rigid hydrofoil solve
    @test test_SolveStaticIso() <= 1e-2 # ss hydrofoil solve
    @test test_SolveStaticComp() <= 1e-2 # cfrp hydrofoil
    # @test test_hydroLoads() <= 1e-2
    # @test test_SolveForcedComp() <= 1e-12 # not ready yet
    @test test_modal() <= 1e-5 # dry and wet modal analysis of cfrp
    # @test test_flutter() <= 1e-5 # flutter analysis of cfrp

end

@testset "Test sensitivities" begin
    # Write your tests here.
    # ************************************************
    #     Unit test derivative tests
    # ************************************************
    # @test test_hydromass() <=1e-4 # hydrodynamic mass
    # @test test_hydrodamp() <= 1e-4
    @test test_eigenvalueAD() <= 1e-5 # eigenvalue dot product
    @test test_interp() <= 1e-1
    @test test_hydroderiv(DVDict, solverOptions) <= 1e-4

end

@testset "Larger scale local test" begin
    # Write your tests here.
    @test test_pkflutterderiv(DVDict, solverOptions) <= 1e-4
end