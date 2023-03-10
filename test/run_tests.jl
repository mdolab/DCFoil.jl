# --- Julia ---
"""
@File    :   run_tests.jl
@Time    :   2022/07/15
@Author  :   Galen Ng, Sicheng He
@Desc    :   Run the test files
"""

using Test

include("test_struct.jl")
include("test_hydro.jl")
include("test_solvers.jl")

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
    # --- Mesh convergence tests ---
    io = open("testConv.out", "w")
    @test test_SolveStaticRigid() <= 1e-4 # rigid hydrofoil solve
    @test test_SolveStaticIso() <= 1e-4 # ss hydrofoil solve
    @test test_SolveStaticComp() <= 1e-4 # cfrp hydrofoil
    close(io)
    # @test test_SolveForcedComp() <= 1e-12 # not ready yet
    @test test_modal() <= 1e-5 # dry and wet modal analysis of cfrp
    @test test_flutter() <= 1e-5 # flutter analysis of cfrp
end
