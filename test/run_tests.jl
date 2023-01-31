# --- Julia ---
"""
@File    :   run_tests.jl
@Time    :   2022/07/15
@Author  :   Galen Ng, Sicheng He
@Desc    :   Run the test files
"""

using Test

# include("test_BVP.jl") # TODO:
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

    # # ************************************************
    # #     Solver tests
    # # ************************************************
    # # Really just looking to see if the solver works
    # io = open("test.out", "w")
    # @test test_SolveStaticRigid() <= 1e-5 # rigid hydrofoil solve
    # @test test_SolveStaticIso() <= 1e-5 # ss hydrofoil solve
    # @test test_SolveStaticComp() <= 1e-5 # not ready yet
    # close(io)
end
