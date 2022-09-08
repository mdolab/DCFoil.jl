# --- Julia ---
"""
@File    :   run_tests.jl
@Time    :   2022/07/15
@Author  :   Galen Ng, Sicheng He
@Desc    :   Run the test files
"""

using Test

include("test_BVP.jl") # TODO:
include("test_struct.jl")
# include("src/test_hydro.jl")
include("test_FiniteElement.jl")

@testset "Test solver" begin
    # Write your tests here.
    # @test test_BVP() <= 1e-3
    @test test_struct() <= 1e-5
    
    # --- FiniteElement tests ---
    @test test_FiniteElement() <= 1e-5
end
