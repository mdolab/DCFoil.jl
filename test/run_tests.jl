# --- Julia ---

# @File    :   run_tests.jl
# @Time    :   2022/07/15
# @Author  :   Galen Ng
# @Desc    :   Run the test files (comment out what you want to test)

# include("src/test_hydro.jl")
# include("src/test_struct.jl")

using Test

include("test_BVP.jl") # TODO:

@testset "Test solver" begin
    # Write your tests here.
    @test test_BVP() <= 1e-3
end
