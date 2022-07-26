"""
Cheap unit test
"""

include("../src/InitModel.jl")
include("../src/GovDiffEqns.jl") # what we want to test

using LinearAlgebra
using .Steady, .InitModel, .Solver

function test_BVP()
    # ==============================================================================
    # Setup the test problem
    # ==============================================================================
    neval = 250
    DVDict = Dict(
        "neval" => neval,
        "α₀" => 6.0, # initial angle of attack [deg]
        "U∞" => 5, # free stream velocity [m/s]
        "Λ" => 30.0 * π / 180, # sweep angle [rad]
        "ρ_f" => 1000, # fluid density [kg/m³]
        "material" => "cfrp", # preselect from material library
        "g" => 0.04, # structural damping percentage
        "c" => 0.81 * ones(neval), # chord length [m]
        "s" => 2.7, # semispan [m]
        "ab" => zeros(neval), # dist from midchord to EA [m]
        "toc" => 0.06, # thickness-to-chord ratio
        "x_α" => zeros(neval), # static imbalance [m]
        "θ" => 10 * π / 180, # fiber angle global [rad]
    )

    foil = InitModel.init_steady(DVDict["neval"], DVDict)

    q⁰ = zeros(8)
    q⁰[5:end] .= 1.0
    tsol, usol = Solver.solve_bvp(Steady.compute_∂q∂y, q⁰, 0, 1, 25, Steady.compute_g, foil)

    # Reference value
    ref_sol = [0.0290, -0.0035]

    # Relative error
    rel_err = LinearAlgebra.norm(usol[1:2, end] - ref_sol, 2) / LinearAlgebra.norm(ref_sol, 2)
    return rel_err

end

