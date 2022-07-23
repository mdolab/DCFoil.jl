"""
Cheap unit test
"""

include("InitModel.jl")
include("GovDiffEqns.jl") # what we want to test

using .Steady, .InitModel

# ==============================================================================
# Setup the test problem
# ==============================================================================
neval = 2
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 10, # free stream velocity [m/s]
    "Λ" => 0.0, # sweep angle [rad]
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

q⁰ = ones(8) # guess

∂q∂y = Steady.compute_∂q∂y(q⁰, 0.0, foil)

println("This is the ∂q∂y:")
println(∂q∂y)