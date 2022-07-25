"""
Cheap unit test
"""

include("InitModel.jl")
include("GovDiffEqns.jl") # what we want to test

using .Steady, .InitModel, .Solver

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

q⁰ = ones(8) # guess

∂q∂y = Steady.compute_∂q∂y(q⁰, 0.5, foil)

println("This is the ∂q∂y:")
println(∂q∂y)

# TODO: somehow q⁰ is being overwritten which I do not understand, why? Seems like line 34 overwrites what is in q⁰ and keeps track of it...
q⁰ = ones(8)
g = Steady.compute_g(q⁰, q⁰, foil)

println("This is the BC residual function (should be zeros when solved)")
println(g)

# ==============================================================================
#                         Test ODE solvers
# ==============================================================================
function basicODE(y, t)
    m = length(y)
    ẏ = zeros(m)
    for ii in 1:1:m
        ẏ[ii] = 2 * t
    end
    return ẏ
end

tsol, ysol = Solver.solve_rk4(basicODE, zeros(8), 0.0, 5.0, 25)

# ************************************************
#     Test the BVP
# ************************************************
q⁰ = zeros(8)
q⁰[5:end] .= 1.0
tsol, usol = Solver.solve_bvp(Steady.compute_∂q∂y, q⁰, 0, 1, 25, Steady.compute_g, foil)

println(usol[:, begin])
println("Tip bending and twist:")
println(usol[1, end], "m")
println(usol[2, end], "rad")
println("Error from expected:")
println(usol[1:2, end] - [0.0290, -0.0035])