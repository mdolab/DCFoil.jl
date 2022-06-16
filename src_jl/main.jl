# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Main executable for the project


include("SolveSteady.jl")

using .SolveSteady

# --- Model parameters ---
DVDict = Dict(
    "neval" => 3,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 1.0, # free stream velocity [m/s]
    "Λ" => 0.0, # sweep angle [rad]
    "ρ_f" => 1025, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.81 * ones(3), # chord length [m]
    "s" => 2.7, # semispan [m]
    "ab" => zeros(3), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_α" => zeros(3), # static imbalance [m]
    "θ" => π / 6, # fiber angle global [rad]
)

# ==============================================================================
# Steady solution
# ==============================================================================
# --- Run the problem ---
SolveSteady.solve(DVDict["neval"], DVDict)

# ==============================================================================
# Dynamic solution
# ==============================================================================