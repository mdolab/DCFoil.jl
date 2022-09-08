# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Main executable for the project


include("src/SolveSteady.jl")

using .SolveSteady

# ==============================================================================
# Setup hydrofoil model
# ==============================================================================
# --- Model parameters ---
neval = 100 # spatial nodes
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 30.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.81 * ones(neval), # chord length [m]
    "s" => 2.7, # semispan [m]
    "ab" => zeros(neval), # dist from midchord to EA [m]
    "toc" => 0.06, # thickness-to-chord ratio
    "x_α" => zeros(neval), # static imbalance [m]
    "θ" => 10*π / 180, # fiber angle global [rad]
)
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 30.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000, # fluid density [kg/m³]
    "material" => "test", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 1 * ones(neval), # chord length [m]
    "s" => 1, # semispan [m]
    "ab" => zeros(neval), # dist from midchord to EA [m]
    "toc" => 1, # thickness-to-chord ratio
    "x_α" => zeros(neval), # static imbalance [m]
    "θ" => 0*π / 180, # fiber angle global [rad]
)

# ==============================================================================
# Steady solution
# ==============================================================================
SolveSteady.solve(DVDict["neval"], DVDict)


# # --- Write out solution files ---
# SolveSteady.write_sol() # intention here is to get pretty plottable data to visualize in paraview or tecplot

# ==============================================================================
# Dynamic solution
# ==============================================================================