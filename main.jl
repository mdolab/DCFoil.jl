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

outputDir = "./OUTPUT/testAero/"

# ************************************************
#     Model parameters
# ************************************************
neval = 3 # spatial nodes

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
# --- Foil from Deniz Akcabay's 2020 paper ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(neval), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 10 * π / 180, # fiber angle global [rad]
)

# ==============================================================================
# Steady solution
# ==============================================================================
SolveSteady.solve(DVDict["neval"], DVDict, outputDir)


# # --- Write out solution files ---
# SolveSteady.write_sol() # intention here is to get pretty plottable data to visualize in paraview or tecplot

# ==============================================================================
# Dynamic solution
# ==============================================================================