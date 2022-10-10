# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Main executable for the project


include("src/SolveSteady.jl")
include("src/SolveDynamic.jl")

using JSON
using .SolveSteady
# using .SolveDynamic

# ==============================================================================
# Setup hydrofoil model and solver settings
# ==============================================================================

# ************************************************
#     I/O
# ************************************************
outputDir = "./OUTPUT/testAero/"
mkpath(outputDir)

# ************************************************
#     Task type
# ************************************************
# --- Set task you want to true ---
run = true # run the solver for a single point
α_sweep = true # sweep angle of attack
U_sweep = true # sweep flow speed
θ_sweep = true # sweep fiber angle

# --- Fill out task details ---
# RUN
if run
    α₀ = 6.0
    U∞ = 10.0
    θ₀ = 0.0
end
# SWEEP AOA
if α_sweep
    α₀ = 0.0:0.5:10.0
    U∞ = 10.0
    θ₀ = 0.0
end
# SWEEP FLOW SPEED
if U_sweep
    α₀ = 6.0
    U∞ = 0.0:0.5:10.0
    θ₀ = 0.0
end
# SWEEP FIBER ANGLE
if θ_sweep
    α₀ = 6.0
    U∞ = 10.0
    θ₀ = 0.0:0.5:10.0
end

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
neval = 30 # spatial nodes
df = 1
fSweep = 0.1:df:100.0 # forcing frequency sweep
# --- Foil from Deniz Akcabay's 2020 paper ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 15.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(neval), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# --- Write the init dict to output folder ---
stringData = JSON.json(DVDict)
open(outputDir * "init_DVDict.json", "w") do io
    write(io, stringData)
end

# ==============================================================================
# Steady solution
# ==============================================================================
SolveSteady.solve(DVDict["neval"], DVDict, outputDir)


# This call is already made in the solve() function
# # --- Write out solution files ---
# SolveSteady.write_sol() # intention here is to get pretty plottable data to visualize in paraview or tecplot

# ==============================================================================
# Dynamic solution
# ==============================================================================
SolveDynamic.solve()