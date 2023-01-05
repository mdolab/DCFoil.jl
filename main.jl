# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Main executable for the project


include("src/DCFoil.jl")

using .DCFoil

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
run_static = false
run_forced = false

# run_static = true
# run_forced = true
run_flutter = true

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
neval = 2 # spatial nodes
df = 1
dU = 1
fSweep = 0.1:df:100.0 # forcing frequency [Hz] sweep
fSearch = 0.01:df:1500.0 # frequency search range [Hz] for flutter modes
uSweep = 0.1:dU:30.0 # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# --- Foil from Deniz Akcabay's 2020 paper ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 10.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(neval), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# --- Yingqian's Viscous FSI Paper (2019) ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.0925 * ones(neval), # chord length [m]
    "s" => 0.2438, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "toc" => 0.03459, # thickness-to-chord ratio
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["cl", "cmy", "lift", "moment"]

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
DCFoil.run_model(DVDict, evalFuncs, run_static, run_forced, run_flutter, fSweep, tipForceMag, uSweep, fSearch, outputDir=outputDir)

