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
# outputDir = "./OUTPUT/testAir/"
# outputDir = "./OUTPUT/testWaterAkcabay/"
# outputDir = "./OUTPUT/IMOCA60KeelCFRP/"
outputDir = "./OUTPUT/IMOCA60KeelSS/"
mkpath(outputDir)

# ************************************************
#     Task type
# ************************************************
# --- Set task you want to true ---
# Defaults
run = true # run the solver for a single point
run_static = false
run_forced = false
run_modal = false
run_flutter = false
debug = false
tipMass = false

# Uncomment here
run_static = true
# run_forced = true
# run_modal = true
# run_flutter = true
# debug = true
tipMass = true

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
    θ₀ = 0.0:10:90.0
end

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
neval = 10 # spatial nodes
nModes = 3 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
df = 1
dU = 1
fSweep = 0.1:df:600.0 # forcing frequency [Hz] sweep
fSearch = 0.01:df:1000.0 # frequency search range [Hz] for flutter modes
uSweep = (5.0:dU:60.0) / 1.9438 # flow speed [m/s] sweep for flutter
# uSweep = 2.0:dU:25.0 # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# --- IMOCA 60 bulb keel ---
DVDict = Dict(
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 50.0 / 1.9438, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1025.0, # fluid density [kg/m³]
    "material" => "ss", # preselect from material library
    "toc" => 0.1, # thickness-to-chord ratio
    # "material" => "cfrp", # preselect from material library
    # "toc" => 0.15, # thickness-to-chord ratio
    "g" => 0.04, # structural damping percentage
    "c" => 0.65 * ones(neval), # chord length [m]
    "s" => 4.0, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => 15 * π / 180, # fiber angle global [rad]
)


# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
DCFoil.run_model(
    DVDict,
    evalFuncs;
    # --- Optional args ---
    run_static=run_static,
    run_forced=run_forced,
    run_modal=run_modal,
    run_flutter=run_flutter,
    fSweep=fSweep,
    tipForceMag=tipForceMag,
    tipMass=tipMass,
    nModes=nModes,
    uSweep=uSweep,
    fSearch=fSearch,
    outputDir=outputDir,
    debug=debug
)

