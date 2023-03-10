# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Verify against deniz's plate

using Printf
include("../../src/DCFoil.jl")

using .DCFoil

# ==============================================================================
# Setup hydrofoil model and solver settings
# ==============================================================================
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

# Uncomment here
# run_static = true
# run_forced = true
# run_modal = true
run_flutter = true

debug = true

# --- Fill out task details ---
U_sweep = 1.0:1.0:40.0

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 20 # spatial nodes
nModes = 4 # number of modes to solve for;
df = 1
dU = 1
fSweep = 0.1:df:100.0 # forcing frequency [Hz] sweep
fSearch = 0.01:df:1000.0 # frequency search range [Hz] for flutter modes
uSweep = 25.0:dU:40 # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# --- Foil from Deniz Akcabay's 2020 paper ---
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 29.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => -15 * π / 180, # fiber angle global [rad]
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

# ==============================================================================
#                         Call DCFoil TWICE
# ==============================================================================
outputDir = @sprintf("./OUTPUT/akcabay_f-15_w0/")
mkpath(outputDir)
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
    nModes=nModes,
    uSweep=uSweep,
    fSearch=fSearch,
    outputDir=outputDir,
    debug=debug
)

outputDir = @sprintf("./OUTPUT/akcabay_f+15_w-15/")
mkpath(outputDir)
DVDict["θ"] = 15 * π / 180
DVDict["Λ"] = -15 * π / 180
uSweep = 170:dU:190 # flow speed [m/s] sweep for flutter
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
    nModes=nModes,
    uSweep=uSweep,
    fSearch=fSearch,
    outputDir=outputDir,
    debug=debug
)
