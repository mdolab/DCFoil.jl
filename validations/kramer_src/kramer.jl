# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Code to verify modal analysis against Kramer's paper

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
run_modal = true
# run_flutter = true

debug = true

# --- Fill out task details ---
θ_sweep = (0.0:10:90.0) * π / 180

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 20 # spatial nodes
nModes = 5 # number of flutter and system modes to solve for
df = 1
dU = 1
fSweep = 0.1:df:100.0 # forcing frequency [Hz] sweep
fSearch = 0.01:df:1000.0 # frequency search range [Hz] for flutter modes
uSweep = 5:dU:30.0 # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# --- Yingqian's Viscous FSI Paper (2019) ---
DVDict = Dict(
    "nNodes" => nNodes,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => 0.0 * π / 180, # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "g" => 0.04, # structural damping percentage
    "c" => 0.0925 * ones(nNodes), # chord length [m]
    "s" => 0.2438, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.03459, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => 0 * π / 180, # fiber angle global [rad]
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
for theta in θ_sweep
    outputDir = @sprintf("./OUTPUT/kramer_theta%02.1f/", (theta * 180 / π))
    mkpath(outputDir)
    DVDict["θ"] = theta
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
end
