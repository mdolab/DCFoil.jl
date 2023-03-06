# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Main executable for the project

using Printf # for better file name
include("src/DCFoil.jl")

using .DCFoil

# ==============================================================================
# Setup hydrofoil model and solver settings
# ==============================================================================
# ************************************************
#     Task type
# ************************************************
# Set task you want to true
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
debug = true
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
neval = 10 # spatial nodes
nModes = 3 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
df = 1
fSweep = 0.1:df:600.0 # forcing frequency [Hz] sweep
uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
# uRange = 2.0:dU:25.0 # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing


# --- IMOCA 60 bulb keel ---
DVDict = Dict(
    "name" => "IMOCA60Keel",
    "neval" => neval,
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 50.0 / 1.9438, # free stream velocity [m/s]
    "Λ" => deg2rad(0.0), # sweep angle [rad]
    "ρ_f" => 1025.0, # fluid density [kg/m³]
    # "material" => "ss", # preselect from material library
    # "toc" => 0.1, # thickness-to-chord ratio
    "material" => "cfrp", # preselect from material library
    "toc" => 0.15, # thickness-to-chord ratio
    "g" => 0.04, # structural damping percentage
    "c" => 0.65 * ones(neval), # chord length [m]
    "s" => 4.0, # semispan [m]
    "ab" => 0 * ones(neval), # dist from midchord to EA [m]
    "x_αb" => 0 * ones(neval), # static imbalance [m]
    "θ" => deg2rad(15), # fiber angle global [rad]
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["w_tip", "psi_tip", "cl", "cmy", "lift", "moment"]

# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_f%.1f_w%.1f/",
    DVDict["name"],
    DVDict["material"],
    rad2deg(DVDict["θ"]),
    rad2deg(DVDict["Λ"]))
mkpath(outputDir)

# ************************************************
#     Set solver options
# ************************************************
solverOptions = Dict(
    # --- I/O ---
    "debug" => debug,
    "outputDir" => outputDir,
    # --- General solver options ---
    "tipMass" => tipMass,
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => run_static,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fSweep" => fSweep,
    "tipForceMag" => tipForceMag,
    # --- Eigen solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
)
# ==============================================================================
#                         Call DCFoil
# ==============================================================================
costFuncs = DCFoil.run_model(
    DVDict,
    evalFuncs;
    # --- Optional args ---
    solverOptions=solverOptions,
)

