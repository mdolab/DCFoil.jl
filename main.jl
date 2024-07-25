# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Desc    :   Main executable for running DCFoil

using Printf, Dates, Profile


# This is the way to import it manually in dev mode
include("src/DCFoil.jl")
using .DCFoil
# Import static package
# using DCFoil

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
# debug = true
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 5 # spatial nodes
nNodesStrut = 5 # spatial nodes
nModes = 4 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
fSweep = range(0.1, 1000.0, 1000) # forcing and search frequency sweep [Hz]
# uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
uRange = [170.0, 190.0] # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# ************************************************
#     Setup solver options
# ************************************************
DVDictRudder = Dict(
    "α₀" => 2.0, # initial angle of attack [deg] (base rake)
    "Λ" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    # "c" => 0.14 * ones(nNodes), # chord length [m]
    "c" => collect(LinRange(0.14, 0.095, nNodes)), # chord length [m]
    "s" => 0.333, # semispan [m]
    "ab" => 0.0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.075 * ones(nNodes), # thickness-to-chord ratio (mean)
    "x_αb" => 0.0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.5, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0, # rake angle about top of strut [deg]
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 1.0, # [m]
    "c_strut" => 0.14 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0.0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_αb_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
    "θ_strut" => deg2rad(0), # fiber angle global [rad]
)


# ************************************************
#     Main T-foil (aka daggerboard)
# ************************************************
# Dimensions are from Day 2019
DVDictMain = Dict(
    "α₀" => 2.0, # initial angle of attack [deg] (base rake)
    "Λ" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => collect(LinRange(0.125, 0.045, nNodes)), # chord length [m]
    "s" => 0.494, # semispan [m]
    "ab" => 0.0 * ones(Float64, nNodes), # dist from midchord to EA [m]
    "toc" => 0.128 * ones(Float64, nNodes), # thickness-to-chord ratio (max from paper)
    "x_αb" => 0.0 * ones(Float64, nNodes), # static imbalance [m]
    "θ" => deg2rad(-15), # fiber angle global [rad]
    # --- Strut vars ---
    "rake" => 0.0, # rake angle about top of strut [deg]
    "depth0" => 0.5, # submerged depth of strut [m] # from Yingqian
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 1.0, # from Yingqian
    "c_strut" => 0.11 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.145 * ones(nNodesStrut), # thickness-to-chord ratio (max from paper)
    "ab_strut" => 0.0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_αb_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
    "θ_strut" => deg2rad(0), # fiber angle global [rad]
)
DVDictList = [DVDictMain, DVDictRudder]

rudderOptions = Dict(
    "compName" => "rudder",
    "config" => "t-foil",
    "nNodes" => nNodes,
    "nNodeStrut" => nNodesStrut,
    # "use_tipMass" => false,
    "xMount" => 3.355,
    "material" => "cfrp", # preselect from material library
    "strut_material" => "cfrp",
)

wingOptions = copy(rudderOptions)
wingOptions["compName"] = "main"
wingOptions["xMount"] = 1.012
appendageList = [wingOptions, rudderOptions]

solverOptions = Dict(
    # ---------------------------
    #   I/O
    # ---------------------------
    # "name" => "akcabay-swept",
    "name" => "t-foil",
    "debug" => debug,
    "writeTecplotSolution" => true,
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList" => appendageList,
    "gravityVector" => [0.0, 0.0, -9.81],
    # ---------------------------
    #   Flow
    # ---------------------------
    "U∞" => 18.0, # free stream velocity [m/s]
    # "U∞" => 11.0, # free stream velocity [m/s]
    "ρ_f" => 1025.0, # fluid density [kg/m³]
    "use_freeSurface" => true,
    "use_cavitation" => false,
    "use_ventilation" => false,
    "use_dwCorrection" => true,
    # ---------------------------
    #   Solver modes
    # ---------------------------
    # --- Static solve ---
    "run_static" => run_static,
    "res_jacobian" => "CS",
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fSweep" => fSweep,
    "tipForceMag" => tipForceMag,
    # --- p-k (Eigen) solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 100.0,
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]

# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_%s_f%.1f_w%.1f/",
    string(Dates.today()),
    solverOptions["name"],
    wingOptions["material"],
    rad2deg(DVDictList[1]["θ"]),
    rad2deg(DVDictList[1]["Λ"]))
mkpath(outputDir)

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
DCFoil.init_model(DVDictList, evalFuncs; solverOptions=solverOptions)
SOLDICT = DCFoil.run_model(DVDictList, evalFuncs; solverOptions=solverOptions)
costFuncs = DCFoil.evalFuncs(SOLDICT, DVDictList, evalFuncs, solverOptions)
costFuncsSens = DCFoil.evalFuncsSens(SOLDICT, DVDictList, evalFuncs, solverOptions; mode="ADJOINT")