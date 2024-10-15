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
debug = true
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
# TODO: PICKUP HERE MAKE IT SO DCFOIL WORKS WITH LE AND TE LINES AND THICKNESS DATA
DVDictRudder = Dict(
    "alfa0" => 0.0, # initial angle of attack [deg]
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => ".dat", # chord length [m]
    "s" => 1.0, # semispan [m]
    "ab" => ".dat", # dist from midchord to EA [m]
    "toc" => ".dat", # thickness-to-chord ratio (mean)
    "x_ab" => ".dat", # static imbalance [m]
    "theta_f" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0, # rake angle about top of strut [deg]
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 2.8, # strut span [m]
    "c_strut" => ".dat", # chord length [m]
    "toc_strut" => ".dat", # thickness-to-chord ratio (mean)
    "ab_strut" => ".dat", # dist from midchord to EA [m]
    "x_ab_strut" => ".dat", # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)


DVDictList = [DVDictRudder]

rudderOptions = Dict(
    "compName" => "rudder",
    "config" => "t-foil",
    "nNodes" => nNodes,
    "nNodeStrut" => nNodesStrut,
    # "use_tipMass" => false,
    "xMount" => 3.355,
    "material" => "cfrp", # preselect from material library
    "strut_material" => "cfrp",
    # "path_to_props" => "./INPUT/1DPROPS", # path to 1D properties
    "path_to_props" => nothing,
)

appendageList = [rudderOptions]

solverOptions = Dict(
    # --- I/O ---
    "name" => "R3E6",
    "debug" => false,
    "writeTecplotSolution" => true,
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList" => appendageList,
    "gravityVector" => [0.0, 0.0, -9.81],
    # ---------------------------
    #   Flow
    # ---------------------------
    "Uinf" => 18.0, # free stream velocity [m/s]
    # "Uinf" => 11.0, # free stream velocity [m/s]
    "rhof" => 1025.0, # fluid density [kg/m³]
    "use_nlll" => true, # use non-linear lifting line code
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
    rad2deg(DVDictList[1]["theta_f"]),
    rad2deg(DVDictList[1]["sweep"]))
mkpath(outputDir)

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
DCFoil.init_model(DVDictList, evalFuncs; solverOptions=solverOptions)
SOLDICT = DCFoil.run_model(DVDictList, evalFuncs; solverOptions=solverOptions)
costFuncs = DCFoil.evalFuncs(SOLDICT, DVDictList, evalFuncs, solverOptions)
costFuncsSens = DCFoil.evalFuncsSens(SOLDICT, DVDictList, evalFuncs, solverOptions; mode="ADJOINT")