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
θ_sweep = deg2rad.(0.0:5.0:90.0)

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 20 # spatial nodes
nNodesStrut = 10 # spatial nodes
nModes = 5 # number of flutter and system modes to solve for
df = 1
dU = 1
fSweep = 0.1:df:100.0 # forcing frequency [Hz] sweep
fSearch = 0.01:df:1000.0 # frequency search range [Hz] for flutter modes
uSweep = 5:dU:30.0 # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# --- Yingqian's Viscous FSI Paper (2019) ---
DVDict = Dict(
    "α₀" => 6.0, # initial angle of attack [deg]
    "U∞" => 5.0, # free stream velocity [m/s]
    "Λ" => deg2rad(0.0), # sweep angle [rad]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.0925 * ones(nNodes), # chord length [m]
    "s" => 0.2438, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.03459*ones(nNodes), # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(0), # fiber angle global [rad]
    "s_strut" => 0.4, # from Yingqian
    # --- Strut vars ---
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 0.4, # from Yingqian
    "c_strut" => 0.14 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_αb_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
    "θ_strut" => deg2rad(0), # fiber angle global [rad]
)

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
    "config" => "wing",
    "nNodes" => nNodes, # number of nodes on foil half wing
    "nNodeStrut" => 10, # nodes on strut
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_tipMass" => false,
    # ---------------------------
    #   Flow
    # ---------------------------
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # ---------------------------
    #   Structure
    # ---------------------------
    "material" => "cfrp", # preselect from material library
    # ---------------------------
    #   Solver modes
    # ---------------------------
    # --- Static solve ---
    "run_static" => run_static,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fSweep" => fSweep,
    "tipForceMag" => tipForceMag,
    # --- p-k (Eigen) solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => [1.0,2.0],
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 80.0,
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment"]

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
for theta in θ_sweep
    DVDict["θ"] = theta
    outputDir = @sprintf("./OUTPUT/kramer_theta%02.1f/", rad2deg(theta))
    mkpath(outputDir)
    solverOptions["outputDir"] = outputDir
    DCFoil.init_model(DVDict, evalFuncs; solverOptions=solverOptions)
    DCFoil.run_model(DVDict, evalFuncs;solverOptions=solverOptions)
end
