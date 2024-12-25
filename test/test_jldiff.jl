# --- Julia ---

# @File    :   test_deriv.jl
# @Time    :   2023/03/16
# @Author  :   Galen Ng
# @Desc    :   Derivatives wrt fiber angle for a much more manual check

using Printf # for better file name
using FileIO
# using ForwardDiff, FiniteDifferences
using Zygote
include("../src/DCFoil.jl")

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
debug = false
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
fRange = [0.0, 1000.0] # forcing and search frequency sweep [Hz]
# uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
uRange = [170.0, 190.0] # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing


# ************************************************
#     Set solver options
# ************************************************
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "sweep" => deg2rad(-15.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12 * ones(nNodes), # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(15), # fiber angle global [rad]
    # --- Strut vars ---
    "rake" => 0.0,
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 0.4, # from Yingqian
    "c_strut" => 0.14 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)

wingOptions = Dict(
    "compName" => "akcabay-swept",
    "material" => "cfrp", # preselect from material library
    "nNodes" => nNodes,
    "nNodeStrut" => nNodesStrut,
    "config" => "wing",
    "use_tipMass" => tipMass,
    "xMount" => 0.0,
)
appendageOptions = [wingOptions]
solverOptions = Dict(
    # --- I/O ---
    "name" => "akcabay-swept",
    "debug" => debug,
    # --- General solver options ---
    "Uinf" => 5.0, # free stream velocity [m/s]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "appendageList" => appendageOptions,
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => run_static,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fRange" => fRange,
    "tipForceMag" => tipForceMag,
    # --- Eigen solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 100,
    "rhoKS" => 100.0,
)
# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_f%.1f_w%.1f/",
    solverOptions["name"],
    wingOptions["material"],
    rad2deg(DVDict["theta_f"]),
    rad2deg(DVDict["sweep"]))
mkpath(outputDir)
# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]
# evalFuncsSensList = ["lift"]

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
steps = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9] # step sizes
dvKey = "theta_f" # dv to test deriv
dvKey = "sweep" # dv to test deriv
# dvKey = "rake" # dv to test deriv
# dvKey = "alfa0" # dv to test deriv
evalFunc = "ksflutter"
# evalFunc = "lift"


# ************************************************
#     Forward difference checks (dumb way)
# ************************************************
derivs = zeros(length(steps))
funcVal = 0.0

for (ii, dh) in enumerate(steps)
    DCFoil.init_model(DVDict, evalFuncs; solverOptions=solverOptions)
    SOL = DCFoil.run_model(DVDict, evalFuncs; solverOptions=solverOptions)
    costFuncs = DCFoil.evalFuncs(SOL, evalFuncs, solverOptions)
    flutt_i = costFuncs["ksflutter"]
    global funcVal = flutt_i
    DVDict[dvKey] += dh
    SOL = DCFoil.run_model(DVDict, evalFuncs; solverOptions=solverOptions)
    costFuncs = DCFoil.evalFuncs(SOL, evalFuncs, solverOptions)
    flutt_f = costFuncs["ksflutter"]

    derivs[ii] = (flutt_f - flutt_i) / dh
    @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

    # --- Reset DV ---
    DVDict[dvKey] -= dh
end

save("./FWDDiff.jld2", "derivs", derivs, "steps", steps, "funcVal", funcVal)

# ************************************************
#     Does it all work?
# ************************************************
funcsSensAD = DCFoil.SolveFlutter.evalFuncsSens([evalFunc], DVDict, solverOptions; mode="RAD")
# funcsSensFD = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode="FiDi")
save("./RAD.jld2", "derivs", funcsSensAD, "funcVal", funcVal)


