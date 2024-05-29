# --- Julia ---

# @File    :   test_deriv.jl
# @Time    :   2023/03/16
# @Author  :   Galen Ng
# @Desc    :   Derivatives wrt fiber angle for a much more manual check

using Printf # for better file name
using FileIO
using Zygote
using AbstractDifferentiation: AbstractDifferentiation as AD

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
nNodes = 3 # spatial nodes
nNodesStrut = 3 # spatial nodes

# ************************************************
#     Set solver options
# ************************************************
DVDict = Dict(
    "α₀" => 2.0, # initial angle of attack [deg]
    "Λ" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12 * ones(nNodes), # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(-15), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0,
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 0.4, # from Yingqian
    "c_strut" => 0.14 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_αb_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
    "θ_strut" => deg2rad(0), # fiber angle global [rad]
)

wingOptions = Dict(
    "compName" => "akcabay-div",
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
    "name" => "akcabay-div",
    "debug" => debug,
    # --- General solver options ---
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "appendageList" => appendageOptions,
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # --- Static solve ---
    "run_static" => run_static,
    "run_body" => false,
)
# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./test_out/%s_%s_f%.1f_w%.1f/",
    solverOptions["name"],
    wingOptions["material"],
    rad2deg(DVDict["θ"]),
    rad2deg(DVDict["Λ"]))
mkpath(outputDir)
# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]
evalFuncsSensList = ["lift"]

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
steps = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16] # step sizes
dvKey = "θ" # dv to test deriv
dvKey = "Λ" # dv to test deriv
dvKey = "rake" # dv to test deriv
# dvKey = "α₀" # dv to test deriv
dvKey = "toc" # dv to test deriv
evalFunc = "ksflutter"
evalFunc = "lift"
evalFunc = "moment"



# ==============================================================================
#                         STATIC DERIV TESTS
# ==============================================================================
derivs = zeros(length(steps))
funcVal = 0.0
# # ************************************************
# #     Setup test values
# # ************************************************
# DCFoil.init_model([DVDict], evalFuncs; solverOptions=solverOptions)
# _, _, SOLVERPARAMS = DCFoil.SolveStatic.setup_problem(DVDict, wingOptions, solverOptions)
# SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
# u_test = SOL["STATIC"][1].structStates

# # ************************************************
# #     Check ∂r∂x
# # ************************************************
# prpx_fidi = DCFoil.SolveStatic.compute_∂r∂x(u_test, DVDict;
#     mode="FiDi", SOLVERPARAMS=SOLVERPARAMS, appendageOptions=wingOptions, solverOptions=solverOptions)

# prpx_cs = DCFoil.SolveStatic.compute_∂r∂x(u_test, DVDict;
#     mode="CS", SOLVERPARAMS=SOLVERPARAMS, appendageOptions=wingOptions, solverOptions=solverOptions)

# prpx_rad = DCFoil.SolveStatic.compute_∂r∂x(u_test, DVDict;
#     mode="RAD", SOLVERPARAMS=SOLVERPARAMS, appendageOptions=wingOptions, solverOptions=solverOptions)


# # ************************************************
# #     Check ∂f∂u
# # ************************************************
# pfpu_fidi = DCFoil.SolveStatic.compute_∂f∂u("lift", SOL["STATIC"][1], DVDict;
#     mode="FiDi", appendageOptions=wingOptions, solverOptions=solverOptions)

# pfpu_rad = DCFoil.SolveStatic.compute_∂f∂u("lift", SOL["STATIC"][1], DVDict;
#     mode="RAD", appendageOptions=wingOptions, solverOptions=solverOptions)

# # ************************************************
# #     Check ∂f∂x
# # ************************************************
# pfpx_fidi = DCFoil.SolveStatic.compute_∂f∂x("lift", SOL["STATIC"][1], DVDict;
#     mode="FiDi", appendageOptions=wingOptions, solverOptions=solverOptions)

# pfpx_rad = DCFoil.SolveStatic.compute_∂f∂x("lift", SOL["STATIC"][1], DVDict;
#     mode="RAD", appendageOptions=wingOptions, solverOptions=solverOptions)

# # ************************************************
# #     FULL TEST
# # ************************************************
# DCFoil.init_model([DVDict], evalFuncs; solverOptions=solverOptions)
# SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
# costFuncs = DCFoil.evalFuncs(SOL, [DVDict], evalFuncs, solverOptions)
# funcsSensAdjoint = DCFoil.evalFuncsSens(SOL, [DVDict], evalFuncsSensList, solverOptions; mode="Adjoint")
# funcsSensAdjoint[1][dvKey]
# for (ii, dh) in enumerate(steps)
#     DCFoil.init_model([DVDict], evalFuncs; solverOptions=solverOptions)
#     SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
#     costFuncs = DCFoil.evalFuncs(SOL, [DVDict], evalFuncs, solverOptions)
#     flutt_i = costFuncs[evalFuncsSensList[1]*"-"*wingOptions["compName"]]
#     global funcVal = flutt_i
#     DVDict[dvKey][1] += dh
#     SOL = DCFoil.run_model([DVDict], evalFuncs; solverOptions=solverOptions)
#     costFuncs = DCFoil.evalFuncs(SOL, [DVDict], evalFuncs, solverOptions)
#     flutt_f = costFuncs[evalFuncsSensList[1]*"-"*wingOptions["compName"]]

#     derivs[ii] = (flutt_f - flutt_i) / dh
#     @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

#     # --- Reset DV ---
#     DVDict[dvKey][1] -= dh
# end

# save("./FWDDiff.jld2", "derivs", derivs, "steps", steps, "funcVal", funcVal)


# ************************************************
#     Complex step derivative
# ************************************************
# # ---------------------------
# #   drdu
# # ---------------------------
# DVDictList = [DVDict, DVDict]
# _, _, solverParams = DCFoil.SolveStatic.setup_problem(DVDictList, wingOptions, solverOptions)

# mode = "CS"
# structStates = zeros(size(solverParams.Kmat, 1) - length(solverParams.dofBlank))
# @time jacobianCS = DCFoil.SolveStatic.compute_∂r∂u(structStates, mode; 
# DVDictList=DVDictList, solverParams=solverParams, appendageOptions=appendageOptions[1], solverOptions=solverOptions)

# mode = "RAD"
# @time jacobianAD = real(DCFoil.SolveStatic.compute_∂r∂u(structStates, mode; DVDictList=DVDictList, solverParams=solverParams, appendageOptions=appendageOptions[1], solverOptions=solverOptions))

# # mode = "Analytic"
# # @time jacobianAN = DCFoil.SolveStatic.compute_∂r∂u(structStates, mode; DVDictList=DVDictList, solverParams=solverParams, appendageOptions=appendageOptions[1], solverOptions=solverOptions)

# # --- Compare ---
# jacobianCS .- jacobianAD


# We don't need this to be complex step safe just yet. Work on later...
# # ---------------------------
# #   drdx
# # ---------------------------
# DVDictList = [DVDict, DVDict]
# _, _, solverParams = DCFoil.SolveStatic.setup_problem(DVDictList, wingOptions, solverOptions)

# # Check if derivs of setup_problem are correct
# DVVec, DVLengths = DCFoil.Utilities.unpack_dvdict(DVDictList[1])
# DVVecCS = complex(copy(DVVec))
# dh = 1e-100
# for ii in eachindex(DVVec)

#     DVDict = DCFoil.Utilities.repack_dvdict(DVVecCS, DVLengths)
#     DVDictList[1] = DVDict
#     _, _, solverParams_i = DCFoil.SolveStatic.setup_problem(DVDictList, wingOptions, solverOptions)

#     DVVecCS[ii] += dh * 1im
#     DVDict = DCFoil.Utilities.repack_dvdict(DVVecCS, DVLengths)
#     DVDictList[1] = DVDict
#     _, _, solverParams_f = DCFoil.SolveStatic.setup_problem(DVDictList, wingOptions, solverOptions)

# end

# mode = "CS"
# structStates = zeros(size(solverParams.Kmat, 1))
# @time jacobianCS = DCFoil.SolveStatic.compute_∂r∂x(
#     structStates, DVDict;
#     mode=mode, DVDictList=DVDictList, SOLVERPARAMS=solverParams, appendageOptions=appendageOptions[1], solverOptions=solverOptions
# )

# mode = "RAD"
# @time jacobianAD = real(DCFoil.SolveStatic.compute_∂r∂x(
#     structStates, DVDict; mode=mode, DVDictList=DVDictList, SOLVERPARAMS=solverParams, appendageOptions=appendageOptions[1], solverOptions=solverOptions
# ))

# # --- Compare ---
# jacobianCS .- jacobianAD