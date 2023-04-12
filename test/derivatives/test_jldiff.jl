# --- Julia ---

# @File    :   test_deriv.jl
# @Time    :   2023/03/16
# @Author  :   Galen Ng
# @Desc    :   Derivatives wrt fiber angle

using Printf # for better file name
using JLD
include("../../src/DCFoil.jl")

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
run_modal = true
run_flutter = true
debug = true
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 8 # spatial nodes
nModes = 4 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
df = 1
fSweep = 0.1:df:1000.0 # forcing and search frequency sweep [Hz]
# uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
uRange = [180.0, 190.0] # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing


# ************************************************
#     Set solver options
# ************************************************
DVDict = Dict(
    "α₀" => 6.0, # initial angle of attack [deg]
    "Λ" => deg2rad(-15.0), # sweep angle [rad]
    "g" => 0.04, # structural damping percentage
    "c" => 0.1 * ones(nNodes), # chord length [m]
    "s" => 0.3, # semispan [m]
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.12, # thickness-to-chord ratio
    "x_αb" => 0 * ones(nNodes), # static imbalance [m]
    "θ" => deg2rad(15), # fiber angle global [rad]
)

solverOptions = Dict(
    # --- I/O ---
    "name" => "akcabay-swept",
    "debug" => debug,
    # --- General solver options ---
    "U∞" => 5.0, # free stream velocity [m/s]
    "ρ_f" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "nNodes" => nNodes,
    "config" => "wing",
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
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
    "maxQIter" => 4000,
    "rhoKS" => 80.0,
)
# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_f%.1f_w%.1f/",
    solverOptions["name"],
    solverOptions["material"],
    rad2deg(DVDict["θ"]),
    rad2deg(DVDict["Λ"]))
mkpath(outputDir)
# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
steps = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12] # step sizes
dvKey = "θ" # dv to test deriv
evalFunc = "ksflutter"



# Right now, we require a complete solve to get the derivatives
include("../../src/solvers/SolveFlutter.jl")
include("../../src/InitModel.jl")
include("../../src/struct/FiniteElements.jl")
include("../../src/solvers/SolverRoutines.jl")
include("../../src/hydro/Hydro.jl")
using .SolveFlutter, .InitModel, .FEMMethods, .SolverRoutines, .Hydro
FOIL = InitModel.init_static(DVDict, solverOptions)
nElem = FOIL.nNodes - 1
structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL; config=solverOptions["config"])
# SOL = SolveFlutter.solve(structMesh, elemConn, DVDict, solverOptions)

# # NOTE: I have just been commenting out the following chunks
# # ************************************************
# #     Forward difference checks
# # ************************************************
# funcVal = 0.0f0
# derivs = zeros(length(steps))

# # for (ii, dh) in enumerate(steps)
# derivs = DCFoil.compute_funcSens(SOL, DVDict, evalFunc; mode="FiDi", solverOptions=solverOptions)
# save("./FINDiff.jld", "derivs", derivs[1], "steps", steps, "funcVal", funcVal) # index the first because it returns a tuple
# println("deriv = ", derivs)

# ************************************************
#     Stupid simple unit tests
# ************************************************
dh = steps[4]
bd = 1.0

pkEqnType = "ng"
dim = 3
ω = 0.1
b = 1.0
U∞ = 1.0
DOFBlankingList = [1, 2, 3, 4]
Mf = zeros(Float64, dim, dim)
Cf_r = zeros(Float64, dim, dim)
Kf_r = zeros(Float64, dim, dim)
Kf_i = zeros(Float64, dim, dim)
Cf_i = zeros(Float64, dim, dim)
KK = zeros(Float64, dim, dim)
MM = zeros(Float64, dim, dim)
# Mf[1:dim, 1:dim] .= 1.0
Kf_r[1:dim, 1:dim] .= -1.0
KK[1:dim, 1:dim] .= 10.0
for ii in 1:dim
    MM[ii, ii] = 10.0
end
p_r, p_rd, p_i, p_id, R_aa_r, R_aa_rd, R_aa_i, R_aa_id = SolveFlutter.solve_eigenvalueProblem_d(pkEqnType, dim, b, bd, U∞, FOIL, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK)
# TODO: PICKUP HERE verify derivatives here

# # ************************************************
# #     FAD checks
# # ************************************************
# SOL = 1
# solverOptions["debug"] = false # You have to turn debug off for RAD to work
# derivs = DCFoil.compute_funcSens(SOL, DVDict, evalFunc; mode="FAD", solverOptions=solverOptions)
# save("./RADDiff.jld", "derivs", derivs[1], "steps", steps, "funcVal", funcVal)
# println("deriv = ", derivs)


# # ************************************************
# #     RAD checks
# # ************************************************
# SOL = 1
# solverOptions["debug"] = false # You have to turn debug off for RAD to work
# derivs = DCFoil.compute_funcSens(SOL, DVDict, evalFunc; mode="RAD", solverOptions=solverOptions)
# save("./RADDiff.jld", "derivs", derivs[1], "steps", steps, "funcVal", funcVal)
# println("deriv = ", derivs)

# # ************************************************
# #     Forward difference checks (dumb way)
# # ************************************************
# funcVal = 0.0f0
# derivs = zeros(length(steps))
# for (ii, dh) in enumerate(steps)
#     costFuncs = DCFoil.run_model(
#         DVDict,
#         evalFuncs;
#         solverOptions=solverOptions
#     )
#     flutt_i = costFuncs["ksflutter"]
#     funcVal = flutt_i
#     DVDict[dvKey] += dh
#     costFuncs = DCFoil.run_model(
#         DVDict,
#         evalFuncs;
#         # --- Optional args ---
#         solverOptions=solverOptions
#     )
#     flutt_f = costFuncs["ksflutter"]

#     derivs[ii] = (flutt_f - flutt_i) / dh
#     @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

#     # --- Reset DV ---
#     DVDict[dvKey] -= dh
# end

# save("./FWDDiff.jld", "derivs", derivs, "steps", steps, "funcVal", funcVal)
# # ************************************************
# #     Central difference checks
# # ************************************************
# funcVal = 0.0f0
# derivs = zeros(length(steps))
# for (ii, dh) in enumerate(steps)

#     # Evaluate f_i
#     costFuncs = DCFoil.run_model(
#         DVDict,
#         evalFuncs;
#         solverOptions=solverOptions
#     )
#     flutt_i = costFuncs["ksflutter"]
#     funcVal = flutt_i
#     # ---------------------------
#     #   Evaluate f_i-1
#     # ---------------------------
#     DVDict[dvKey] -= dh
#     costFuncs = DCFoil.run_model(
#         DVDict,
#         evalFuncs;
#         # --- Optional args ---
#         solverOptions=solverOptions
#     )
#     flutt_b = costFuncs["ksflutter"]
#     # ---------------------------
#     #   Evalute f_i+1
#     # ---------------------------
#     DVDict[dvKey] += 2 * dh
#     costFuncs = DCFoil.run_model(
#         DVDict,
#         evalFuncs;
#         # --- Optional args ---
#         solverOptions=solverOptions
#     )
#     flutt_f = costFuncs["ksflutter"]


#     # --- Central difference scheme ---
#     derivs[ii] = (flutt_f - flutt_b) / (2 * dh)
#     @sprintf("dh = %f, deriv = %f", dh, derivs[ii])

#     # --- Reset DV ---
#     DVDict[dvKey] -= dh
# end

# save("./CENTDiff.jld", "derivs", derivs, "steps", steps, "funcVal", funcVal)
