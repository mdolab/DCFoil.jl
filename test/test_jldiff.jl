# --- Julia ---

# @File    :   test_deriv.jl
# @Time    :   2023/03/16
# @Author  :   Galen Ng
# @Desc    :   Derivatives wrt fiber angle for a much more manual check

using Printf # for better file name
using FileIO
using ForwardDiff, FiniteDifferences, Zygote
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
debug = false
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 5 # spatial nodes
nModes = 4 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
fSweep = range(0.1, 1000.0, 1000) # forcing and search frequency sweep [Hz]
# uRange = [5.0, 50.0] / 1.9438 # flow speed [m/s] sweep for flutter
uRange = [170.0, 190.0] # flow speed [m/s] sweep for flutter
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
    "use_tipMass" => tipMass,
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
    "maxQIter" => 100,
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

# # Right now, we require a complete solve to get the derivatives
include("../../src/solvers/SolveFlutter.jl")
include("../../src/InitModel.jl")
include("../../src/struct/FiniteElements.jl")
include("../../src/solvers/SolverRoutines.jl")
include("../../src/hydro/HydroStrip.jl")
include("../../src/constants/SolutionConstants.jl")
using .SolveFlutter, .InitModel, .FEMMethods, .SolverRoutines, .HydroStrip, .SolutionConstants
# FOIL = InitModel.init_model_wrapper(DVDict, solverOptions; uRange=solverOptions["uRange"], fSweep=solverOptions["fSweep"])
# nElem = FOIL.nNodes - 1
# structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"])
# abVec = DVDict["ab"]
# x_αbVec = DVDict["x_αb"]
# chordVec = DVDict["c"]
# ebVec = 0.25 * chordVec .+ abVec
# globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, "BT2", FOIL.constitutive)
# Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, [1, 2, 3, 4])
# global CONSTANTS = SolutionConstants.DCFoilConstants(Ks, Ms, "BT2", structMesh, zeros(2, 2), "FAD", 0.0)

# # ************************************************
# #     Stupid simple unit tests
# # ************************************************
# dh = steps[6]
# bd = 1.0

# pkEqnType = "ng"
# dim = size(globalKs, 1) #big probl
# ω = 0.1
# b = 1.0
# U∞ = 1.0
# DOFBlankingList = [1, 2, 3, 4]
# Mf = zeros(Float64, dim, dim)
# Cf_r = zeros(Float64, dim, dim)
# Kf_r = zeros(Float64, dim, dim)
# Kf_i = zeros(Float64, dim, dim)
# Cf_i = zeros(Float64, dim, dim)
# KK = zeros(Float64, dim, dim)
# MM = zeros(Float64, dim, dim)
# # Mf[1:dim, 1:dim] .= 1.0
# Kf_r[1:dim, 1:dim] .= -1.0
# KK[1:dim, 1:dim] .= 10.0
# for ii in 1:dim
#     MM[ii, ii] = 10.0
# end

# # ---------------------------
# #   FD check
# # ---------------------------
# w_r, w_i, _, _, VR_r, VR_i = SolverRoutines.cmplxStdEigValProb(A_r, A_i, dim)
# A_r[2, 1] += dh
# # A_i[1, 1] += dh
# w_rf, w_i, _, _, VR_rf, VR_if = SolverRoutines.cmplxStdEigValProb(A_r, A_i, dim)
# fd = (w_rf - w_r) ./ dh
# fdVr = (VR_rf - VR_r) ./ dh
# A_r[2, 1] -= dh
# println("FD:")
# println("d w_r / d A_r ", fd)
# println("d VR_r / d A_r ", fdVr)

# ==============================================================================
#                         FLUTTER DERIV TESTS
# ==============================================================================
# ************************************************
#     RAD checks
# ************************************************
# SOL = 1
# solverOptions["debug"] = false # You have to turn debug off for RAD to work
# Mr, Kr, Qr = SolveFlutter.compute_modalSpace(Ms, Ks; reducedSize=dim - 4) # do it on already BC applied matrices
# tmpFactor = U∞ * cos(DVDict["Λ"]) / b_ref
# div_tmp = 1 / tmpFactor
# ωSweep = 2π * FOIL.fSweep # sweep of circular frequencies
# kSweep = ωSweep * div_tmp

# ************************************************
#     pk flutter derivatives
# ************************************************
# NOTE: why can't I compile this by itself? ANS: correlation mat deriv but it's fixed now
# derivs, = Zygote.jacobian(x -> SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, x, Λ, chordVec, abVec, ebVec, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5), b_ref)
# primal = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref, Λ, chordVec, abVec, ebVec, FOIL, dim, N_R, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5)
# derivs = Zygote.jacobian((x1, x2, x3, x4, x5, x6) ->
#         SolveFlutter.compute_pkFlutterAnalysis(
#             uRange, x1, x2, x3, x4, x5, x6, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, Ms, Ks; Δu=0.5),
#     structMesh, b_ref, Λ, chordVec, abVec, ebVec)
# fdderivs1, = FiniteDifferences.jacobian(central_fdm(2, 1), (x) ->
#         SolveFlutter.compute_pkFlutterAnalysis(
#             uRange, x, b_ref, Λ, chordVec, abVec, ebVec, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5),
#     structMesh)
# fdderivs2, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) ->
#         SolveFlutter.compute_pkFlutterAnalysis(
#             uRange, structMesh, x, Λ, chordVec, abVec, ebVec, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5),
#     b_ref)


# # ************************************************
# #     KS aggregation derivatives
# # ************************************************

# structMesh, elemConn, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug = SolveFlutter.setup_solver(0.1, 0.0, 1.0, DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_αb"], DVDict["g"], DVDict["θ"], solverOptions)

# p_r, p_i, true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory, NTotalModesFound, nFlow = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref, Λ, chordVec, abVec, ebVec, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5)
# derivs, = Zygote.jacobian((x) -> SolveFlutter.postprocess_damping(N_MAX_Q_ITER, flowHistory, NTotalModesFound, nFlow, x, iblank, 80), p_r)
# fdderivs, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) -> SolveFlutter.postprocess_damping(N_MAX_Q_ITER, flowHistory, NTotalModesFound, nFlow, x, iblank, 80), p_r)
# # f1 = SolveFlutter.postprocess_damping(N_MAX_Q_ITER, flowHistory, NTotalModesFound, nFlow, true_eigs_r, iblank, 80)
# # f2 = SolveFlutter.postprocess_damping(N_MAX_Q_ITER, flowHistory, NTotalModesFound, nFlow, true_eigs_r .+ 1e-8, iblank, 80)

# derivs = Zygote.jacobian((x1, x2, x3) -> SolveFlutter.solve(
#         x1, solverOptions, uRange, x2, x3, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug),
#     structMesh, b_ref, chordVec)
# fdderivs, = FiniteDifferences.jacobian(central_fdm(3, 1), (x1) -> SolveFlutter.solve(structMesh, solverOptions, uRange, x1, chordVec, abVec, ebVec, Λ, FOIL, dim, N_R, globalDOFBlankingList, N_MAX_Q_ITER, nModes, CONSTANTS, debug), b_ref)
# # derivs = DCFoil.compute_funcSens(SOL, DVDict, evalFunc; mode="RAD", solverOptions=solverOptions)
# # save("./RADDiff.jld2", "derivs", derivs[1], "steps", steps, "funcVal", funcVal)
# # println("deriv = ", derivs)

# # THIS IS THE FINAL STEP
# # ************************************************
# #     are the hydro slopes correct?
# # ************************************************
# # TODO: no they are not, this test shows that
# derivs, = Zygote.jacobian((x) ->
#         HydroStrip.compute_glauert_circ(x, ones(5), 0.2, 6.0, 5, nothing, false), 1.0)
# fdderivs, = FiniteDifferences.jacobian(central_fdm(3, 1), (x) ->
#         HydroStrip.compute_glauert_circ(x, ones(5), 0.2, 6.0, 5), 1.0)

# ************************************************
#     Forward difference checks (dumb way)
# ************************************************
derivs = zeros(length(steps))
funcVal = 0.0f0
for (ii, dh) in enumerate(steps)
    costFuncs = DCFoil.run_model(
        DVDict,
        evalFuncs;
        solverOptions=solverOptions
    )
    flutt_i = costFuncs["ksflutter"]
    funcVal = flutt_i
    DVDict[dvKey] += dh
    costFuncs = DCFoil.run_model(
        DVDict,
        evalFuncs;
        # --- Optional args ---
        solverOptions=solverOptions
    )
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
funcsSensAD = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode="RAD")
funcsSensFD = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode="FiDi")
save("./RAD.jld2", "derivs", funcsSensAD, "funcVal", funcVal)
