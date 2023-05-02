# --- Julia ---

# @File    :   test_deriv.jl
# @Time    :   2023/03/16
# @Author  :   Galen Ng
# @Desc    :   Derivatives wrt fiber angle

using Printf # for better file name
using JLD
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
include("../../src/constants/SolutionConstants.jl")
using .SolveFlutter, .InitModel, .FEMMethods, .SolverRoutines, .Hydro, .SolutionConstants
# FOIL = InitModel.init_static(DVDict, solverOptions)
FOIL = InitModel.init_dynamic(DVDict, solverOptions; uRange=solverOptions["uRange"], fSweep=solverOptions["fSweep"])
nElem = FOIL.nNodes - 1
structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL; config=solverOptions["config"])
globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, "BT2", FOIL.constitutive)
Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, [1, 2, 3, 4])
global CONSTANTS = SolutionConstants.DCFoilConstants(Ks, Ms, "BT2", structMesh, zeros(2, 2), "FAD", 0.0)
# obj, pmg, SOL = SolveFlutter.solve(structMesh, elemConn, DVDict, solverOptions)

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
dh = steps[6]
bd = 1.0

pkEqnType = "ng"
dim = size(globalKs, 1) #big probl
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
solverOptions["debug"] = false # You have to turn debug off for RAD to work
b_ref = sum(FOIL.c) / FOIL.nNodes # mean semichord
Mr, Kr, Qr = SolveFlutter.compute_modalSpace(Ms, Ks; reducedSize=dim - 4) # do it on already BC applied matrices
tmpFactor = U∞ * cos(FOIL.Λ) / b_ref
div_tmp = 1 / tmpFactor
ωSweep = 2π * FOIL.fSweep # sweep of circular frequencies
kSweep = ωSweep * div_tmp

# Progressively smaller unit tests!
# TODO: PICKUP HERE WHY IS THE RAD SO SLOW??
# globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = Hydro.compute_AICs!(Mf, Cf_r, Cf_i, Kf_r, Kf_i, structMesh, FOIL.Λ, FOIL, U∞, ωSweep[1], "BT2")
# Kffull_r, Cffull_r, _ = Hydro.apply_BCs(globalKf_r, globalCf_r, globalMf, [1,2,3,4])
# Kffull_i, Cffull_i, _ = Hydro.apply_BCs(globalKf_i, globalCf_i, globalMf, [1,2,3,4])
# # Mode space reduction
# Kf_r = Qr' * Kffull_r * Qr
# Kf_i = Qr' * Kffull_i * Qr
# Cf_r = Qr' * Cffull_r * Qr
# Cf_i = Qr' * Cffull_i * Qr
# SolveFlutter.solve_eigenvalueProblem(pkEqnType, dimwithBC, b_ref, U∞, FOIL, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK)
# derivs, = Zygote.jacobian(x -> SolveFlutter.sweep_kCrossings(dim, kSweep, x, FOIL.Λ, U∞, Mr, Kr, Qr, structMesh, FOIL, [1, 2, 3, 4], 5000), b_ref)
# func = SolveFlutter.sweep_kCrossings(dim, kSweep, b_ref, FOIL.Λ, U∞, Mr, Kr, Qr, structMesh, FOIL, [1, 2, 3, 4], 5000)
# funcd = SolveFlutter.sweep_kCrossings(dim, kSweep, b_ref+1e-8, FOIL.Λ, U∞, Mr, Kr, Qr, structMesh, FOIL, [1, 2, 3, 4], 5000)

# derivs, = Zygote.jacobian(x -> SolveFlutter.compute_kCrossings(dim, kSweep, x, FOIL.Λ, FOIL, U∞, Mr, Kr, Qr, structMesh, [1, 2, 3, 4]; debug=false, qiter=1), b_ref)
# func = SolveFlutter.compute_kCrossings(dim, kSweep, b_ref, FOIL.Λ, FOIL, U∞, Mr, Kr, Qr, structMesh, [1, 2, 3, 4]; debug=false, qiter=1)
# funcd = SolveFlutter.compute_kCrossings(dim, kSweep, b_ref + 1e-8, FOIL.Λ, FOIL, U∞, Mr, Kr, Qr, structMesh, [1, 2, 3, 4]; debug=false, qiter=1)

uRange = [187.0, 190.0] # flow speed [m/s] sweep for flutter
N_MAX_Q_ITER = 2
# obj, pmG, FLUTTERSOL = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], 1000, 3, Ms, Ks; Δu=0.4)
derivs, = Zygote.jacobian(x -> SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, x, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5), b_ref)
# true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory, NTotalModesFound, nFlow = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], 100, 3, Ms, Ks; Δu=0.5)

# func = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5)
# funcd1 = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref + 1e-8, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5)
# funcd2 = SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, b_ref + 1e-4, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5)
fdderivs, = FiniteDifferences.jacobian(central_fdm(3, 1), (x)-> SolveFlutter.compute_pkFlutterAnalysis(uRange, structMesh, x, FOIL.Λ, FOIL, dim, 8, [1, 2, 3, 4], N_MAX_Q_ITER, 3, Ms, Ks; Δu=0.5), b_ref)

# derivs = DCFoil.compute_funcSens(SOL, DVDict, evalFunc; mode="RAD", solverOptions=solverOptions)
# save("./RADDiff.jld", "derivs", derivs[1], "steps", steps, "funcVal", funcVal)
# println("deriv = ", derivs)

# # ---------------------------
# #   Test simple matrix operations for 3x3 matrices
# # ---------------------------
# # derivs = Zygote.jacobian(x -> A*x, rand(3, 3))
# # derivs = Zygote.jacobian(x -> inv(x), rand(3, 3))
# n = 2
# A_r = rand(n, n)
# A_i = rand(n, n)
# derivs, = Zygote.jacobian((x_r, x_i) -> SolverRoutines.cmplxStdEigValProb2(x_r, x_i, n), A_r, A_i)
# # Unpack derivatives properly
# w_r_jac = derivs[1:n, :]
# w_i_jac = derivs[n+1:n^2, :]
# # All of the eigenvector derivatives are wrong :(
# VR_r_jac = derivs[n^2+1:2n^2, :]
# VR_i_jac = derivs[2n^2+1:end, :]

# # --- FD check ---
# FDJac, = FiniteDifferences.jacobian(central_fdm(3, 1), (x_r, x_i) -> SolverRoutines.cmplxStdEigValProb2(x_r, x_i, n), A_r, A_i)

# FDJacN = zeros(2 * n + 2 * n^2, n^2 * 2)
# y = SolverRoutines.cmplxStdEigValProb2(A_r, A_i, n)
# dh = 1e-8
# A_r[1, 1] += dh
# dy = SolverRoutines.cmplxStdEigValProb2(A_r, A_i, n)
# FDJacN[:, 1] = (dy - y) / dh
# A_r[1, 1] -= dh
# println("RADJac=", derivs[:, 1])
# println("FDJac = ", FDJac[:, 1])
# println("FDJacN = ", FDJacN[:, 1])


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
