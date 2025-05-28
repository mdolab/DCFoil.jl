# --- Julia---
"""
@File    :   SolveForced.jl
@Time    :   2022/10/07
@Author  :   Galen Ng
@Desc    :   Similar to SolveStatic.jl but now it is a second order dynamical system!
"""

module SolveForced
"""
Frequency domain hydroelastic solver
"""

# --- PACKAGES ---
using LinearAlgebra, Statistics
using JSON
using Zygote
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
using FileIO

# --- DCFoil modules ---
for headerName in [
    "../struct/FEMMethods",
    "../hydro/LiftingLine",
    "../io/MeshIO",
    "../io/TecplotIO",
    "../constants/SolutionConstants",
    "../hydro/HydroStrip",
    "../solvers/SolverRoutines",
    "../constants/DesignConstants",
    "../solvers/DCFoilSolution",
    "../InitModel",
    "../hydro/OceanWaves",
    "../solvers/SetupSolver",
]
    include("$(headerName).jl")
end

using .FEMMethods
using .LiftingLine
using .HydroStrip

# ==============================================================================
#                         COMMON VARIABLES
# ==============================================================================
const loadType = "force"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solveFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions::AbstractDict, appendageOptions::AbstractDict)
    """
    Solve
        (-ω²[M]-jω[C]+[K]){ũ} = {f̃}
    using the 'tipForceMag' as the harmonic forcing on the RHS
    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    outputDir = solverOptions["outputDir"]
    tipForceMag = solverOptions["tipForceMag"]

    WING, STRUT, SOLVERPARAMS, FEMESH, LLSystem, LLOutputs, FlowCond = setup_problem(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)
    sweepAng = LLSystem.sweepAng
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped")
    println("====================================================================================")
    println("        BEGINNING HARMONIC FORCED HYDROELASTIC SOLUTION")
    println("====================================================================================")



    # --- Initialize stuff ---
    u = zeros(size(SOLVERPARAMS.Kmat)[1])
    fSweep = solverOptions["fRange"][1]:solverOptions["df"]:solverOptions["fRange"][end]

    # # --- Tip twist approach ---
    # extForceVec = zeros(size(SOLVERPARAMS.Cmat)[1] - length(DOFBlankingList)) # this is a vector excluded the BC nodes
    # ChainRulesCore.ignore_derivatives() do
    #     extForceVec[end-NDOF+ΘIND] = tipForceMag # this is applying a tip twist
    #     extForceVec[end-NDOF+WIND] = tipForceMag # this is applying a tip lift
    # end
    # --- Wave-induced forces ---
    extForceVec, Aw = compute_fextwave(fSweep * 2π, FEMESH, WING, LLSystem, LLOutputs, FlowCond, appendageParams, appendageOptions)

    LiftDyn = zeros(ComplexF64, length(fSweep)) # * 0im
    MomDyn = zeros(ComplexF64, length(fSweep)) # * 0im

    ũout = zeros(ComplexF64, length(fSweep), length(u))

    # --- Initialize transfer functions ---
    # These describe the output deflection relation wrt an input force vector
    GenXferFcn = zeros(ComplexF64, length(fSweep), length(u) - length(DOFBlankingList), length(u) - length(DOFBlankingList))

    # These RAOs describe the outputs relation wrt an input wave amplitude
    LiftRAO = zeros(ComplexF64, length(fSweep))
    MomRAO = zeros(ComplexF64, length(fSweep))
    DeflectionRAO = zeros(ComplexF64, length(fSweep), length(u) - length(DOFBlankingList))
    DeflectionMagRAO = zeros(Float64, length(fSweep), length(u) - length(DOFBlankingList))

    dim = NDOF * (size(FEMESH.elemConn)[1] + 1)
    Ms = SOLVERPARAMS.Mmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Ks = SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Cs = SOLVERPARAMS.Cmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]

    maxK = fSweep[end] * 2π / FlowCond.Uinf
    nK = 22

    globalMf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, kSweep = HydroStrip.compute_genHydroLoadsMatrices(
        maxK, nK, FlowCond.Uinf, 1.0, dim, FEMESH, LLSystem.sweepAng, WING, LLSystem, LLOutputs, FlowCond.rhof, FlowCond, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    # ************************************************
    #     For every frequency, solve the system
    # ************************************************
    tFreq = @elapsed begin
        for (f_ctr, f) in enumerate(fSweep)
            if f_ctr % 20 == 1 # header every 10 iterations
                println("Forcing: ", f, "Hz")
            end

            ω = 2π * f # circular frequency

            # ---------------------------
            #   Assemble hydro matrices
            # ---------------------------
            # TODO: PICKUP HERE to accelerate forced solution
            # # --- Interpolate AICs ---
            k = ω / FlowCond.Uinf
            globalCf_r, globalCf_i = HydroStrip.interpolate_influenceCoeffs(k, kSweep, Cf_r_sweep, Cf_i_sweep, dim, "ng")
            globalKf_r, globalKf_i = HydroStrip.interpolate_influenceCoeffs(k, kSweep, Kf_r_sweep, Kf_i_sweep, dim, "ng")

            Kf_r, Cf_r, Mf = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, DOFBlankingList)
            Kf_i, Cf_i, _ = HydroStrip.apply_BCs(globalKf_i, globalCf_i, globalMf, DOFBlankingList)

            Cf = Cf_r + 1im * Cf_i
            Kf = Kf_r + 1im * Kf_i

            #  Dynamic matrix. also written as Λ_ij in na520 notes
            D = -1 * ω^2 * (Ms + Mf) + im * ω * (Cf + Cs) + (Ks + Kf)
            # Rows are outputs "i" and columns are inputs "j"

            # # Complex AIC
            # AIC = -1 * ω^2 * (Mf) + im * ω * Cf + (Kf)

            # Store constants
            # global CONSTANTS = SolutionConstants.DCFoilDynamicConstants(D, AIC, extForceVec)
            # global DFOIL = FOIL

            # ---------------------------
            #   Solve for dynamic states
            # ---------------------------
            # The below way is the numerical way to do it but might skip if this doesntwork
            # qSol, _ = SolverRoutines.converge_resNonlinear(compute_residuals, compute_∂r∂u, q, is_cmplx=true, is_verbose=false)

            fextω = extForceVec[:, f_ctr]
            H = inv(D) # system transfer function matrix for a force
            # qSol = real(H * extForceVec)
            qSol = H * fextω[1:end.∉[DOFBlankingList]]
            ũSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE)
            # uSol = real(ũSol)
            uSol = abs.(ũSol) # proper way to do it

            ũout[f_ctr, :] = ũSol

            # ---------------------------
            #   Get hydroloads at freq
            # ---------------------------
            fullAIC = -1 * ω^2 * (globalMf) + im * ω * (globalCf_r + 1im * globalCf_i) + (globalKf_r + 1im * globalKf_i)

            fDynamic, L̃, M̃ = HydroStrip.integrate_hydroLoads(uSol, fullAIC, appendageParams["alfa0"], appendageParams["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE;
                appendageOptions=appendageOptions, solverOptions=solverOptions)

            # --- Store total force and tip deflection values ---
            LiftDyn[f_ctr] = L̃
            MomDyn[f_ctr] = M̃

            GenXferFcn[f_ctr, :, :] = H

            DeflectionRAO[f_ctr, :] = ũSol[1:end.∉[DOFBlankingList]] / Aw[f_ctr]
            # DeflectionMagRAO[f_ctr, :] = uSol[1:end.∉[DOFBlankingList]] / Aw[f_ctr]

            # # DEBUG QUIT ON FIRST FREQ
            # break
            f_ctr += 1
        end
    end

    LiftRAO = LiftDyn ./ Aw
    MomRAO = MomDyn ./ Aw

    SOL = DCFoilSolution.ForcedVibSolution(fSweep, ũout, Aw, DeflectionRAO, LiftRAO, MomRAO, GenXferFcn)

    return SOL
end

function compute_fextwave(ωRange, AEROMESH, WING, LLSystem, LLOutputs, FlowCond, appendageParams, appendageOptions)
    """
    Computes the forces and moments due to wave loads on the elevator
    """

    # --- Wave loads ---
    ω_wave = 0.125 # Peak wave frequency
    Awsig = 0.5 # Wave amplitude [m]
    ωe = OceanWaves.compute_encounterFreq(π, ωRange, FlowCond.Uinf)

    stripVecs = HydroStrip.get_strip_vecs(AEROMESH, appendageOptions)
    spanLocs = AEROMESH.mesh[:, YDIM]
    nVec = stripVecs
    stripWidths = .√(nVec[:, XDIM] .^ 2 + nVec[:, YDIM] .^ 2 + nVec[:, ZDIM] .^ 2) # length of elem
    xeval = LLSystem.collocationPts[YDIM, :]
    claVec = Interpolation.do_linear_interp(xeval, LLOutputs.cla, spanLocs)

    # Elevator chord lengths
    chordLengths = vcat(WING.chord, WING.chord[2:end])
    fAey, mAey, Aw = OceanWaves.compute_waveloads(chordLengths, FlowCond.Uinf, FlowCond.rhof, ωe, ωRange, Awsig, appendageParams["depth0"], stripWidths, claVec)

    extForceVec = zeros(ComplexF64, length(spanLocs) * NDOF, length(ωRange))

    # --- Populate the force vector ---
    # println("shape:$(size(fAey))")
    # println("shape:$(size(extForceVec[WIND:NDOF:end, :]))")
    extForceVec[WIND:NDOF:end, :] .= transpose(fAey)
    extForceVec[ΘIND:NDOF:end, :] .= transpose(mAey)

    return extForceVec, Aw
end

function write_sol(SOL, outputDir="./OUTPUT/")
    """
    Write out the dynamic results
    """

    fSweep = SOL.fSweep
    Aw = SOL.Awave
    ũout = SOL.dynStructStates
    DeflectionRAO = SOL.Zdeflection
    LiftRAO = SOL.Zlift
    MomRAO = SOL.Zmom
    GenXferFcn = SOL.RAO

    workingOutput = outputDir * "forced/"
    mkpath(workingOutput)

    # --- Write frequency sweep ---
    fname = workingOutput * "freqSweep.dat"
    outfile = open(fname, "w")
    for f ∈ fSweep
        write(outfile, string(f) * "\n")
    end
    close(outfile)

    # --- Write tip bending ---
    fname = workingOutput * "tipBendDyn.jld2"
    WVec = ũout[:, WIND:NDOF:end] # bend
    TipBendDyn = abs.(WVec[:, end])
    save(fname, "data", TipBendDyn)

    # --- Write tip twist ---
    fname = workingOutput * "tipTwistDyn.jld2"
    ΦVec = ũout[:, ΦIND:NDOF:end] # bend
    TipTwistDyn = abs.(ΦVec[:, end])
    save(fname, "data", TipTwistDyn)

    # --- Deflection RAO ---
    fname = workingOutput * "deflectionRAO.jld2"
    save(fname, "data", DeflectionRAO)

    # --- Write dynamic lift ---
    fname = workingOutput * "totalLiftRAO.jld2"
    save(fname, "data", (LiftRAO))

    # --- Write dynamic moment ---
    fname = workingOutput * "totalMomentRAO.jld2"
    save(fname, "data", (MomRAO))

    # --- General transfer function ---
    fname = workingOutput * "GenXferFcn.jld2"
    save(fname, "data", GenXferFcn)

    # --- Wave amplitude spectrum A(ω) ---
    fname = workingOutput * "waveAmpSpectrum.jld2"
    save(fname, "data", Aw)

end

function setup_problem(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions; verbose=false)

    tipForceMag = solverOptions["tipForceMag"]

    WING, STRUT, _, FEMESH, LLOutputs, LLSystem, FlowCond = InitModel.init_modelFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions, appendageOptions)
    # ************************************************
    #     Assemble structural matrices
    # ************************************************

    globalKs, globalMs, globalF = FEMMethods.assemble(FEMESH, appendageParams["x_ab"], WING, ELEMTYPE, WING.constitutive; config=appendageOptions["config"], STRUT=STRUT, x_αb_strut=appendageParams["x_ab_strut"], verbose=verbose)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped")
    Ks, Ms, _ = FEMMethods.apply_BCs(globalKs, globalMs, globalF, DOFBlankingList)

    # ---------------------------
    #   Get structural damping
    # ---------------------------
    # these structural damping constants are hidden in this dictionary to keep them constant throughout optimization
    haskey(solverOptions, "alphaConst") || error("solverOptions must contain 'alphaConst'")
    # alphaConst, betaConst = FEMMethods.compute_proportional_damping(real(Ks), real(Ms), appendageParams["zeta"], solverOptions["nModes"])
    alphaConst = solverOptions["alphaConst"]
    betaConst = solverOptions["betaConst"]
    Cs = alphaConst * Ms .+ betaConst * Ks
    globalCs = alphaConst * globalMs .+ betaConst * globalKs

    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalKs, globalMs, globalCs, zeros(2, 2), 0.0, 0.0)

    return WING, STRUT, SOLVERPARAMS, FEMESH, LLSystem, LLOutputs, FlowCond
end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function compute_funcs(evalFunc)

end

function evalFuncsSens(VIBSOL)

end

end # end module