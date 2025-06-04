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
using ChainRulesCore: ChainRulesCore
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
    "../solvers/SolverSetup",
    "../ComputeForcedFunctions",
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
function solveFromCoords(FEMESH, b_ref, chordVec, claVec, abVec, ebVec, Λ, alfa0, rake, FOIL, CONSTANTS, idxTip, LLSystem, FlowCond, solverOptions::AbstractDict)
    """
    Solve

    (-ω²[M]-jω[C]+[K]){ũ} = {f̃}

    using the ocean wave spectra as the harmonic forcing on the RHS (wave-induced forces)

    claVec: vector of lift slope for each beam node
    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    appendageOptions = solverOptions["appendageList"][1]
    # outputDir = solverOptions["outputDir"]
    # tipForceMag = solverOptions["tipForceMag"]

    # WING, STRUT, CONSTANTS, FEMESH, LLSystem, LLOutputs, FlowCond = setup_problem(LECoords, TECoords, nodeConn, appendageParams, appendageOptions, solverOptions)
    # sweepAng = LLSystem.sweepAng
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped")

    # --- Initialize stuff ---
    u = zeros(size(CONSTANTS.Kmat)[1])
    fSweep = solverOptions["fRange"][1]:solverOptions["df"]:solverOptions["fRange"][end]

    # # --- Tip twist approach ---
    # extForceVec = zeros(size(CONSTANTS.Cmat)[1] - length(DOFBlankingList)) # this is a vector excluded the BC nodes
    # ChainRulesCore.ignore_derivatives() do
    #     extForceVec[end-NDOF+ΘIND] = tipForceMag # this is applying a tip twist
    #     extForceVec[end-NDOF+WIND] = tipForceMag # this is applying a tip lift
    # end
    # --- Wave-induced forces ---
    headingAngle = solverOptions["headingAngle"] # [rad] angle of the wave heading relative to the flow direction, π is headseas
    extForceVec, Aw = compute_fextwave(headingAngle, fSweep * 2π, FEMESH, chordVec, claVec, FlowCond, appendageOptions)

    LiftDyn = zeros(ComplexF64, length(fSweep)) # * 0im
    MomDyn = zeros(ComplexF64, length(fSweep)) # * 0im

    ũout = zeros(ComplexF64, length(fSweep), length(u))

    # --- Initialize transfer functions ---
    # These describe the output deflection relation wrt an input force vector
    GenXferFcn = zeros(ComplexF64, length(fSweep), length(u) - length(DOFBlankingList), length(u) - length(DOFBlankingList))

    # These RAOs describe the outputs relation wrt an input wave amplitude
    LiftRAO = zeros(ComplexF64, length(fSweep))
    MomRAO = zeros(ComplexF64, length(fSweep))

    # Z(ω)
    DeflectionRAO = zeros(ComplexF64, length(fSweep), length(u) - length(DOFBlankingList))
    # H(ω)
    DeflectionMagRAO = zeros(Float64, length(fSweep), length(u) - length(DOFBlankingList))

    dim = NDOF * (size(FEMESH.elemConn)[1] + 1)
    Ms = CONSTANTS.Mmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Ks = CONSTANTS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Cs = CONSTANTS.Cmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]

    maxK = fSweep[end] * 2π * b_ref / (FlowCond.Uinf * cos(LLSystem.sweepAng)) # max reduced frequency
    nK = 22

    globalMf, Cf_r_sweep, Cf_i_sweep, Kf_r_sweep, Kf_i_sweep, kSweep = HydroStrip.compute_genHydroLoadsMatrices(
        maxK, nK, FlowCond.Uinf, b_ref, dim, FEMESH, LLSystem.sweepAng, FOIL, LLSystem, claVec, FlowCond.rhof, FlowCond, ELEMTYPE;
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    # ************************************************
    #     For every frequency, solve the system
    # ************************************************
    tFreq = @elapsed begin
        print_vibration_text(1, 1, 1, 0, 0, 0, 0; printHeader=true)
        for (f_ctr, f) in enumerate(fSweep)

            ω = 2π * f # circular frequency

            # ---------------------------
            #   Assemble hydro matrices
            # ---------------------------
            # # --- Interpolate AICs ---
            k0 = ω * b_ref / (FlowCond.Uinf * cos(LLSystem.sweepAng)) # reduced frequency
            globalCf_r, globalCf_i = HydroStrip.interpolate_influenceCoeffs(k0, kSweep, Cf_r_sweep, Cf_i_sweep, dim, "ng")
            globalKf_r, globalKf_i = HydroStrip.interpolate_influenceCoeffs(k0, kSweep, Kf_r_sweep, Kf_i_sweep, dim, "ng")

            Kf_r, Cf_r, Mf = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, DOFBlankingList)
            Kf_i, Cf_i, _ = HydroStrip.apply_BCs(globalKf_i, globalCf_i, globalMf, DOFBlankingList)

            Cf = Cf_r + 1im * Cf_i
            Kf = Kf_r + 1im * Kf_i

            #  Dynamic matrix. also written as Λ_ij in na520 notes
            D = -1 * ω^2 * (Ms + Mf) + im * ω * (Cf + Cs) + (Ks + Kf)
            # Rows are outputs "i" and columns are inputs "j"

            # ---------------------------
            #   Solve for dynamic states
            # ---------------------------
            fextω = extForceVec[:, f_ctr]
            H = inv(D) # system transfer function matrix for a force
            # qSol = real(H * extForceVec)
            qSol = H * fextω[1:end.∉[DOFBlankingList]]
            ũSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE)
            # uSol = real(ũSol)
            uSol = abs.(ũSol) # proper way to do it

            # Deformations when subjected to the wave amplitude of 1m at all frequencies...not very realistic
            ũout_z[f_ctr, :] = ũSol

            # ---------------------------
            #   Get hydroloads at freq
            # ---------------------------
            fullAIC = -1 * ω^2 * (globalMf) + im * ω * (globalCf_r + 1im * globalCf_i) + (globalKf_r + 1im * globalKf_i)

            fDynamic, L̃, M̃ = HydroStrip.integrate_hydroLoads(uSol, fullAIC, alfa0, rake, DOFBlankingList, CONSTANTS.downwashAngles, ELEMTYPE;
                appendageOptions=appendageOptions, solverOptions=solverOptions)

            # --- Store total force and tip deflection values ---
            LiftDyn_z[f_ctr] = L̃
            MomDyn_z[f_ctr] = M̃

            GenXferFcn[f_ctr, :, :] = H

            DeflectionRAO[f_ctr, :] = ũSol[1:end.∉[DOFBlankingList]] / Aw[f_ctr]
            # DeflectionMagRAO[f_ctr, :] = uSol[1:end.∉[DOFBlankingList]] / Aw[f_ctr]

            if f_ctr % 20 == 1 # header every 10 iterations
                print_vibration_text(f_ctr, f, k0, abs(L̃), abs(M̃), abs(ũSol[end-NDOF+WIND]), abs(ũSol[end-NDOF+ΘIND]))
            end

            # # DEBUG QUIT ON FIRST FREQ
            # break
            f_ctr += 1
        end
    end

    LiftRAO = LiftDyn ./ Aw
    MomRAO = MomDyn ./ Aw

    SOL = ForcedVibSolution(fSweep, ũout, Aw, DeflectionRAO, LiftRAO, MomRAO, GenXferFcn)

    return SOL
end

function print_vibration_text(f_ctr, freq, k0, Lmag, Mmag, tipBend, tipTwist; printHeader=false)
    if printHeader
        println("+-------+-------------+--------------+-------------+-------------+-------------+-------------+")
        println("| nIter | freq_0 [Hz] | Red. freq k0 | mag L̃ (self)| mag M̃ (self)|   w tip [m] | θ tip [rad] |")
        println("+-------+-------------+--------------+-------------+-------------+-------------+-------------+")
    else
        println(@sprintf("|  %04d |   %1.3e | %12.6f | %1.5e | %1.5e | %1.5e | %1.5e |", f_ctr, freq, k0, Lmag, Mmag, tipBend, tipTwist))
    end
end

function compute_fextwave(headingAngle, ωRange, AEROMESH, chordVec, clαVec, FlowCond, appendageOptions)
    """
    Computes the forces and moments due to wave loads on the elevator

    Inputs
    ------
    headingAngle: [rad] angle of the wave heading relative to the flow direction, π is headseas
    """

    # --- Wave loads ---
    ω_wave = 0.125 # Peak wave frequency
    Awsig = 0.5 # Wave amplitude [m]
    ωe = compute_encounterFreq(headingAngle, ωRange, FlowCond.Uinf)

    stripVecs = HydroStrip.get_strip_vecs(AEROMESH, appendageOptions)
    spanLocs = AEROMESH.mesh[:, YDIM]
    nVec = stripVecs
    stripWidths = .√(nVec[:, XDIM] .^ 2 + nVec[:, YDIM] .^ 2 + nVec[:, ZDIM] .^ 2) # length of elem

    fAey, mAey, Aw = compute_waveloads(chordVec, FlowCond.Uinf, FlowCond.rhof, ωe, ωRange, Awsig, FlowCond.depth, stripWidths, clαVec)
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
    # Store solutions here
    workingOutput = outputDir * "forced/"
    mkpath(workingOutput)
    println("Writing out forced solution to ", workingOutput)

    fSweep = SOL.fSweep
    Aw = SOL.Awave
    ũout = SOL.dynStructStates
    DeflectionRAO = SOL.Zdeflection
    LiftRAO = SOL.Zlift
    MomRAO = SOL.Zmom
    GenXferFcn = SOL.RAO

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
function compute_funcsFromDVsOM(evalFunc, ptVec, nodeConn, displacementsCol, claVec, theta_f, toc, alfa0, appendageParams, solverOptions; return_all=false)

    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    appendageParams["theta_f"] = theta_f
    appendageParams["toc"] = toc
    appendageParams["alfa0"] = alfa0

    FEMESH, LLSystem, FlowCond, uRange, b_ref, chordVec, abVec, x_αbVec, ebVec, Λ, FOIL, dim, N_R, N_MAX_Q_ITER, nModes, CONSTANTS, debug = setup_solverOM(displacementsCol, LECoords, TECoords, nodeConn, appendageParams, solverOptions, "HARMONIC FORCED VIBRATION")

    SOL = solveFromCoords(FEMESH, b_ref, chordVec, claVec, abVec, ebVec, Λ, alfa0, 0.0, FOIL, CONSTANTS, FEMESH.idxTip, LLSystem, FlowCond, solverOptions)

    omegaSweep = SOL.fSweep * 2π # [rad/s] frequency sweep in ω_0 space (not encounter frequency)
    # omega_eSweep = compute_encounterFreq(solverOptions["headingAngle"], omegaSweep, FlowCond.Uinf) # compute the encounter frequency
    Hsig = solverOptions["Hsig"] # significant wave height [m]
    if solverOptions["waveSpectrum"] == "ISSC"
        ω_z = solverOptions["omegaz"] # significant zero-crossing frequency [rad/s]
        Swave = compute_BSWaveSpectrum(Hsig, ω_z, omegaSweep)
    elseif solverOptions["waveSpectrum"] == "PM"
        Swave, wm, lm, var, Vwind = compute_PMWaveSpectrum(Hsig, omega_eSweep)
    else
        waveSpectrum = solverOptions["waveSpectrum"]
        error("$(waveSpectrum) wave spectrum not implemented")
    end

    gvib_bend, gvib_twist = compute_PSDArea(SOL, SOL.fSweep, b_ref, Swave)

    ksbend, kstwist = compute_dynDeflectionPk(SOL, solverOptions)
    
    obj = [gvib_bend, gvib_twist, ksbend, kstwist]

    if return_all
        return obj, SOL
    else
        return SOL
    end
end

function evalFuncsSens(VIBSOL;mode="RAD")
    """
    TODO
    """

    if uppercase(mode) == "RAD"
    Zygote.Jacobian
    else
        error("Mode $(mode) not implemented")
    end

end

end # end module