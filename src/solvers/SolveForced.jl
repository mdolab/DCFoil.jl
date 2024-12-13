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

# --- Public functions ---
export solve

# --- PACKAGES ---
using LinearAlgebra, Statistics
using JSON
using Zygote
using ChainRulesCore: @ignore_derivatives
using FileIO

# --- DCFoil modules ---
using ..InitModel, ..HydroStrip, ..BeamProperties
using ..FEMMethods
using ..SolveStatic
using ..SolutionConstants
using ..SolverRoutines
using ..EBBeam: NDOF, UIND, VIND, WIND, ΦIND, ΨIND, ΘIND
using ..DCFoilSolution
using ..OceanWaves

# ==============================================================================
#                         COMMON VARIABLES
# ==============================================================================
const ELEMTYPE = "COMP2"
const loadType = "force"

# ==============================================================================
#                         Top level API routines
# ==============================================================================
# function solve(FEMESH, DVDict, solverOptions::Dict, appendageOptions::Dict)
#     """
#     Solve
#         (-ω²[M]-jω[C]+[K]){ũ} = {f̃}
#     using the 'tipForceMag' as the harmonic forcing on the RHS
#     """
#     # ---------------------------
#     #   Initialize
#     # ---------------------------
#     outputDir = solverOptions["outputDir"]
#     fRange = solverOptions["fRange"]
#     tipForceMag = solverOptions["tipForceMag"]
#     global FOIL, STRUT, _ = InitModel.init_modelFromDVDict(DVDict, solverOptions, appendageOptions; fRange=fRange)

#     println("====================================================================================")
#     println("        BEGINNING HARMONIC FORCED HYDROELASTIC SOLUTION")
#     println("====================================================================================")

#     # ************************************************
#     #     Assemble structural matrices
#     # ************************************************

#     abVec = DVDict["ab"]
#     x_αbVec = DVDict["x_ab"]
#     chordVec = DVDict["c"]
#     ebVec = 0.25 * chordVec .+ abVec
#     Λ = DVDict["sweep"]
#     U∞ = solverOptions["Uinf"]
#     α₀ = DVDict["alfa0"]
#     rake = DVDict["rake"]
#     zeta = DVDict["zeta"]
#     structMesh = FEMESH.mesh
#     elemConn = FEMESH.elemConn
#     globalKs, globalMs, globalF = FEMMethods.assemble(FEMESH, x_αbVec, FOIL, ELEMTYPE, FOIL.constitutive)

#     # ---------------------------
#     #   Apply BC blanking
#     # ---------------------------
#     globalDOFBlankingList = 0
#     @ignore_derivatives() do
#         globalDOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped")
#     end
#     Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, globalDOFBlankingList)

#     # ---------------------------
#     #   Get damping
#     # ---------------------------
#     alphaConst, betaConst = FEMMethods.compute_proportional_damping(Ks, Ms, zeta, solverOptions["nModes"])
#     Cs = alphaConst * Ms .+ betaConst * Ks

#     # --- Initialize stuff ---
#     u = copy(globalF)
#     globalMf = copy(globalMs) * 0
#     globalCf_r = copy(globalKs) * 0
#     globalKf_r = copy(globalKs) * 0
#     globalKf_i = copy(globalKs) * 0
#     globalCf_i = copy(globalKs) * 0
#     extForceVec = copy(F) * 0 # this is a vector excluded the BC nodes
#     extForceVec[end-1] = tipForceMag # this is applying a tip twist
#     extForceVec[end-3] = tipForceMag # this is applying a tip lift
#     LiftDyn = zeros(length(fSweep)) # * 0im
#     MomDyn = zeros(length(fSweep)) # * 0im
#     TipBendDyn = zeros(length(fSweep)) # * 0im
#     TipTwistDyn = zeros(length(fSweep)) # * 0im
#     RAO = zeros(ComplexF64, length(fSweep), length(F), length(F))

#     # ---------------------------
#     #   Pre-solve system
#     # ---------------------------
#     q = FEMMethods.solve_structure(Ks, Ms, F)

#     # --- Populate displacement vector ---
#     u[globalDOFBlankingList] .= 0.0
#     idxNotBlanked = [x for x ∈ eachindex(u) if x ∉ globalDOFBlankingList] # list comprehension
#     u[idxNotBlanked] .= q

#     # ************************************************
#     #     For every frequency, solve the system
#     # ************************************************
#     f_ctr = 1
#     @time for f in fSweep

#         if f_ctr % 20 == 1 # header every 10 iterations
#             println("Forcing: ", f, "Hz")
#         end

#         ω = 2π * f # circular frequency

#         # ---------------------------
#         #   Assemble hydro matrices
#         # ---------------------------
#         # globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(globalMf_0, globalCf_r_0, globalCf_i_0, globalKf_r_0, globalKf_i_0, structMesh, FOIL, FOIL.U∞, ω, elemType)
#         # globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType)
#         globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(FEMESH, size(globalMs)[1], Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, ELEMTYPE)
#         Kf_r, Cf_r, Mf = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
#         Kf_i, Cf_i, _ = HydroStrip.apply_BCs(globalKf_i, globalCf_i, globalMf, globalDOFBlankingList)

#         Cf = Cf_r + 1im * Cf_i
#         Kf = Kf_r + 1im * Kf_i

#         #  Dynamic matrix
#         D = -1 * ω^2 * (Ms + Mf) + im * ω * (Cf + Cs) + (Ks + Kf)

#         # Complex AIC
#         AIC = -1 * ω^2 * (Mf) + im * ω * Cf + (Kf)

#         # Store constants
#         global CONSTANTS = SolutionConstants.DCFoilDynamicConstants(ELEMTYPE, structMesh, D, AIC, extForceVec)
#         # global DFOIL = FOIL

#         # ---------------------------
#         #   Solve for dynamic states
#         # ---------------------------
#         # The below way is the numerical way to do it but might skip if this doesntwork
#         # qSol, _ = SolverRoutines.converge_resNonlinear(compute_residuals, compute_∂r∂u, q, is_cmplx=true, is_verbose=false)
#         H = inv(D) # RAO
#         qSol = real(H * extForceVec)
#         uSol, _ = FEMMethods.put_BC_back(qSol, CONSTANTS.elemType)

#         # ---------------------------
#         #   Get hydroloads at freq
#         # ---------------------------
#         fullAIC = -1 * ω^2 * (globalMf) + im * ω * (globalCf_r + 1im * globalCf_i) + (globalKf_r + 1im * globalKf_i)
#         fDynamic, DynLift, DynMoment = HydroStrip.integrate_hydroLoads(uSol, fullAIC, α₀, rake, CONSTANTS.elemType; solverOptions=solverOptions)#compute_hydroLoads(uSol, fullAIC)

#         # --- Store total force and tip deflection values ---
#         LiftDyn[f_ctr] = (DynLift)
#         MomDyn[f_ctr] = (DynMoment)
#         RAO[f_ctr, :, :] = H
#         if ELEMTYPE == "BT2"
#             TipBendDyn[f_ctr] = (uSol[end-3])
#             TipTwistDyn[f_ctr] = (uSol[end-1])
#             phaseAngle = angle(uSol[end-3])
#         elseif ELEMTYPE == "COMP2"
#             TipBendDyn[f_ctr] = (uSol[end-6])
#             TipTwistDyn[f_ctr] = (uSol[end-4])
#             phaseAngle = angle(uSol[end-6])
#         else
#             println("Invalid element type")
#         end

#         # # DEBUG QUIT ON FIRST FREQ
#         # break
#         f_ctr += 1
#     end


#     # ************************************************
#     #     Write solution out to files
#     # ************************************************
#     write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, RAO, outputDir)

#     # TODO:
#     costFuncs = nothing
#     # costFuncs = SolverRoutines.compute_costFuncs()

#     return costFuncs
# end

function solveFromCoords(LECoords, TECoords, nodeConn, appendageParams, solverOptions::Dict, appendageOptions::Dict)
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
    DOFBlankingList = FEMMethods.get_fixed_dofs(ELEMTYPE, "clamped")
    println("====================================================================================")
    println("        BEGINNING HARMONIC FORCED HYDROELASTIC SOLUTION")
    println("====================================================================================")



    # --- Initialize stuff ---
    u = zeros(size(SOLVERPARAMS.Kmat)[1])
    fSweep = solverOptions["fRange"][1]:solverOptions["df"]:solverOptions["fRange"][end]

    # --- Tip twist approach ---
    extForceVec = zeros(size(SOLVERPARAMS.Cmat)[1]) # this is a vector excluded the BC nodes
    @ignore_derivatives() do
        extForceVec[end-NDOF+ΘIND] = tipForceMag # this is applying a tip twist
        extForceVec[end-NDOF+WIND] = tipForceMag # this is applying a tip lift
    end
    # TODO: PICKUP here
    extForceVec = compute_fextwave(fSweep * 2π,)

    LiftDyn = zeros(ComplexF64, length(fSweep)) # * 0im
    MomDyn = zeros(ComplexF64, length(fSweep)) # * 0im
    TipBendDyn = zeros(length(fSweep)) # * 0im
    TipTwistDyn = zeros(length(fSweep)) # * 0im

    ũout = zeros(ComplexF64, length(fSweep), length(u))
    GenXferFcn = zeros(ComplexF64, length(fSweep), length(u) - length(DOFBlankingList), length(u) - length(DOFBlankingList))
    DeflectionXferFcn = zeros(ComplexF64, length(fSweep), length(u) - length(DOFBlankingList))

    dim = NDOF * (size(FEMESH.elemConn)[1] + 1)
    Ms = SOLVERPARAMS.Mmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Ks = SOLVERPARAMS.Kmat[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]]
    Cs = SOLVERPARAMS.Cmat

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
            globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(FEMESH, WING, LLSystem, LLOutputs, FlowCond.rhof, dim, appendageParams["sweep"], FlowCond.Uinf, ω, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, use_nlll=solverOptions["use_nlll"])
            Kf_r, Cf_r, Mf = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, DOFBlankingList)
            Kf_i, Cf_i, _ = HydroStrip.apply_BCs(globalKf_i, globalCf_i, globalMf, DOFBlankingList)

            Cf = Cf_r + 1im * Cf_i
            Kf = Kf_r + 1im * Kf_i

            #  Dynamic matrix. also written as Λ_ij in na520 notes
            D = -1 * ω^2 * (Ms + Mf) + im * ω * (Cf + Cs) + (Ks + Kf)

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
            qSol = H * fextω
            ũSol, _ = FEMMethods.put_BC_back(qSol, ELEMTYPE)
            # uSol = real(ũSol)
            uSol = abs.(ũSol) # proper way to do it

            ũout[f_ctr, :] = ũSol

            # ---------------------------
            #   Get hydroloads at freq
            # ---------------------------
            fullAIC = -1 * ω^2 * (globalMf) + im * ω * (globalCf_r + 1im * globalCf_i) + (globalKf_r + 1im * globalKf_i)

            fDynamic, L̃, M̃ = HydroStrip.integrate_hydroLoads(uSol, fullAIC, appendageParams["alfa0"], appendageParams["rake"], DOFBlankingList, SOLVERPARAMS.downwashAngles, ELEMTYPE; appendageOptions=appendageOptions, solverOptions=solverOptions)

            # --- Store total force and tip deflection values ---
            LiftDyn[f_ctr] = L̃
            MomDyn[f_ctr] = M̃

            # phaseAngle = angle(L̃)
            GenXferFcn[f_ctr, :, :] = H


            # # DEBUG QUIT ON FIRST FREQ
            # break
            f_ctr += 1
        end
    end


    # ************************************************
    #     Write solution out to files
    # ************************************************
    write_sol(fSweep, ũout, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, GenXferFcn, outputDir)

    SOL = DCFoilSolution.ForcedVibSolution(LiftDyn, MomDyn, GenXferFcn)

    # # TODO:
    # costFuncs = nothing
    # costFuncs = SolverRoutines.compute_costFuncs()

    return SOL
end

function compute_fextwave(ωRange, FlowCond, appendageParams)

    # TODO: PICKUP HERE
    # --- Wave loads ---
    fAey, mAey = OceanWaves.compute_waveloads(chordLengths, FlowCond.Uinf, FlowCond.rhof, ωe, ωRange, appendageParams["depth0"], stripWidths, claVec)

    extForceVec = zeros(ComplexF64, length(FEMESH.elemConn) * NDOF, length(ωRange))

    return extForceVec
end


function write_sol(fSweep, ũout, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, RAO, outputDir="./OUTPUT/")
    """
    Write out the dynamic results
    """
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

    # --- Write dynamic lift ---
    fname = workingOutput * "totalLiftDyn.jld2"
    save(fname, "data", abs.(LiftDyn))

    # --- Write dynamic moment ---
    fname = workingOutput * "totalMomentDyn.jld2"
    save(fname, "data", abs.(MomDyn))

    fname = workingOutput * "RAO.jld2"
    save(fname, "data", RAO)

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
    #   Get damping
    # ---------------------------
    alphaConst, betaConst = FEMMethods.compute_proportional_damping(real(Ks), real(Ms), appendageParams["zeta"], solverOptions["nModes"])
    Cs = alphaConst * Ms .+ betaConst * Ks

    SOLVERPARAMS = SolutionConstants.DCFoilSolverParams(globalKs, globalMs, Cs, zeros(2, 2), 0.0, 0.0)

    return WING, STRUT, SOLVERPARAMS, FEMESH, LLSystem, LLOutputs, FlowCond
end

# ==============================================================================
#                         Sensitivity routines
# ==============================================================================
function compute_residuals(unfoldedStructuralStates)
    """
    Compute residual for every node that is not the clamped root node

    r(u) = [D(ω)]{ũ} - {f(ω)}

    where f is the force vector from the current solution

    Inputs
    ------
    structuralStates : array
        Unfolded residual state vector with nodal DOFs and deformations EXCLUDING BCs (concatenation of real and imag)

    Outputs
    -------
    resVec : array
        Unfolded residual vector of real data type (concatenation of real and imag parts)
    """

    # --- Fold them ---
    structuralStates = unfoldedStructuralStates[1:end÷2] + 1im * unfoldedStructuralStates[end÷2+1:end]

    if ELEMTYPE == "BT2"
        foilDynamicStates, _ = FEMMethods.put_BC_back(structuralStates, ELEMTYPE)
    else
        println("Invalid element type")
    end

    # --- Stack them ---
    cmplxResVec = CONSTANTS.Dmat * structuralStates - CONSTANTS.extForceVec

    # --- Unfold them ---
    resVec = [real(cmplxResVec); imag(cmplxResVec)]


    return resVec
end

function compute_∂r∂u(structuralStates, mode="FiDi")
    """
    Jacobian of residuals with respect to dynamic structural states
    EXCLUDING BC NODES
    """

    # --- Convert to real data-type ---
    unfolded = vcat(real(structuralStates), imag(structuralStates))

    if mode == "FiDi" # Finite difference
        # First derivative using 3 stencil points
        ∂r∂u = FiniteDifferences.jacobian(central_fdm(3, 1), compute_residuals, structuralStates)

    elseif mode == "RAD" # Forward automatic differentiation
        ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)

        # elseif mode == "RAD" # Reverse automatic differentiation
        #     @time ∂r∂u = ReverseDiff.jacobian(compute_residuals, structuralStates)
    elseif mode == "Analytical"
        ∂r∂u = CONSTANTS.Dmat
    else
        error("Invalid mode")
    end

    return ∂r∂u

end

end # end module