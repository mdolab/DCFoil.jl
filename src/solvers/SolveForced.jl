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

# --- Libraries ---
using FLOWMath: linear, akima
using LinearAlgebra, Statistics
using JSON
using Zygote
using JLD

# --- DCFoil modules ---
# First include them
include("../InitModel.jl")
include("../struct/BeamProperties.jl")
include("../struct/FiniteElements.jl")
include("../hydro/Hydro.jl")
include("SolveStatic.jl")
include("../constants/SolutionConstants.jl")
include("./SolverRoutines.jl")
# then use them
using .InitModel, .Hydro, .StructProp
using .FEMMethods
using .SolveStatic
using .SolutionConstants
using .SolverRoutines

function solve(DVDict, solverOptions::Dict)
    """
    Solve
        (-ω²[M]-jω[C]+[K]){ũ} = {f̃}
    using the 'tipForceMag' as the harmonic forcing on the RHS
    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    outputDir = solverOptions["outputDir"]
    fSweep = solverOptions["fSweep"]
    tipForceMag = solverOptions["tipForceMag"]
    global FOIL = InitModel.init_dynamic(DVDict; fSweep=fSweep)
    nElem = FOIL.neval - 1
    constitutive = FOIL.constitutive

    println("====================================================================================")
    println("        BEGINNING HARMONIC FORCED HYDROELASTIC SOLUTION")
    println("====================================================================================")

    # ************************************************
    #     Assemble structural matrices
    # ************************************************
    elemType = "BT2"
    loadType = "force"

    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)
    globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, constitutive)
    FEMMethods.apply_tip_load!(globalF, elemType, loadType)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    globalDOFBlankingList = FEMMethods.get_fixed_nodes(elemType, "clamped")
    Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, globalDOFBlankingList)

    # --- Initialize stuff ---
    u = copy(globalF)
    globalMf = copy(globalMs) * 0
    globalCf_r = copy(globalKs) * 0
    globalKf_r = copy(globalKs) * 0
    globalKf_i = copy(globalKs) * 0
    globalCf_i = copy(globalKs) * 0
    extForceVec = copy(F) * 0 # this is a vector excluded the BC nodes
    extForceVec[end-1] = tipForceMag # this is applying a tip twist
    LiftDyn = zeros(length(fSweep)) # * 0im
    MomDyn = zeros(length(fSweep)) # * 0im
    TipBendDyn = zeros(length(fSweep)) # * 0im
    TipTwistDyn = zeros(length(fSweep)) # * 0im

    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(Ks, Ms, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ 1:length(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q

    # ************************************************
    #     For every frequency, solve the system
    # ************************************************
    f_ctr = 1
    for f in fSweep
        println("Solving for frequency: ", f, "Hz")
        ω = 2π * f # circular frequency
        # ---------------------------
        #   Assemble hydro matrices
        # ---------------------------
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = Hydro.compute_AICs!(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, FOIL, FOIL.U∞, ω, elemType)
        Kf_r, Cf_r, Mf = Hydro.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
        Kf_i, Cf_i, _ = Hydro.apply_BCs(globalKf_i, globalCf_i, globalMf, globalDOFBlankingList)

        # TODO: I split up the stuff before to separate imag and real math
        Cf = Cf_r + 1im * Cf_i
        Kf = Kf_r + 1im * Kf_i

        #  Dynamic matrix
        D = -1 * ω^2 * (Ms + Mf) + im * ω * Cf + (Ks + Kf)

        # Complex AIC
        AIC = -1 * ω^2 * (Mf) + im * ω * Cf + (Kf)

        # Store constants
        global CONSTANTS = SolutionConstants.DCFoilDynamicConstants(elemType, structMesh, D, AIC, extForceVec)
        global DFOIL = FOIL

        # ---------------------------
        #   Solve for dynamic states
        # ---------------------------
        # qSol, _ = converge_r(q)
        qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q, is_cmplx=true, is_verbose=false)
        uSol, _ = FEMMethods.put_BC_back(qSol, CONSTANTS.elemType)

        # ---------------------------
        #   Get hydroloads at freq
        # ---------------------------
        fullAIC = -1 * ω^2 * (globalMf) + im * ω * (globalCf_r + 1im * globalCf_i) + (globalKf_r + 1im * globalKf_i)
        fDynamic, DynLift, DynMoment = Hydro.integrate_hydroLoads(uSol, fullAIC, DFOIL, CONSTANTS.elemType)#compute_hydroLoads(uSol, fullAIC)

        # --- Store total force and tip deflection values ---
        global LiftDyn[f_ctr] = abs(DynLift[end])
        global MomDyn[f_ctr] = abs(DynMoment[end])
        if elemType == "BT2"
            global TipBendDyn[f_ctr] = abs(uSol[end-3])
            global TipTwistDyn[f_ctr] = abs(uSol[end-1])
            phaseAngle = angle(uSol[end-3])
        end

        # # DEBUG QUIT ON FIRST FREQ
        # break
        f_ctr += 1
    end


    # ************************************************
    #     Write solution out to files
    # ************************************************
    write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, outputDir)

    # TODO:
    costFuncs = nothing
    # costFuncs = SolverRoutines.compute_costFuncs()

    return costFuncs
end

function write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, outputDir="./OUTPUT/")
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
    fname = workingOutput * "tipBendDyn.jld"
    save(fname, "data", TipBendDyn)

    # --- Write tip twist ---
    fname = workingOutput * "tipTwistDyn.jld"
    save(fname, "data", TipTwistDyn)

    # --- Write dynamic lift ---
    fname = workingOutput * "totalLiftDyn.jld"
    save(fname, "data", LiftDyn)

    # --- Write dynamic moment ---
    fname = workingOutput * "totalMomentDyn.jld"
    save(fname, "data", MomDyn)

end

# # ==============================================================================
# #                         Solver routines
# # ==============================================================================
# function do_newton_rhapson_cmplx(u, maxIters=200, tol=1e-12, verbose=true, mode="FAD")
#     """
#     Simple complex data type Newton-Rhapson solver

#     Inputs
#     ------
#     u : complex vector
#         Initial guess
#     Outputs
#     -------
#     converged_u : complex vector
#         Converged solution
#     converged_r : complex vector
#         Converged residual
#     """

#     uUnfolded = [real(u); imag(u)]
#     for ii in 1:maxIters
#         # println(u)
#         # NOTE: these functions handle a complex input but return the unfolded output
#         # (i.e., concatenation of real and imag)
#         res = compute_residuals(uUnfolded)
#         ∂r∂u = compute_∂r∂u(uUnfolded, mode)
#         jac = ∂r∂u[1]

#         # --- Newton step ---
#         # show(stdout, "text/plain", jac)
#         Δu = -jac \ res

#         # --- Update ---
#         uUnfolded = uUnfolded + Δu

#         resNorm = norm(res, 2)

#         # --- Printout ---
#         if verbose
#             if ii == 1
#                 println("resNorm | stepNorm ")
#             end
#             println(resNorm, "|", norm(Δu, 2))
#         end

#         # --- Check norm ---
#         # Note to self, the for and while loop in Julia introduce a new scope...this is pretty stupid
#         if resNorm < tol
#             println("Converged in ", ii, " iterations")
#             global converged_u = uUnfolded[1:end÷2] + 1im * uUnfolded[end÷2+1:end]
#             global converged_r = res[1:end÷2] + 1im * res[end÷2+1:end]
#             break
#         elseif ii == maxIters
#             println("Failed to converge. res norm is", resNorm)
#             println("DID THE FOIL STATICALLY DIVERGE? CHECK DEFLECTIONS IN POST PROC")
#             global converged_u = uUnfolded[1:end÷2] + 1im * uUnfolded[end÷2+1:end]
#             global converged_r = res[1:end÷2] + 1im * res[end÷2+1:end]
#         else
#             global converged_u = uUnfolded[1:end÷2] + 1im * uUnfolded[end÷2+1:end]
#             global converged_r = res[1:end÷2] + 1im * res[end÷2+1:end]
#         end
#     end

#     # println("Size of state vector")
#     # print(size(converged_u))

#     return converged_u, converged_r
# end

# function converge_r(u0, maxIters=200, tol=1e-6, verbose=true, mode="FAD")
#     """
#     Given initial input u0, solve system r(u) = 0 with complex Newton
#     """
#     # ************************************************
#     #     Main solver loop
#     # ************************************************
#     converged_u, converged_r = do_newton_rhapson_cmplx(u0, maxIters, tol, verbose, mode)

#     return converged_u, converged_r
# end

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

    if CONSTANTS.elemType == "BT2"
        foilDynamicStates, _ = FEMMethods.put_BC_back(structuralStates, CONSTANTS.elemType)
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

    elseif mode == "FAD" # Forward automatic differentiation
        ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)

        # elseif mode == "RAD" # Reverse automatic differentiation
        #     @time ∂r∂u = ReverseDiff.jacobian(compute_residuals, structuralStates)
    else
        error("Invalid mode")
    end

    return ∂r∂u

end

end # end module