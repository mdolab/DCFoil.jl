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
using LinearAlgebra, Statistics
using JSON
using Zygote
using FileIO

# --- DCFoil modules ---
# First include them
include("../InitModel.jl")
include("../struct/BeamProperties.jl")
include("../struct/FiniteElements.jl")
include("../hydro/HydroStrip.jl")
include("SolveStatic.jl")
include("../constants/SolutionConstants.jl")
include("./SolverRoutines.jl")
# then use them
using .InitModel, .HydroStrip, .StructProp
using .FEMMethods
using .SolveStatic
using .SolutionConstants
using .SolverRoutines

function solve(structMesh, elemConn, DVDict, solverOptions::Dict)
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
    global FOIL = InitModel.init_model_wrapper(DVDict, solverOptions; fSweep=fSweep)

    println("====================================================================================")
    println("        BEGINNING HARMONIC FORCED HYDROELASTIC SOLUTION")
    println("====================================================================================")

    # ************************************************
    #     Assemble structural matrices
    # ************************************************
    elemType = "COMP2"
    loadType = "force"

    abVec = DVDict["ab"]
    x_αbVec = DVDict["x_αb"]
    chordVec = DVDict["c"]
    ebVec = 0.25 * chordVec .+ abVec
    Λ = DVDict["Λ"]
    U∞ = solverOptions["U∞"]
    α₀ = DVDict["α₀"]
    globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, abVec, x_αbVec, FOIL, elemType, FOIL.constitutive)
    # Get transformation matrix for the tip load
    angleDefault = deg2rad(90) # default angle of rotation from beam local about z
    axisDefault = "z"
    T1 = FEMMethods.get_rotate3dMat(angleDefault, axis=axisDefault)
    T = T1
    transMatL2G = [
        T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
        zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
        zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3)
        zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3)
        zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3)
        zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T
        ]
    FEMMethods.apply_tip_load!(globalF, elemType, transMatL2G, loadType)

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
    extForceVec[end-3] = tipForceMag # this is applying a tip lift
    LiftDyn = zeros(length(fSweep)) # * 0im
    MomDyn = zeros(length(fSweep)) # * 0im
    TipBendDyn = zeros(length(fSweep)) # * 0im
    TipTwistDyn = zeros(length(fSweep)) # * 0im
    RAO = zeros(ComplexF64, length(fSweep), length(F), length(F))

    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(Ks, Ms, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ eachindex(u) if x ∉ globalDOFBlankingList] # list comprehension
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
        # globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(globalMf_0, globalCf_r_0, globalCf_i_0, globalKf_r_0, globalKf_i_0, structMesh, FOIL, FOIL.U∞, ω, elemType)
        # globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType)
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(size(globalMs)[1], structMesh, Λ, chordVec, abVec, ebVec, FOIL, U∞, ω, elemType)
        Kf_r, Cf_r, Mf = HydroStrip.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
        Kf_i, Cf_i, _ = HydroStrip.apply_BCs(globalKf_i, globalCf_i, globalMf, globalDOFBlankingList)

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
        # The below way is the numerical way to do it but might skip if this doesntwork
        # qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q, is_cmplx=true, is_verbose=false)
        H = inv(D) # RAO
        qSol = real(H * extForceVec)
        uSol, _ = FEMMethods.put_BC_back(qSol, CONSTANTS.elemType)

        # ---------------------------
        #   Get hydroloads at freq
        # ---------------------------
        fullAIC = -1 * ω^2 * (globalMf) + im * ω * (globalCf_r + 1im * globalCf_i) + (globalKf_r + 1im * globalKf_i)
        fDynamic, DynLift, DynMoment = HydroStrip.integrate_hydroLoads(uSol, fullAIC, α₀, CONSTANTS.elemType)#compute_hydroLoads(uSol, fullAIC)

        # --- Store total force and tip deflection values ---
        LiftDyn[f_ctr] = (DynLift)
        MomDyn[f_ctr] = (DynMoment)
        RAO[f_ctr, :, :] = H
        if elemType == "BT2"
            TipBendDyn[f_ctr] = (uSol[end-3])
            TipTwistDyn[f_ctr] = (uSol[end-1])
            phaseAngle = angle(uSol[end-3])
        end

        # # DEBUG QUIT ON FIRST FREQ
        # break
        f_ctr += 1
    end


    # ************************************************
    #     Write solution out to files
    # ************************************************
    write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, RAO, outputDir)

    # TODO:
    costFuncs = nothing
    # costFuncs = SolverRoutines.compute_costFuncs()

    return costFuncs
end

function write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, RAO, outputDir="./OUTPUT/")
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
    save(fname, "data", TipBendDyn)

    # --- Write tip twist ---
    fname = workingOutput * "tipTwistDyn.jld2"
    save(fname, "data", TipTwistDyn)

    # --- Write dynamic lift ---
    fname = workingOutput * "totalLiftDyn.jld2"
    save(fname, "data", LiftDyn)

    # --- Write dynamic moment ---
    fname = workingOutput * "totalMomentDyn.jld2"
    save(fname, "data", MomDyn)

    fname = workingOutput * "RAO.jld2"
    save(fname, "data", RAO)

end

# # ==============================================================================
# #                         Solver routines
# # ==============================================================================
# function do_newton_rhapson_cmplx(u, maxIters=200, tol=1e-12, verbose=true, mode="RAD")
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

# function converge_r(u0, maxIters=200, tol=1e-6, verbose=true, mode="RAD")
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