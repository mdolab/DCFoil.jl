# --- Julia---
"""
@File    :   SolveDynamic.jl
@Time    :   2022/10/07
@Author  :   Galen Ng
@Desc    :   Similar to SolveSteady.jl but now it is a second order dynamical system!
"""

module SolveDynamic
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

# --- DCFoil modules ---
# First include them
include("InitModel.jl")
include("Struct.jl")
include("struct/FiniteElements.jl")
include("Hydro.jl")
include("SolveSteady.jl")
# then use them
using .InitModel, .Hydro, .StructProp
using .FEMMethods
using .SolveSteady

function solve(DVDict, outputDir::String, fSweep, tipForceMag)
    """
    Solve (-ω²[M]-jω[C]+[K]){ũ} = {f̃}
    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    global FOIL = InitModel.init_dynamic(fSweep, DVDict)
    nElem = FOIL.neval - 1
    constitutive = FOIL.constitutive

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
    extForceVec[end] = tipForceMag
    LiftDyn = zeros(length(fSweep))
    MomDyn = zeros(length(fSweep))
    TipBendDyn = zeros(length(fSweep))
    TipTwistDyn = zeros(length(fSweep))

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

        #   Dynamic matrix
        D = -1 * ω^2 * (Ms + Mf) + im * ω * Cf + (Ks + Kf)

        # Complex AIC 
        AIC = -1 * ω^2 * (Mf) + im * ω * Cf + (Kf)

        # Store constants
        global CONSTANTS = InitModel.DCFoilDynamicConstants(elemType, structMesh, D, AIC, extForceVec)

        # ---------------------------
        #   Solve for dynamic states
        # ---------------------------
        qSol, _ = converge_r(q)
        if CONSTANTS.elemType == "BT2"
            uSol = vcat([0, 0, 0, 0], qSol)
        end

        # ---------------------------
        #   Get hydroloads at freq
        # ---------------------------
        fDynamic, DynLift, DynMoment = compute_hydroLoads(uSol)

        # --- Store tip values ---
        LiftDyn[f_ctr] = DynLift[end]
        MomDyn[f_ctr] = DynMoment[end]
        if elemType == "BT2"
            TipBendDyn[f_ctr] = uSol[end-3]
            TipTwistDyn[f_ctr] = uSol[end-1]
        end

        # DEBUG QUIT ON FIRST FREQ
        exit()

    end


    # ************************************************
    #     Write solution out to files
    # ************************************************
    write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, outputDir)

end

function compute_hydroLoads(foilDynamicStructuralStates)
    """
    Compute the lift and moment vectors (and totals) for the fluctating loads
        f_hydro,dyn
    """
    # --- Initializations ---
    # This is dynamic deflection + rigid shape of foil
    foilTotalDynStates, nDOF = SolveSteady.return_totalStates(foilDynamicStructuralStates, CONSTANTS.elemType)
    nGDOF = FOIL.neval * nDOF

    # --- Strip theory ---
    fDynamic = CONSTANTS.AICmat * foilTotalDynStates

    if elemType == "bend-twist"
        Moments = fDynamic[nDOF:nDOF:end]
    elseif elemType == "BT2"
        Moments = fDynamic[3:nDOF:end]
    else
        error("Invalid element type")
    end
    Lift = fDynamic[1:nDOF:end]

    # --- Total dynamic hydro force calcs ---
    TotalLift = sum(Lift) * FOIL.s / FOIL.neval
    TotalMoment = sum(Moments) * FOIL.s / FOIL.neval

    return fDynamic, TotalLift, TotalMoment
end

function write_sol(fSweep, TipBendDyn, TipTwistDyn, LiftDyn, MomDyn, outputDir="./OUTPUT/")
    """
    Write out the dynamic results
    """
    mkpath(outputDir)

    # --- Write frequency sweep ---
    fname = outputDir * "fSweep.dat"
    outfile = open(fname, "w")
    for f ∈ fSweep
        write(outfile, string(f) * "\n")
    end

    # --- Write tip bending ---
    fname = outputDir * "TipBendDyn.dat"
    outfile = open(fname, "w")
    for h ∈ TipBendDyn
        write(outfile, string(h) * "\n")
    end
    close(outfile)

    # --- Write tip twist ---
    fname = outputDir * "TipTwistDyn.dat"
    outfile = open(fname, "w")
    for ψ ∈ TipTwistDyn
        write(outfile, string(ψ) * "\n")
    end
    close(outfile)

    # --- Write dynamic lift ---
    fname = outputDir * "LiftDyn.dat"
    outfile = open(fname, "w")
    for L ∈ LiftDyn
        write(outfile, string(L) * "\n")
    end

    # --- Write dynamic moment ---
    fname = outputDir * "MomentDyn.dat"
    outfile = open(fname, "w")
    for M ∈ MomDyn
        write(outfile, string(M) * "\n")
    end

end
# ==============================================================================
#                         Solver routines
# ==============================================================================
function do_newton_rhapson_cmplx(u, maxIters=200, tol=1e-12, verbose=true, mode="FAD")
    """
    Simple Newton-Rhapson solver
    """
    # TODO:
    for ii in 1:maxIters
        # println(u)
        res = compute_residuals(u)
        ∂r∂u = compute_∂r∂u(u, mode)
        jac = ∂r∂u[1]

        # --- Newton step ---
        # show(stdout, "text/plain", jac)
        Δu = -jac \ res

        # --- Update ---
        u = u + Δu

        resNorm = norm(res, 2)

        # --- Printout ---
        if verbose
            if ii == 1
                println("resNorm | stepNorm ")
            end
            println(resNorm, "|", norm(Δu, 2))
        end

        # --- Check norm ---
        # Note to self, the for and while loop in Julia introduce a new scope...this is pretty stupid
        if resNorm < tol
            println("Converged in ", ii, " iterations")
            global converged_u = copy(u)
            global converged_r = copy(res)
            break
        elseif ii == maxIters
            println("Failed to converge. res norm is", resNorm)
            println("DID THE FOIL STATICALLY DIVERGE? CHECK DEFLECTIONS IN POST PROC")
            global converged_u = copy(u)
            global converged_r = copy(res)
        else
            global converged_u = copy(u)
            global converged_r = copy(res)
        end
    end

    # println("Size of state vector")
    # print(size(converged_u))

    return converged_u, converged_r
end

function converge_r(u0, maxIters=200, tol=1e-6, verbose=true, mode="FAD")
    """
    Given initial input u0, solve system r(u) = 0 with complex Newton
    """
    # ************************************************
    #     Main solver loop
    # ************************************************
    converged_u, converged_r = do_newton_rhapson_cmplx(u0, maxIters, tol, verbose, mode)

    return converged_u, converged_r
end

# ==============================================================================
#                         Sensitivity routines
# ==============================================================================
function compute_residuals(structuralStates)
    """
    Compute residual for every node that is not the clamped root node

    r(u) = [D(ω)]{ũ} - {f(ω)}

    where f is the force vector from the current solution

    Inputs
    ------
    structuralStates : array
        State vector with nodal DOFs and deformations EXCLUDING BCs
    """
    if CONSTANTS.elemType == "BT2"

        foilDynamicStates = vcat([0, 0, 0, 0], structuralStates)
    else
        println("Invalid element type")
    end

    # --- Stack them ---
    resVec = CONSTANTS.Dmat * structuralStates - CONSTANTS.extForceVec

    return resVec
end

function compute_∂r∂u(structuralStates, mode="FiDi")
    """
    Jacobian of residuals with respect to dynamic structural states 
    EXCLUDING BC NODES
    """
    
end

end # end module