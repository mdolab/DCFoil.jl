# --- Julia 1.9---
"""
@File    :   SolveBodyDynamics.jl
@Time    :   2024/01/28
@Author  :   Galen Ng
@Desc    :   Solver routines for body dynamics of the foiling craft
"""

module SolveBodyDynamics
    
# --- PACKAGES ---
using LinearAlgebra
using FiniteDifferences, Zygote
using Printf

# --- DCFoil modules ---
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE
using ..Vessel
using ..SolveStatic
using ..SolverRoutines
using ..DCFoilSolution

function solve_trim(DVDictList, FEMESHLIST, HULL, solverOptions::Dict, rudderIdx=2)
    """
    This routine solves for the steady-state trim condition of the foiling craft

    Reuse the STATSOL from one solution
    """
    
    # ************************************************
    #     First compute hull parameters
    # ************************************************
    if isnothing(HULL)
        error("You need to define the hull parameters!")
    # else
    #     # Vessel.compute_cm(STATSOL)
    #     pass
    end
    println("====================================================================================")
    println("          BEGINNING BODY DYNAMICS SOLUTION")
    println("====================================================================================")

    # --- Initial control inputs ---
    trim0 = 0.0 #rad
    rudderRake = DVDictList[rudderIdx]["rake"]
    deltaC0 = [trim0, rudderRake]
    # Global vars needed by the residual function
    global gDVDictList = DVDictList
    global gsolverOptions = solverOptions
    global gFEMESHLIST = FEMESHLIST
    global gevalFuncs = ["lift", "moment"]
    global gHULL = HULL


    # ************************************************
    #     Converge r(u) = 0 for vessel
    # ************************************************
    deltaCsol = SolverRoutines.converge_resNonlinear(compute_residuals, compute_∂r∂u, deltaC0)


    return BODYSTATSOL, APPENDAGESTATSOL
end

function compute_fhb(deltaC::Vector, HULL, DVDictList::Vector, FEMESHLIST::Vector, solverOptions::Dict, evalFuncs)
    """
    Compute hydrofoil loads in the body frame about the COG

    Parameters
    ----------
    deltaC : array
        control inputs for the vessel
    DVDictList : array
        dictionary of design variables for each component
    FEMESHLIST : array
        list of FEMESH objects for each component
    solverOptions : dict
        dictionary of solver options
    """

    f_hB = zeros(3)
    m_hB = zeros(3)
    for iComp in eachindex(DVDictList)
        DVDict = DVDictList[iComp]
        FEMESH = FEMESHLIST[iComp]
        appendageOptions = solverOptions["appendageList"][iComp]
        STATSOL = SolveStatic.solve(FEMESH, DVDict, evalFuncs, solverOptions, appendageOptions)
        costFuncs = SolveStatic.evalFuncs(STATSOL, evalFuncs)

        # --- Compute hydrofoil loads in COG frame ---
        liftComp = [0,0,costFuncs["lift"]]
        momentComp = [0, costFuncs["moment"], 0]
        # Distance from aerodynamic center to midchord
        hydroArm = momentComp / liftComp

        f_hB_icomp = liftComp
        
        # Arm about CG
        xCGArm =  HULL.xcg - appendageOptions["xMount"] # these distances go from the bow
        CGArm = [xCGArm, 0, 0]
        println("CGArm: ", CGArm)
        # Moment is formally a cross-product r x F
        m_hB_icomp = cross(CGArm, f_hB_icomp)

        # --- Aggregate ---
        f_hB += f_hB_icomp
        m_hB += m_hB_icomp
    end

    return f_hB, m_hB
end

function compute_residuals(deltaC::Vector)
    """
    Compute sum of the forces and moments on the vessel

    Parameters
    ----------
    deltaC : array
        control inputs for the vessel
    """
    
    f_hB, m_hB = compute_fhb(deltaC, gHULL, gDVDictList, gFEMESHLIST, gsolverOptions, gevalFuncs)

    f_gB = Vessel.compute_gravloads(gHULL, gsolverOptions)

    # --- Sum of forces and moments ---
    bforces = f_hB + f_gB
    bmoments = m_hB
    # r(u) = [Fx, Fy, Fz, Mx, My, Mz]^T
    resVec = vcat(bforces, bmoments)


    return resVec
end

function compute_∂r∂u(deltaC, mode="FiDi")
    """
    Jacobian of residuals with respect to control inputs
    """

    if mode == "FiDi"
        ∂r∂u = FiniteDifferences.jacobian(compute_residuals, deltaC)
    elseif mode == "RAD"
        # This is a tuple
        ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)
    elseif mode == "analytic"
        error("Not implemented yet")
    else
        error("Invalid mode")
    end

    return ∂r∂u

end

end

