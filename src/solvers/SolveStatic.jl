# --- Julia ---

# @File    :   SolveStatic.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Solve the beam equations w/ steady aero/hydrodynamics and compute gradients

module SolveStatic
"""
Static hydroelastic solver
"""

# --- Public functions ---
export solve, do_newton_rhapson

# --- Libraries ---
using FLOWMath: linear, akima
using FiniteDifferences
using LinearAlgebra, Statistics
using JSON
# using ForwardDiff
# using ReverseDiff
using Zygote

# --- DCFoil modules ---
# First include them
include("../InitModel.jl")
include("../struct/BeamProperties.jl")
include("../struct/FiniteElements.jl")
include("../hydro/Hydro.jl")
include("../constants/SolutionConstants.jl")
include("./SolverRoutines.jl")
# then use them
using .InitModel, .Hydro, .StructProp
using .FEMMethods
using .SolutionConstants
using .SolverRoutines

function solve(DVDict, evalFuncs, outputDir::String)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)

    """
    # ************************************************
    #     INITIALIZE
    # ************************************************
    # --- Write the init dict to output folder ---
    stringData = JSON.json(DVDict)
    open(outputDir * "init_DVDict.json", "w") do io
        write(io, stringData)
    end

    neval = DVDict["neval"]
    global FOIL = InitModel.init_steady(neval, DVDict) # seems to only be global in this module
    nElem = neval - 1
    constitutive = FOIL.constitutive

    println("====================================================================================")
    println("          BEGINNING STATIC HYDROELASTIC SOLUTION              ")
    println("====================================================================================")

    # ************************************************
    #     SOLVE FEM FIRST TIME
    # ************************************************
    # elemType = "bend"
    # elemType = "bend-twist"
    elemType = "BT2"
    loadType = "force"

    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)

    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, constitutive)
    FEMMethods.apply_tip_load!(globalF, elemType, loadType)

    # --- Initialize states ---
    u = copy(globalF)

    # ---------------------------
    #   Get initial fluid tracts
    # ---------------------------
    fTractions, AIC, planformArea = Hydro.compute_static_hydroLoads(u, structMesh, FOIL, elemType)
    globalF = fTractions

    # # --- Debug printout of matrices in human readable form ---
    # println("Global stiffness matrix:")
    # println("------------------------")
    # show(stdout, "text/plain", globalK)
    # println("")
    # # println("Global mass matrix:")
    # # println("-------------------")
    # # show(stdout, "text/plain", globalM)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    globalDOFBlankingList = FEMMethods.get_fixed_nodes(elemType, "clamped")
    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    # # --- Debug printout of matrices in human readable form after BC application ---
    # println("Global stiffness matrix:")
    # println("------------------------")
    # show(stdout, "text/plain", K)
    # println("")
    # println("Global mass matrix:")
    # println("-------------------")
    # show(stdout, "text/plain", M)


    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(K, M, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ 1:length(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q


    # ************************************************
    #     CONVERGE r(u) = 0
    # ************************************************
    # --- Assign constants accessible in this module ---
    # This is needed for derivatives!
    derivMode = "FAD"
    global CONSTANTS = SolutionConstants.DCFoilConstants(K, elemType, structMesh, AIC, derivMode, planformArea)

    qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q)
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, CONSTANTS.elemType)

    # --- Get hydroLoads again on solution ---
    fHydro, AIC, _ = compute_hydroLoads(uSol, structMesh, elemType)

    # ************************************************
    #     COMPUTE FUNCTIONS OF INTEREST
    # ************************************************
    costFuncs = compute_cost_func(uSol, fHydro, evalFuncs)

    # ************************************************
    #     COMPUTE SENSITIVITIES
    # ************************************************
    mode = "FAD"
    ∂r∂u = compute_∂r∂u(qSol, mode)
    # # TODO:I'm not really sure how to do these yet
    # ∂r∂x = compute_∂r∂x(qSol, mode)
    # ∂f∂u = compute_∂f∂u(qSol, mode)
    # ∂f∂x = compute_∂f∂x(qSol, mode)

    # ************************************************
    #     WRITE SOLUTION OUT TO FILES
    # ************************************************
    write_sol(uSol, fHydro, costFuncs, elemType, outputDir)

end

function return_totalStates(foilStructuralStates, FOIL, elemType="BT2")
    """
    Returns the deflected + rigid shape of the foil
    """

    alfaRad = FOIL.α₀ * π / 180

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
        staticOffset = [0, 0, alfaRad]
    elseif elemType == "BT2"
        nDOF = 4
        staticOffset = [0, 0, alfaRad, 0] #TODO: pretwist will change this
    end

    # Add static angle of attack to deflected foil
    w = foilStructuralStates[1:nDOF:end]
    foilTotalStates = copy(foilStructuralStates) + repeat(staticOffset, outer=[length(w)])

    return foilTotalStates, nDOF
end

function compute_hydroLoads(foilStructuralStates, mesh, elemType="bend-twist")
    """
    Computes the steady hydrodynamic vector loads 
    given the solved hydrofoil shape (strip theory)
    """
    # ---------------------------
    #   Initializations
    # ---------------------------
    foilTotalStates, nDOF = return_totalStates(foilStructuralStates, FOIL, elemType)
    nGDOF = FOIL.neval * nDOF

    # ---------------------------
    #   Strip theory
    # ---------------------------
    AIC = zeros(nGDOF, nGDOF)
    _, planformArea = Hydro.compute_static_AICs!(AIC, mesh, FOIL, elemType)

    # --- Compute fluid tractions ---
    fTractions = -1 * AIC * foilTotalStates # aerodynamic forces are on the RHS so we negate

    # # --- Debug printout ---
    # println("AIC")
    # show(stdout, "text/plain", AIC)
    # println("")
    # println("Aero loads")
    # println(fTractions)

    return fTractions, AIC, planformArea
end

function compute_cost_func(states, forces, evalFuncs)
    """
    Given {u} and the forces, compute the cost functions
    """

    if CONSTANTS.elemType == "BT2"
        nDOF = 4
        Ψ = states[3:nDOF:end]
        Moments = forces[3:nDOF:end]
        W = states[1:nDOF:end]
        Lift = forces[1:nDOF:end]
    else
        println("Invalid element type")
    end

    # ************************************************
    #     COMPUTE COST FUNCS
    # ************************************************
    costFuncs = Dict() # initialize empty costFunc dictionary
    if "lift" in evalFuncs
        TotalLift = sum(Lift) * FOIL.s / FOIL.neval
        costFuncs["lift"] = TotalLift
    end
    if "moment" in evalFuncs
        TotalMoment = sum(Moments) * FOIL.s / FOIL.neval
        costFuncs["moment"] = TotalMoment
    end
    if "cl" in evalFuncs
        CL = TotalLift / (0.5 * FOIL.ρ_f * FOIL.U∞^2 * CONSTANTS.planformArea)
        costFuncs["cl"] = CL
    end
    if "cmy" in evalFuncs
        CM = TotalMoment / (0.5 * FOIL.ρ_f * FOIL.U∞^2 * CONSTANTS.planformArea * mean(FOIL.c))
        costFuncs["cmy"] = CM
    end

    return costFuncs
end

function write_sol(states, forces, funcs, elemType="bend", outputDir="./OUTPUT/")
    """
    Inputs
    ------
    states: vector of structural states from the [K]{u} = {f}
    """

    mkpath(outputDir)

    # --- First print costFuncs to screen in a box ---
    println("+", "-"^50, "+")
    println("|                costFunc dictionary:              |")
    println("+", "-"^50, "+")
    for kv in funcs
        println("| ", kv)
    end

    fname = outputDir * "funcs.json"
    stringData = JSON.json(funcs)
    open(fname, "w") do io
        write(io, stringData)
    end

    if elemType == "bend"
        nDOF = 2
    elseif elemType == "bend-twist"
        nDOF = 3
        Ψ = states[nDOF:nDOF:end]
        Moments = forces[nDOF:nDOF:end]
    elseif elemType == "BT2"
        nDOF = 4
        Ψ = states[3:nDOF:end]
        Moments = forces[3:nDOF:end]
    else
        error("Invalid element type")
    end

    W = states[1:nDOF:end]
    Lift = forces[1:nDOF:end]

    # --- Write bending ---
    fname = outputDir * "bending.dat"
    outfile = open(fname, "w")
    # write(outfile, "Bending\n")
    for wⁿ ∈ W
        write(outfile, string(wⁿ, "\n"))
    end
    close(outfile)

    # --- Write twist ---
    if @isdefined(Ψ)
        fname = outputDir * "twisting.dat"
        outfile = open(fname, "w")
        # write(outfile, "Twist\n")
        for Ψⁿ ∈ Ψ
            write(outfile, string(Ψⁿ, "\n"))
        end
        close(outfile)
    end

    # --- Write lift and moments ---
    fname = outputDir * "lift.dat"
    outfile = open(fname, "w")
    for Fⁿ in Lift
        write(outfile, string(Fⁿ, "\n"))
    end
    close(outfile)
    fname = outputDir * "moments.dat"
    outfile = open(fname, "w")
    for Mⁿ in Moments
        write(outfile, string(Mⁿ, "\n"))
    end
    close(outfile)


end

# ==============================================================================
#                         Sensitivity routines
# ==============================================================================
function compute_∂f∂x(foilPDESol)

end

function compute_∂r∂x(foilPDESol)

end

function compute_∂f∂u(foilPDESol)

end

function compute_∂r∂u(structuralStates, mode="FiDi")
    """
    Jacobian of residuals with respect to structural states 
    EXCLUDING BC NODES
    """

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

function compute_residuals(structuralStates)
    """
    Compute residual for every node that is not the clamped root node

    r(u) = [K]{u} - {f(u)}

    where f(u) is the force vector from the current solution

    Inputs
    ------
    structuralStates : array
        State vector with nodal DOFs and deformations EXCLUDING BCs
    """

    # NOTE THAT WE ONLY DO THIS CALL HERE
    if CONSTANTS.elemType == "bend-twist" # knock off the root element
        exit()
    elseif CONSTANTS.elemType == "BT2" # knock off root element
        completeStates, _ = FEMMethods.put_BC_back(structuralStates, CONSTANTS.elemType)
        foilTotalStates, nDOF = return_totalStates(completeStates, FOIL, CONSTANTS.elemType)
        F = -CONSTANTS.AICmat * foilTotalStates
        FOut = F[5:end]
    else
        println(CONSTANTS.elemType)
        println("Invalid element type")
    end


    # --- Stack them ---
    resVec = CONSTANTS.Kmat * structuralStates - FOut

    return resVec
end

function compute_direct()
    """
    Computes direct vector 
    """
end

function compute_adjoint()

end

function compute_jacobian(stateVec)
    """
    Compute the jacobian df/dx

    Inputs:
        stateVec: 

    returns:
        

    """
    # ************************************************
    #     Compute cost func gradients
    # ************************************************

    # TODO:

end


end # end module