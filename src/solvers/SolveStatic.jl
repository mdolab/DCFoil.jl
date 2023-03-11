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

# ==============================================================================
#                         Top level API routines
# ==============================================================================
function solve(structMesh, elemConn, DVDict::Dict, evalFuncs, solverOptions::Dict)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)
    Inputs
    ------
    DVDict: Dict
        The dictionary of design variables
    evalFuncs: Dict
        The dictionary of functions to evaluate
    solverOptions: Dict
        The dictionary of solver options
    Returns
    -------
    costFuncs: Dict
        The dictionary of cost functions
    """
    # ************************************************
    #     INITIALIZE
    # ************************************************
    outputDir = solverOptions["outputDir"]
    nNodes = DVDict["nNodes"]
    global FOIL = InitModel.init_static(nNodes, DVDict) # seems to only be global in this module

    println("====================================================================================")
    println("          BEGINNING STATIC HYDROELASTIC SOLUTION")
    println("====================================================================================")

    # ************************************************
    #     SOLVE FEM FIRST TIME
    # ************************************************
    # elemType = "bend"
    # elemType = "bend-twist"
    elemType = "BT2"
    loadType = "force"

    globalK, globalM, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, FOIL.constitutive)
    FEMMethods.apply_tip_load!(globalF, elemType, loadType)
    # if solverOptions["tipMass"]
    #     bulbMass = 2200 #[kg]
    #     bulbInertia = 900 #[kg-m²]
    #     FOIL.x_αb[end] = -0.1 # [m]
    #     FEMMethods.apply_tip_mass!(globalMs, bulbMass, bulbInertia, structMesh[2] - structMesh[1], FOIL, elemType)
    # end

    # --- Initialize states ---
    u = copy(globalF)

    # ---------------------------
    #   Get initial fluid tracts
    # ---------------------------
    fTractions, AIC, planformArea = Hydro.compute_steady_hydroLoads(u, structMesh, FOIL, elemType)
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
    idxNotBlanked = [x for x ∈ eachindex(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q


    # ************************************************
    #     CONVERGE r(u) = 0
    # ************************************************
    # --- Assign constants accessible in this module ---
    # This is needed for derivatives!
    derivMode = "FAD"
    global CONSTANTS = SolutionConstants.DCFoilConstants(K, zeros(2, 2), elemType, structMesh, AIC, derivMode, planformArea)

    qSol, _ = SolverRoutines.converge_r(compute_residuals, compute_∂r∂u, q)
    # qSol = q # just use pre-solve solution
    uSol, _ = FEMMethods.put_BC_back(qSol, CONSTANTS.elemType)

    # --- Get hydroLoads again on solution ---
    fHydro, AIC, _ = Hydro.compute_steady_hydroLoads(uSol, structMesh, FOIL, elemType)

    # ************************************************
    #     COMPUTE FUNCTIONS OF INTEREST
    # ************************************************
    costFuncs = evalFuncs(uSol, fHydro, evalFuncs)

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
    # Also a quick static divergence check
    if costFuncs["psitip"] * DVDict["θ"] > 0.0
        println("+---------------------------------------------------+")
        println("|  WARNING: STATIC DIVERGENCE CONDITION DETECTED!   |")
        println("|  PRODUCT OF FIBER ANGLE AND TIP TWIST ARE +VE     |")
        println("+---------------------------------------------------+")
    end
    write_sol(uSol, fHydro, costFuncs, elemType, outputDir)

    return costFuncs
end


function write_sol(states, forces, funcs, elemType="bend", outputDir="./OUTPUT/")
    """
    Inputs
    ------
    states: vector of structural states from the [K]{u} = {f}
    """

    # --- Make output directory ---
    workingOutputDir = outputDir * "static/"
    mkpath(workingOutputDir)

    # --- First print costFuncs to screen in a box ---
    println("+", "-"^50, "+")
    println("|                costFunc dictionary:              |")
    println("+", "-"^50, "+")
    for kv in funcs
        println("| ", kv)
    end

    fname = workingOutputDir * "funcs.json"
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
    fname = workingOutputDir * "bending.dat"
    outfile = open(fname, "w")
    # write(outfile, "Bending\n")
    for wⁿ ∈ W
        write(outfile, string(wⁿ, "\n"))
    end
    close(outfile)

    # --- Write twist ---
    if @isdefined(Ψ)
        fname = workingOutputDir * "twisting.dat"
        outfile = open(fname, "w")
        # write(outfile, "Twist\n")
        for Ψⁿ ∈ Ψ
            write(outfile, string(Ψⁿ, "\n"))
        end
        close(outfile)
    end

    # --- Write lift and moments ---
    fname = workingOutputDir * "lift.dat"
    outfile = open(fname, "w")
    for Fⁿ in Lift
        write(outfile, string(Fⁿ, "\n"))
    end
    close(outfile)
    fname = workingOutputDir * "moments.dat"
    outfile = open(fname, "w")
    for Mⁿ in Moments
        write(outfile, string(Mⁿ, "\n"))
    end
    close(outfile)


end

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function evalFuncs(states, forces, evalFuncs)
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
    if "wtip" in evalFuncs
        w_tip = W[end]
        costFuncs["wtip"] = w_tip
    end
    if "psitip" in evalFuncs
        psi_tip = Ψ[end]
        costFuncs["psitip"] = psi_tip
    end
    if "lift" in evalFuncs
        TotalLift = sum(Lift)
        costFuncs["lift"] = TotalLift
    end
    if "moment" in evalFuncs
        TotalMoment = sum(Moments)
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
        foilTotalStates, nDOF = SolverRoutines.return_totalStates(completeStates, FOIL, CONSTANTS.elemType)
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


end


end # end module