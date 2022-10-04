# --- Julia ---

# @File    :   SolveSteady.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Solve the beam equations w/ steady aero/hydrodynamics and compute gradients

module SolveSteady
"""
Steady hydroelastic solver module
"""

# --- Public functions ---
export solve

# --- Libraries ---
using FLOWMath: linear, akima
using FiniteDifferences
using LinearAlgebra, Statistics
using JSON
# using ForwardDiff
# using ReverseDiff
using Zygote
include("InitModel.jl")
include("Struct.jl")
include("struct/FiniteElements.jl")
include("Hydro.jl")
include("GovDiffEqns.jl")
using .InitModel, .Hydro, .StructProp, .Steady, .Solver
using .FEMMethods

function solve(neval::Int64, DVDict, outputDir::String)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)

    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    global FOIL = InitModel.init_steady(neval, DVDict) # seems to only be global in this module
    nElem = neval - 1
    constitutive = FOIL.constitutive

    # ************************************************
    #     Solve FEM first time
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
    fTractions, AIC, planformArea = compute_hydroLoads(u, structMesh, elemType)
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
    #     Converge r(u) = 0
    # ************************************************
    # --- Assign constants accessible in this module ---
    # This is needed for derivatives!
    derivMode = "FAD"
    global CONSTANTS = InitModel.DCFoilConstants(K, elemType, structMesh, AIC, derivMode, planformArea)

    qSol, resVec = converge_r(q)
    if CONSTANTS.elemType == "BT2"
        uSol = vcat([0, 0, 0, 0], qSol)
    end

    # --- Get hydroLoads again on solution ---
    fTractions, AIC, _ = compute_hydroLoads(uSol, structMesh, elemType)


    # # ************************************************
    # #     Compute sensitivities
    # # ************************************************
    # mode = "FAD"
    # ∂r∂u = compute_∂r∂u(q, mode)


    # ************************************************
    #     Write solution out to files
    # ************************************************
    # Write solution to .dat file
    TotalLift, TotalMoment, CL, CM = write_sol(uSol, globalF, elemType, outputDir)

    println("TotalLift = ", TotalLift, " N")
    println("TotalMoment (about midchord) = ", TotalMoment, " N-m")
    println("CL = ", CL)
    println("CM = ", CM)

end

function return_totalStates(foilStructuralStates, elemType="BT2")
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
    foilTotalStates, nDOF = return_totalStates(foilStructuralStates, elemType)
    nGDOF = FOIL.neval * nDOF

    # ---------------------------
    #   Strip theory
    # ---------------------------
    AIC = zeros(nGDOF, nGDOF)
    _, planformArea = compute_AIC!(AIC, mesh, elemType)

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

function compute_AIC!(AIC, mesh, elemType="BT2")

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
    elseif elemType == "BT2"
        nDOF = 4
    end

    # fluid dynamic pressure    
    qf = 0.5 * FOIL.ρ_f * FOIL.U∞^2

    # strip width
    dy = FOIL.s / (FOIL.neval)
    planformArea = 0.0

    jj = 1 # node index
    for yⁿ in mesh
        K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
        E_f = copy(K_f)  # Sweep correction matrix

        # --- Linearly interpolate values based on y loc ---
        clα = linear(mesh, FOIL.clα, yⁿ)
        c = linear(mesh, FOIL.c, yⁿ)
        b = 0.5 * c # semichord for more readable code
        ab = linear(mesh, FOIL.ab, yⁿ)
        eb = linear(mesh, FOIL.eb, yⁿ)

        # --- Compute forces ---
        # Aerodynamic stiffness (1st row is lift, 2nd row is pitching moment)
        k_hα = -2 * b * clα # lift due to angle of attack
        k_αα = -2 * eb * b * clα # moment due to angle of attack
        K_f = qf * cos(FOIL.Λ)^2 *
              [
                  0.0 k_hα
                  0.0 k_αα
              ]
        # Sweep correction to aerodynamic stiffness
        e_hh = 2 * clα # lift due to w'
        e_hα = -clα * b * (1 - ab / b) # lift due to ψ'
        e_αh = clα * b * (1 + ab / b) # moment due to w'
        e_αα = π * b^2 - 0.5 * clα * b^2 * (1 - (ab / b)^2) # moment due to ψ'
        E_f = qf * sin(FOIL.Λ) * cos(FOIL.Λ) * b *
              [
                  e_hh e_hα
                  e_αh e_αα
              ]

        # --- Compute Compute local AIC matrix for this element ---
        if elemType == "bend-twist"
            println("These aerodynamics are all wrong BTW...")
            AICLocal = -1 * [
                0.00000000 0.0 K_f[1, 2] # Lift
                0.00000000 0.0 0.00000000
                0.00000000 0.0 K_f[2, 2] # Pitching moment
            ]
        elseif elemType == "BT2"
            AICLocal = [
                0.0 E_f[1, 1] K_f[1, 2] E_f[1, 2]  # Lift
                0.0 0.0 0.0 0.0
                0.0 E_f[2, 1] K_f[2, 2] E_f[2, 2] # Pitching moment
                0.0 0.0 0.0 0.0
            ]
        else
            println("nothing else works")
        end


        GDOFIdx = nDOF * (jj - 1) + 1

        AIC[GDOFIdx:GDOFIdx+nDOF-1, GDOFIdx:GDOFIdx+nDOF-1] = AICLocal

        # Add rectangle to planform area
        planformArea += c * dy

        jj += 1 # increment strip counter
    end

    return AIC, planformArea
end

function write_sol(states, forces, elemType="bend", outputDir="./OUTPUT/")
    """
    Inputs
    ------
        states: vector of structural states from the [K]{u} = {F}
    """

    mkpath(outputDir)

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

    # --- Total hydro force calcs ---
    TotalLift = sum(Lift) * FOIL.s / FOIL.neval
    TotalMoment = sum(Moments) * FOIL.s / FOIL.neval
    # CL and CM
    CL = TotalLift / (0.5 * FOIL.ρ_f * FOIL.U∞^2 * CONSTANTS.planformArea)
    CM = TotalMoment / (0.5 * FOIL.ρ_f * FOIL.U∞^2 * CONSTANTS.planformArea * mean(FOIL.c))

    funcs = Dict(
        "StaticLift" => TotalLift,
        "StaticMoment" => TotalMoment,
        "CL" => CL,
        "CM" => CM,
    )

    fname = outputDir * "funcs.json"
    stringData = JSON.json(funcs)
    open(fname, "w") do io
        write(io, stringData)
    end

    return TotalLift, TotalMoment, CL, CM
end

# ==============================================================================
#                         Solver routine
# ==============================================================================
function do_newton_rhapson(u, maxIters=200, tol=1e-12, verbose=true, mode="FAD")
    """
    Simple Newton-Rhapson solver
    """

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


function converge_r(u, maxIters=200, tol=1e-6, verbose=true, mode="FAD")
    """
    Given input u, solve the system r(u) = 0
    """

    # ************************************************
    #     Main solver loop
    # ************************************************
    converged_u, converged_r = do_newton_rhapson(u, maxIters, tol, verbose, mode)

    return converged_u, converged_r

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
    Jacobian of residuals with respect to structural states EXCLUDING BC
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
        foilTotalStates, nDOF = return_totalStates(vcat([0.0, 0.0, 0.0, 0.0], structuralStates), CONSTANTS.elemType)
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

function compute_jacobian(stateVec, FOIL=nothing)
    """
    Compute the jacobian df/dx

    Inputs:
        stateVec: array, shape (8), state vector

    returns:
        array, shape (, 8), square jacobian matrix

    """
    # ************************************************
    #     Compute cost func gradients
    # ************************************************

    # TODO:

end


end # end module