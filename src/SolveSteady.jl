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
using LinearAlgebra
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
    #   Get fluid tractions
    # ---------------------------
    fTractions = compute_hydroLoads(u, structMesh, elemType)
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
    #   Solve system
    # ---------------------------
    q = FEMMethods.solve_structure(K, M, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ 1:length(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q

    # --- Assign constants accessible in this module ---
    # This is needed for derivatives!
    global CONSTANTS = InitModel.DCFoilConstants(K, elemType, structMesh)

    # # ************************************************
    # #     Converge r(u) = 0
    # # ************************************************
    # uSol, resVec = converge_r(q, 1)


    # # ************************************************
    # #     Compute sensitivities
    # # ************************************************
    # mode = "FAD"
    # ∂r∂u = compute_∂r∂u(q, mode)


    # ************************************************
    #     Write solution out to files
    # ************************************************
    # Write solution to .dat file
    write_sol(u, globalF, elemType, outputDir)


end

function compute_hydroLoads(foilStructuralStates, mesh, elemType="bend-twist")
    """
    Computes the steady hydrodynamic vector loads 
    given the solved hydrofoil shape (strip theory)

    TODO: I THINK THE BUG IS HERE

    """
    # ---------------------------
    #   Initializations
    # ---------------------------
    alfaRad = FOIL.α₀ * π / 180
    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
        staticOffset = [0, 0, alfaRad] #TODO: pretwist will change this
    elseif elemType == "BT2"
        nDOF = 4
        staticOffset = [0, 0, alfaRad, 0] #TODO: pretwist will change this
    end

    # Add static angle of attack to deflected foil
    w = foilStructuralStates[1:nDOF:end]
    foilTotalStates = copy(foilStructuralStates) + repeat(staticOffset, outer=[length(w)])

    # fluid dynamic pressure    
    qf = 0.5 * FOIL.ρ_f * FOIL.U∞^2

    K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
    E_f = copy(K_f)  # Sweep correction matrix

    F = zeros(FOIL.neval) # Hydro force vec
    M = copy(F) # Hydro moment vec
    fTractions = Vector{Float64}(undef, FOIL.neval * nDOF)


    # ---------------------------
    #   Strip theory loop
    # ---------------------------
    dummyAIC = zeros(FOIL.neval * nDOF, FOIL.neval * nDOF)
    bufferAIC = Zygote.Buffer(dummyAIC) # AIC matrix
    jj = 1 # node index
    for yⁿ in mesh
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


        globalDOFIdx = nDOF * (jj - 1) + 1

        bufferAIC[globalDOFIdx:globalDOFIdx+nDOF-1, globalDOFIdx:globalDOFIdx+nDOF-1] = AICLocal

        jj += 1 # increment strip counter
    end
    AIC = copy(bufferAIC)

    fTractions = AIC * foilStructuralStates

    return fTractions
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


end

# ==============================================================================
#                         Solver routine
# ==============================================================================
function do_newton_rhapson(u, maxIters=200, tol=1e-12, verbose=true)
    """
    Simple Newton-Rhapson solver
    """

    mode = "FAD"

    for ii in 1:maxIters

        res = compute_residuals(u)
        ∂r∂u = compute_∂r∂u(u, mode)
        jac = ∂r∂u[1]

        # --- Newton step ---
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
            global converged_u = copy(u)
            global converged_r = copy(res)
        else
            global converged_u = copy(u)
            global converged_r = copy(res)
        end
    end
    print(converged_u)

    return converged_u, converged_r
end


function converge_r(u, maxIters=200, tol=1e-6)
    """
    Given input u, solve the system r(u) = 0
    """

    # ************************************************
    #     Main solver loop
    # ************************************************
    converged_u, converged_r = do_newton_rhapson(u, maxIters, tol)

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
        @time ∂r∂u = FiniteDifferences.jacobian(central_fdm(3, 1), compute_residuals, structuralStates)

    elseif mode == "FAD" # Forward automatic differentiation
        @time ∂r∂u = Zygote.jacobian(compute_residuals, structuralStates)

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
        F = compute_hydroLoads(vcat([0.0, 0.0, 0.0], structuralStates), CONSTANTS.mesh, CONSTANTS.elemType)
        FOut = F[4:end]
    elseif CONSTANTS.elemType == "BT2" # knock off root element
        F = compute_hydroLoads(vcat([0.0, 0.0, 0.0, 0.0], structuralStates), CONSTANTS.mesh, CONSTANTS.elemType)
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