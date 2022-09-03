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
include("InitModel.jl")
include("Struct.jl")
include("struct/FiniteElements.jl")
include("Hydro.jl")
include("GovDiffEqns.jl")
using .InitModel, .Hydro, .StructProp, .Steady, .Solver
using .FEMMethods

function solve(neval::Int64, DVDict)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)

    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    foil = InitModel.init_steady(neval, DVDict)
    η = LinRange(0, 1, foil.neval) # parametric spanwise var
    y = LinRange(0, foil.s, foil.neval) # real spanwise var
    # initialize a BC vector at the root. We know first 4 are zero
    q⁰ = zeros(8)
    q⁰[5:end] .= 1.0

    # ************************************************
    #     Solve FEM
    # ************************************************
    nElem = neval - 1
    elemType = "bend-twist"
    constitutive = "isotropic"
    globalDOFBlankingList = [1, 2, 3] # NOTE: THIS BLANKS THE ROOT NODE AND SHOULD BE A COMMAND-LINE INPUT
    globalF = zeros(6) # DUMMY NOTE: FIX THIS LATER

    structMesh, elemConn = FEMMethods.make_mesh(nElem, foil)
    globalK, globalM = FEMMethods.assemble(structMesh, elemConn, foil, elemType, constitutive)

    # # --- Debug printout of matrices in human readable form ---
    # println("Global stiffness matrix:")
    # println("------------------------")
    # show(stdout, "text/plain", globalK)
    # println("")
    # println("Global mass matrix:")
    # println("-------------------")
    # show(stdout, "text/plain", globalM)

    K, M, F = FEMMethods.apply_BCs(globalK, globalM, globalF, globalDOFBlankingList)

    # # --- Debug printout of matrices in human readable form ---
    # println("Global stiffness matrix:")
    # println("------------------------")
    # show(stdout, "text/plain", K)
    # println("")
    # println("Global mass matrix:")
    # println("-------------------")
    # show(stdout, "text/plain", M)

    # TODO: finish the verification and do some visualization 22/09/02

    # yBaseSol = LinRange(0, foil.s, nElem)
    # for ii ∈ 1:8 # spline the soln into the vector TODO: this is wrong
    #     foilPDESol[ii, :] = akima(yBaseSol, qsol[ii, :], y)
    # end

    # # ---------------------------
    # #   Hydro loads
    # # ---------------------------
    # F₀, M₀ = compute_hydroLoads(foilPDESol, foil)

    # # ************************************************
    # #     Compute residuals
    # # ************************************************
    # resVec = 

    # # ************************************************
    # #     Compute sensitivities
    # # ************************************************



end

function compute_hydroLoads(foilPDESol, foil)
    """
    Computes the steady hydrodynamic vector loads 
    given the solved hydrofoil shape (strip theory)

    """
    # ---------------------------
    #   Initializations
    # ---------------------------
    # --- Unpack solution ---
    w = foilPDESol[1, :]
    ψ = foilPDESol[2, :]
    ∂w∂y = foilPDESol[3, :]
    ∂ψ∂y = foilPDESol[4, :]

    y = LinRange(0, foil.s, foil.neval) # real spanwise var

    qf = 0.5 * foil.ρ_f * foil.U∞^2 # fluid dynamic pressure

    K_f = zeros(2, 2) # Fluid de-stiffening (disturbing) matrix
    E_f = copy(K_f)     # Sweep correction matrix

    F = zeros(foil.neval) # Hydro force vec
    M = copy(F) # Hydro moment vec

    # ---------------------------
    #   Strip theory loop
    # ---------------------------
    jj = 1
    for yⁿ in y
        # --- Linearly interpolate values based on y loc ---
        clα = linear(y, foil.clα, yⁿ)
        c = linear(y, foil.c, yⁿ)
        b = 0.5 * c # semichord for more readable code
        ab = linear(y, foil.ab, yⁿ)
        eb = linear(y, foil.eb, yⁿ)

        # --- Generalized coord vec w/ 2DOF---
        qGen = [w[jj], ψ[jj] + foil.α₀ * π / 180]
        ∂qGen∂y = [∂w∂y[jj], ∂ψ∂y[jj]]

        # --- Compute forces ---
        K_f = qf * cos(foil.Λ)^2 *
              [
                  0.0 -2*b*clα
                  0.0 -2*eb*b*clα
              ]

        E_f = qf * sin(foil.Λ) * cos(foil.Λ) * b *
              [
                  2*clα -clα*b*(1-ab/b)
                  clα*b*(1+ab/b) π*b^2-0.5*clα*b^2*(1-(ab/b)^2)
              ]

        # Force (+ve up)
        FStrip = -K_f[1, 2] .* qGen[2]
        -E_f[1, 1] .* qGen[1]
        -E_f[1, 1] * ∂qGen∂y[1]

        # Moment about mid-chord (+ve nose up)
        MStrip = -K_f[2, 2] .* qGen[2]
        -E_f[2, 1] .* ∂qGen∂y[1]
        -E_f[2, 2] .* ∂qGen∂y[2]

        # Append to solution
        F[jj] = FStrip
        M[jj] = MStrip

        jj += 1 # increment strip counter
    end

    return F, M
end

function write_sol()

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

function compute_∂r∂u(foilPDESol)

end

# function compute_residual(stateVec, FHydro, MHydro, foil)
#     """
#     NOTE: This is actually all wrong, disregard this and do not use this code
#     Compute residual

#     Inputs
#     ------
#     stateVec : array
#         State vector [w, ψ, ∂w∂y, ∂ψ∂y]
#     """

#     # --- Unpack values ---
#     ∂⁴w∂y⁴ = stateVec[9, :]
#     ∂⁴ψ∂y⁴ = stateVec[10, :]
#     ∂³w∂y³ = stateVec[7, :]
#     ∂³ψ∂y³ = stateVec[8, :]
#     ∂²ψ∂y² = stateVec[6, :]

#     # --- Governing PDEs as residuals ---
#     # TODO: WHAT IS THE CAUSE OF THIS BEING HELLA WRONG? DO WE NEED TO DO FEM instead?
#     r₁ = foil.EIₛ .* ∂⁴w∂y⁴ + foil.ab .* foil.EIₛ .* ∂⁴ψ∂y⁴ + foil.Kₛ .* ∂³ψ∂y³ - FHydro
#     r₂ = foil.ab .* foil.EIₛ .* ∂⁴w∂y⁴ - foil.Kₛ .* ∂³w∂y³ + foil.Sₛ .* ∂⁴ψ∂y⁴ - foil.GJₛ .* ∂²ψ∂y² - MHydro
#     #Something may be wrong here with dimensions

#     # --- Get 8 additional res from BCs ---
#     resBC = Steady.compute_g(stateVec[1:8, 1], stateVec[1:8, end], foil)

#     # --- Stack them ---
#     resVec = []

# end

function compute_direct()
    """
    Computes direct vector 
    """
end

function compute_adjoint()

end

function compute_jacobian(stateVec, foil=nothing)
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