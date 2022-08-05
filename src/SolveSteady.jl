# --- Julia ---

# @File    :   SolveSteady.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Solve the beam equations for a steady aero/hydrodynamics and compute gradients


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
include("Hydro.jl")
include("GovDiffEqns.jl")
using .InitModel, .Hydro, .StructProp, .Steady, .Solver

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
    #     Solve the diff eq
    # ************************************************
    foilPDESol = zeros(8, foil.neval)
    # --- Solves a 2PT BVP ---
    ysol, qsol = Solver.solve_bvp(Steady.compute_∂q∂y, q⁰, 0, 1, 25, Steady.compute_g, foil)
    foilPDESol[1, :] = akima(ysol, qsol[1, :], y)
    w = foilPDEsol[1, :]
    ψ = foilPDEsol[2, :]
    ∂w∂y = foilPDEsol[3, :]
    ∂ψ∂y = foilPDEsol[4, :]

    # ---------------------------
    #   Hydro loads
    # ---------------------------
    F₀, M₀ = compute_hydroLoads(foilPDESol, foil)
    # TODO:maybe use mutable struct to store the solution?

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
    w = foilPDEsol[1, :]
    ψ = foilPDEsol[2, :]
    ∂w∂y = foilPDEsol[3, :]
    ∂ψ∂y = foilPDEsol[4, :]

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
        qGen = [w[jj], ψ[jj] + foil.α₀]
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

function compute_residual(stateVec, FHydro, MHydro)
    """
    Compute residual
    """

    # --- Unpack values ---
    ∂⁴w∂y⁴ =  
    ∂⁴ψ∂y⁴ =
    ∂³w∂y³ = 
    ∂³ψ∂y³ =
    ∂²ψ∂y² = 

    # --- Governing PDEs as residuals ---
    r₁ = foil.EIₛ*∂⁴w∂y⁴ + foil.ab*foil.EIₛ*∂⁴ψ∂y⁴ + foil.Kₛ* ∂³ψ∂y³ - FHydro
    r₂ = foil.ab*foil.EIₛ*∂⁴w∂y⁴ - foil.Kₛ* ∂³w∂y³ + foil.Sₛ*∂⁴ψ∂y⁴ - foil.GJₛ*∂²ψ∂y² - MHydro 

    # --- Get 8 additional res from BCs ---
    resBC = Steady.compute_g(ua, ub, foil)

    # --- Stack them ---
    resVec = 

end

function compute_jacobian(stateVec)
    """
    Compute the jacobian dr/du

    Inputs:
        stateVec: array, shape (neval, 8), state vector for 'neval' spatial nodes

    returns:
        array, shape (neval, neval), square jacobian matrix

    """
    # TODO:

end

function write_sol()

end

end # end module