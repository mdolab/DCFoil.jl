# --- Julia ---

# @File    :   hydro.jl
# @Time    :   2022/05/18
# @Author  :   Galen Ng
# @Desc    :   Contains hydrodynamic routines
# TODO: declare data types for performance improvements

module Hydro
"""
Hydrodynamics module
"""
# --- Public functions ---
export compute_theodorsen, compute_glauert_circ, compute_added_mass

# --- Libraries ---
using FLOWMath: linear
using SpecialFunctions
using LinearAlgebra
using Plots

function compute_theodorsen(k)
    """
    Theodorsen's transfer function for unsteady aero/hydrodynamics 
    w/ separate real and imaginary parts. This is potential flow theory.

    Inputs:
        k: float, reduced frequency of oscillation (a.k.a. Strouhal number)

    return:
        C(k)

    NOTE:
    Undefined for k = ωb/Ucos(Λ) = 0 (steady aero)
    """
    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ = besselj0(k)
    H₀²ᵢ = -bessely0(k)
    H₁²ᵣ = besselj1(k)
    H₁²ᵢ = -bessely1(k)

    denom = ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    𝙲ᵣ = (H₁²ᵣ * H₁²ᵣ - H₁²ᵣ * H₀²ᵢ + H₁²ᵢ * (H₀²ᵣ + H₁²ᵢ)) / denom
    𝙲ᵢ = -(-H₁²ᵢ * (H₁²ᵣ - H₀²ᵢ) + H₁²ᵣ * (H₀²ᵣ + H₁²ᵢ)) / denom

    ans = [𝙲ᵣ, 𝙲ᵢ]

    return ans
end

function compute_glauert_circ(; semispan, chordVec, α₀, U∞, neval)
    """
    Glauert's solution for the lift slope on a 3D hydrofoil

    The coordinate system is

    clamped root                         free tip
    `+-----------------------------------------+  (x=0 @ LE)
    `|                                         |
    `|               +-->y                     |
    `|               |                         |
    `|             x v                         |
    `+-----------------------------------------+
    `
    (y=0 @ root)

    where z is out of the page (thickness dir.)
    inputs:
        α₀: float, angle of attack [rad]

    returns:
        cl_α : array, shape (neval,)
            sectional lift slopes for a 3D wing [rad⁻¹] starting from the root
            sometimes denoted in literature as 'a₀'

    NOTE:
    We use keyword arguments (denoted by the ';' to be more explicit)

    This follows the formulation in 
    'Principles of Naval Architecture Series (PNA) - Propulsion 2010' 
    by Justin Kerwin & Jacques Hadler
    """

    ỹ = π / 2 * ((1:1:neval) / neval) # parametrized y-coordinate (0, π/2) NOTE: in PNA, ỹ is from 0 to π for the full span
    y = -semispan * cos.(ỹ) # the physical coordinate (y) is only calculated to the root (-semispan, 0)

    # ---------------------------
    #   PLANFORM SHAPES: rectangular is outdated
    # ---------------------------
    # # --- Rectangular ---
    # chordₚ = chord
    # --- Elliptical planform ---
    chordₚ = chordVec .* sin.(ỹ) # parametrized chord goes from 0 to the original chord value from tip to root...corresponds to amount of downwash w(y)?

    n = (1:1:neval) * 2 - ones(neval) # node numbers x2 (node multipliers)

    b = π / 4 * (chordₚ / semispan) * α₀ .* sin.(ỹ) # RHS vector

    ỹn = ỹ .* n' # outer product of ỹ and n, matrix of [0, π/2]*node multipliers

    sinỹ_mat = repeat(sin.(ỹ), outer=[1, neval]) # parametrized square matrix where the columns go from 0 to 1
    chord_ratio_mat = π / 4 * chordₚ / semispan .* n' # outer product of [0,...,tip chord-semispan ratio] and [1:2:neval*2-1] so the columns are the chord-span ratio vector times node multipliers with π/4 in front

    chord11 = sin.(ỹn) .* (chord_ratio_mat + sinỹ_mat) #matrix-matrix multiplication to get the [A] matrix

    # --- Solve for the coefficients in Glauert's Fourier series ---
    ã = chord11 \ b

    γ = 4 * U∞ * semispan .* (sin.(ỹn) * ã) # span-wise free vortex strength (Γ/semispan)

    cl = (2 * γ) ./ (U∞ * chordVec) # sectional lift coefficient cl(y) = cl_α*α
    clα = cl / (α₀ + 1e-12) # sectional lift slope clα but on parametric domain; use safe check on α=0

    # --- Interpolate lift slopes onto domain ---
    pGlauert = plot(LinRange(0, 2.7, 250), clα)
    cl_α = linear(y, clα, LinRange(-semispan, 0, neval)) # Use BYUFLOW lab math function

    return reverse!(cl_α)
end

# function compute_added_mass(ρ_f, chordVec)
#     """
#     Compute the added mass for a rectangular cross section

#     return:
#         added mass, Array
#         added inertia, Array
#     """
#     mₐ = π * ρ_f * chordVec .* chordVec / 4 # Fluid-added mass vector [kg/m]
#     Iₐ = π * ρ_f * chordVec .^ 4 / 128 # Fluid-added inertia [kg-m]

#     return mₐ, Iₐ
# end

# ************************************************
#     Hydrodynamic strip forces
# ************************************************
function compute_node_stiff(clα, b, eb, ab, U∞, Λ, ω, rho_f)
    qf = 0.5 * rho_f * U∞^2 # Dynamic pressure
    k = ω * b / (U∞ * cos(Λ)) # reduced frequency

    # Do computation once for efficiency
    CK = compute_theodorsen(k)
    Ck = CK[1] + 1im * CK[2] # TODO: for now, put it back together so solve is easy to debug

    # Aerodynamic quasi-steady stiffness 
    # (1st row is lift, 2nd row is pitching moment)

    k_hα = -2 * b * clα * Ck # lift due to angle of attack
    k_αα = -2 * eb * b * clα * Ck # moment due to angle of attack
    K_f = qf * cos(Λ)^2 *
          [
              0.0 k_hα
              0.0 k_αα
          ]
    # Sweep correction to aerodynamic quasi-steady stiffness (THERE ARE TIME DERIV TERMS)
    e_hh =
    # lift due to w'
        U∞ * cos(Λ) * 2 * clα * Ck +
        # lift due to ∂²w/∂t∂y
        2 * π * b * im * ω
    e_hα =
    # lift due to ψ'
        U∞ * cos(Λ) * -clα * b * (1 - ab / b) * Ck +
        # lift due to ∂²ψ/∂t∂y
        2 * π * ab * b * im * ω
    e_αh =
    # moment due to w'
        U∞ * cos(Λ) * clα * b * (1 + ab / b) * Ck +
        # moment due to ∂²w/∂t∂y
        2 * π * ab * b * im * ω
    e_αα =
    # moment due to ψ'
        U∞ * cos(Λ) * π * b^2 - 0.5 * clα * b^2 * (1 - (ab / b)^2) * Ck +
        # moment due to ∂²ψ/∂t∂y
        2 * π * b^3 * (0.125 + (ab / b)^2) * im * ω
    E_f = qf / U∞ * sin(Λ) * b *
          [
              e_hh e_hα
              e_αh e_αα
          ]
    return K_f, E_f
end


function compute_node_damp(clα, b, eb, ab, U∞, Λ, ω, rho_f)
    """
    Fluid-added damping matrix
    """
    qf = 0.5 * rho_f * U∞^2 # Dynamic pressure
    k = ω * b / (U∞ * cos(Λ)) # reduced frequency

    # Do computation once for efficiency
    CK = compute_theodorsen(k)
    Ck = CK[1] + 1im * CK[2] # TODO: for now, put it back together so solve is easy to debug

    # Aerodynamic quasi-steady damping
    # (1st row is lift, 2nd row is pitching moment)
    c_hh = 2 * clα * Ck
    c_hα = -b * (2 * π + clα * (1 - 2 * ab / b) * Ck)
    c_αh = 2 * eb * clα * Ck
    c_αα = 0.5 * b * (1 - 2 * ab / b) * (2 * π * b - 2 * clα * eb * Ck)
    C_f = qf / U∞ * cos(Λ) * b *
          [
              c_hh c_hα
              c_αh c_αα
          ]
    return C_f
end
function compute_node_mass(b, ab, ω, rho_f)
    """
    Fluid-added mass matrix
    """
    m_hh = 1
    m_hα = ab
    m_αh = ab
    m_αα = b^2 * (0.125 + (ab / b)^2)
    M_f = π * rho_f * b^2 *
          [
              m_hh m_hα
              m_αh m_αα
          ]

    return M_f
end

end # end module

