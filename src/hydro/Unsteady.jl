# --- Julia 1.9---
"""
@File    :   Unsteady.jl
@Time    :   2024/01/31
@Author  :   Galen Ng
@Desc    :   Contains the unsteady hydrodynamics routines
"""

# --- PACKAGES ---
using SpecialFunctions

function compute_theodorsen(k::Number)
    """
    Theodorsen's transfer function for unsteady aero/hydrodynamics of a sinusoidally oscillating foil.
    w/ separate real and imaginary parts. 
    This is potential flow theory.
    This form is also wrong for k < 0 and should use the modified bessel funcs

    Inputs:
        k: float, reduced frequency of oscillation (a.k.a. Strouhal number)

    return:
        C(k)

    NOTE:
    Undefined for k = ωb/Ucos(Λ) = 0 (steady aero)
        """
    if k < 1.11e-16
        # println("You can't use the Theodorsen function for k = 0!")
        ans = [1.0, 0.0]
        return ans
    end

    # # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    # H₀²ᵣ = besselj0(k)
    # H₀²ᵢ = -bessely0(k)
    # H₁²ᵣ = besselj1(k)
    # H₁²ᵢ = -bessely1(k)

    # divDenom = 1 / ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    # # --- These are the analytic solutions to Theodorsen's function ---
    # C_r_analytic = (H₁²ᵣ * H₁²ᵣ - H₁²ᵣ * H₀²ᵢ + H₁²ᵢ * (H₀²ᵣ + H₁²ᵢ)) * divDenom
    # C_i_analytic = -(-H₁²ᵢ * (H₁²ᵣ - H₀²ᵢ) + H₁²ᵣ * (H₀²ᵣ + H₁²ᵢ)) * divDenom

    # ans = [C_r_analytic, C_i_analytic]

    # Modified Bessel functions of the second kind
    K₁ = besselk(1, 1im * k)
    K₀ = besselk(0, 1im * k)
    Ck = K₁ / (K₁ + K₀)
    ans = [real(Ck), imag(Ck)]

    return ans
end

function compute_sears(k)
    """
    Sears transfer function for an airfoil subject to sinusoidal gusts.
    This is potential flow theory.
    """

    # # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    # H₀²ᵣ = besselj0(k)
    # H₀²ᵢ = -bessely0(k)
    # H₁²ᵣ = besselj1(k)
    # H₁²ᵢ = -bessely1(k)

    # TODO: do in real data type only
    # divDenom = 1 / ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    # S_r = divDenom

    # H02 = H₀²ᵣ + 1im * H₀²ᵢ
    # H12 = H₁²ᵣ + 1im * H₁²ᵢ
    # Sk = 2 * 1im / (π * k) / (H12 + 1im * H02)

    # Modified Bessel functions of the second kind
    K₁ = besselk(1, 1im * k)
    K₀ = besselk(0, 1im * k)

    Sk = 1 / (1im * k) / (K₁ + K₀)

    # Leading edge Sears function
    S0k = exp(-1im * k) * Sk

    ans = [Sk, S0k]

    return ans
end

function compute_pade(k)
    """
    3-term Pade approximation of Theodorsen's function
    Swinney 1990 'A fractional calculus model of aeroelasticity'
    """
    s̄ = 1im * k
    scube = s̄^3
    ssquare = s̄^2
    C = (scube + 3.5 * ssquare + 2.7125 * s̄ + 0.46875) / (2 * scube + 6.5 * ssquare + 4.25 * s̄ + 0.46875)
    C_r = real(C)
    C_i = imag(C)
    ans = [C_r, C_i]
    return ans
end

function compute_fraccalc(k)
    """
    Fractional calculus approximation of Theodorsen's function
    Swinney 1990 'A fractional calculus model of aeroelasticity'
    """
    s̄ = 1im * k
    F = 2.19
    β = 5 / 6
    prod = F * s̄^β
    C = (1 + prod) / (1 + 2 * prod)
    C_r = real(C)
    C_i = imag(C)
    ans = [C_r, C_i]
    return ans
end

function compute_fraccalc_d(k)
    """
    Fractional calculus approximation of Theodorsen's function
    Swinney 1990 'A fractional calculus model of aeroelasticity'
    Undefined at s = 0 b/c beta = 5/6 :(
    """
    s̄ = 1im * k
    F = 2.19
    β = 5 / 6
    prod = F * s̄^β
    prod2 = F * s̄^(β - 1)
    C = ((1 + 2 * prod) * (β * 1im * prod2) - (1 + prod) * (2 * β * 1im * prod2)) / (1 + 2 * prod)^2
    C_r = real(C)
    C_i = imag(C)
    ans = [C_r, C_i]
    return ans
end

function compute_node_stiff_faster(
    clα, b, eb, ab, U∞, clambda, slambda, rho_f, Ck
)
    """
    Hydrodynamic stiffness force
    THIS ASSUMES THE MOMENT ABOUT THE AERODYNAMIC CENTER IS ZERO
    """
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b
    Uclambda = U∞ * clambda
    clalphabCk = clα * b * Ck
    # K_f = @SMatrix zeros(ComplexF64, 2, 2)
    # K̂_f = @SMatrix zeros(ComplexF64, 2, 2)
    # Aerodynamic quasi-steady stiffness
    # (1st row is lift, 2nd row is pitching moment)


    k_hα = -2 * clalphabCk # lift due to angle of attack
    k_αα = k_hα * eb # moment due to angle of attack (disturbing)
    K_f = qf * clambda * clambda *
          [
              0.0 k_hα
              0.0 k_αα
          ]

    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh = Uclambda * 2 * clα * Ck
    e_hα = Uclambda * (1 - a) * (-clalphabCk)
    e_αh = Uclambda * (1 + a) * clalphabCk
    e_αα = Uclambda *
           (π * b * b - clalphabCk * eb * (1 - 2 * (a)))
    K̂_f = qf / U∞ * slambda * b *
           [
               e_hh e_hα
               e_αh e_αα
           ]

    return K_f, K̂_f
end

function compute_node_stiff_dcla(
    b, eb, ab, U∞, clambda, slambda, rho_f, Ck
)
    """
    cla derivative of hydrodynamic stiffness force
    """
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b
    Uclambda = U∞ * clambda
    bCk = b * Ck


    k_hα = -2 * bCk # lift due to angle of attack
    k_αα = k_hα * eb # moment due to angle of attack (disturbing)
    dK_f = qf * clambda * clambda *
           [
               0.0 k_hα
               0.0 k_αα
           ]

    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh = Uclambda * 2 * Ck
    e_hα = Uclambda * (1 - a) * (-bCk)
    e_αh = Uclambda * (1 + a) * bCk
    e_αα = Uclambda *
           (-bCk * eb * (1 - 2 * (a)))
    dK̂_f = qf / U∞ * slambda * b *
            [
                e_hh e_hα
                e_αh e_αα
            ]

    return dK_f, dK̂_f
end

function compute_node_stiff_faster(
    clα::Number, b::Number, eb::Number, ab::Number, U∞::Number, clambda::Number, slambda::Number, rho_f::Number, Ck_r::Number, Ck_i::Number
)
    """
    Hydrodynamic stiffness force
    THIS ASSUMES THE MOMENT ABOUT THE AERODYNAMIC CENTER IS ZERO
    """
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b
    Uclambda = U∞ * clambda
    clalphabCk_r = clα * b * Ck_r
    clalphabCk_i = clα * b * Ck_i
    # K_f = @SMatrix zeros(ComplexF64, 2, 2)
    # K̂_f = @SMatrix zeros(ComplexF64, 2, 2)
    # Aerodynamic quasi-steady stiffness
    # (1st row is lift, 2nd row is pitching moment)


    k_hα_i = -2 * clalphabCk_r # lift due to angle of attack
    k_hα_r = -2 * clalphabCk_i # lift due to angle of attack
    k_αα_r = k_hα_r * eb # moment due to angle of attack (disturbing)
    k_αα_i = k_hα_i * eb # moment due to angle of attack (disturbing)
    K_f_r = qf * clambda * clambda *
            [
                0.0 k_hα_r
                0.0 k_αα_r
            ]
    K_f_i = qf * clambda * clambda *
            [
                0.0 k_hα_i
                0.0 k_αα_i
            ]


    # Sweep correction to aerodynamic quasi-steady stiffness
    e_hh_r = Uclambda * 2 * clα * Ck_r
    e_hh_i = Uclambda * 2 * clα * Ck_i
    e_hα_r = Uclambda * (1 - a) * (-clalphabCk_r)
    e_hα_i = Uclambda * (1 - a) * (-clalphabCk_i)
    e_αh_r = Uclambda * (1 + a) * clalphabCk_r
    e_αh_i = Uclambda * (1 + a) * clalphabCk_i
    # I MIGHT BE WRONG HERE
    e_αα_r = Uclambda * (π * b * b - clalphabCk_r * eb * (1 - 2 * (a)))
    e_αα_i = Uclambda * (-clalphabCk_i * eb * (1 - 2 * (a)))
    K̂_f_r = qf / U∞ * slambda * b *
             [
                 e_hh_r e_hα_r
                 e_αh_r e_αα_r
             ]
    K̂_f_i = qf / U∞ * slambda * b *
             [
                 e_hh_i e_hα_i
                 e_αh_i e_αα_i
             ]

    return K_f_r, K_f_i, K̂_f_r, K̂_f_i
end

function compute_node_damp_faster(clα, b, eb, ab, U∞, clambda, slambda, rho_f, Ck)
    """
    Fluid-added damping matrix
    """
    # --- Precomputes ---
    qf = 0.5 * rho_f * U∞ * U∞ # Dynamic pressure
    a = ab / b
    coeff = qf / U∞ * b

    # Aerodynamic quasi-steady damping
    # (1st row is lift, 2nd row is pitching moment)
    c_hh = 2 * clα * Ck
    c_hα = -b * (2π + clα * (1 - 2 * a) * Ck)
    c_αh = 2 * eb * clα * Ck
    c_αα = 0.5 * b * (1 - 2 * a) * (2π * b - 2 * clα * eb * Ck)
    C_f = coeff * clambda *
          [
              c_hh c_hα
              c_αh c_αα
          ]

    # Sweep correction to aerodynamic quasi-steady damping
    e_hh = 2π * b
    e_hα = 2π * ab * b
    e_αh = e_hα
    e_αα = 2π * b^3 * (0.125 + a * a)
    Ĉ_f = coeff * slambda *
           [
               e_hh e_hα
               e_αh e_αα
           ]

    return C_f, Ĉ_f
end

function compute_node_mass(b, ab, rho_f)
    """
    Fluid-added mass matrix
    """
    # --- Precomputes ---
    bSquared = b * b # precompute square of b
    a = ab / b # precompute division by b to get a

    m_hh = 1.0
    m_hα = ab
    m_αh = m_hα
    m_αα = bSquared * (0.125 + a * a)
    M_f = π * rho_f * bSquared *
          [
              m_hh m_hα
              m_αh m_αα
          ]

    return M_f
end

# let
#     using .Unsteady
#     precompile(Unsteady.compute_node_mass, (Float64, Float64, Float64))
#     precompile(Unsteady.compute_node_damp_faster, (Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64))
# end
