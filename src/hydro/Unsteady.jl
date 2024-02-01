# --- Julia 1.9---
"""
@File    :   Unsteady.jl
@Time    :   2024/01/31
@Author  :   Galen Ng
@Desc    :   Contains the unsteady hydrodynamics routines
"""

module Unsteady

# --- Public functions ---
export compute_theodorsen, compute_sears

# --- Libraries ---
using SpecialFunctions
using LinearAlgebra
using Statistics
using Zygote, ChainRulesCore
using Printf, DelimitedFiles

# --- Globals ---
include("../constants/SolutionConstants.jl")
using .SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE

function compute_theodorsen(k)
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
        println("You can't use the Theodorsen function for k = 0!")
        #     # println(k)
        #     k += 1.11e-16 # force it to be non-zero
        #     #     CᵣLim = 1.0
        #     #     Cᵢ = 0.0
        #     #     ans = [Cᵣ, Cᵢ]
    end

    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ = besselj0(k)
    H₀²ᵢ = -bessely0(k)
    H₁²ᵣ = besselj1(k)
    H₁²ᵢ = -bessely1(k)

    divDenom = 1 / ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    # --- These are the analytic solutions to Theodorsen's function ---
    C_r_analytic = (H₁²ᵣ * H₁²ᵣ - H₁²ᵣ * H₀²ᵢ + H₁²ᵢ * (H₀²ᵣ + H₁²ᵢ)) * divDenom
    C_i_analytic = -(-H₁²ᵢ * (H₁²ᵣ - H₀²ᵢ) + H₁²ᵣ * (H₀²ᵣ + H₁²ᵢ)) * divDenom

    # # --- Zero frequency limit ---
    # Cᵣ_lim = 1.0
    # Cᵢ_lim = 0.0
    # kSigmoid = 1000.0 # sigmoid steepness
    # logistic = 1 / (1 + exp(-kSigmoid * -1 * (k - 0.0))) # this is a L-R flipped sigmoid so below 0 the function is 1.0

    # C_r = Cᵣ_lim * logistic + C_r_analytic
    # C_i = Cᵢ_lim * logistic + C_i_analytic
    ans = [C_r_analytic, C_i_analytic]

    return ans
end

function compute_sears(k)
    """
    Sears transfer function for an airfoil subject to sinusoidal gusts.
    This is potential flow theory.
    """

    # Hankel functions (Hᵥ² = 𝙹ᵥ - i𝚈ᵥ) of the second kind with order `ν`
    H₀²ᵣ = besselj0(k)
    H₀²ᵢ = -bessely0(k)
    H₁²ᵣ = besselj1(k)
    H₁²ᵢ = -bessely1(k)

    # TODO: do in real data type only
    # divDenom = 1 / ((H₁²ᵣ - H₀²ᵢ) * (H₁²ᵣ - H₀²ᵢ) + (H₀²ᵣ + H₁²ᵢ) * (H₀²ᵣ + H₁²ᵢ))

    # S_r = divDenom

    H02 = H₀²ᵣ + 1im * H₀²ᵢ
    H12 = H₁²ᵣ + 1im * H₁²ᵢ
    S = 2 * 1im / (π * k) / (H12 + 1im * H02)

    return S
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

end # end module
