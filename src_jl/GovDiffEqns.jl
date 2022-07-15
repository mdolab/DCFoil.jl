# --- Julia ---

# @File    :   GovDiffEqns.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module containing the governing differential equations recasting as a linear system
#              q' = f(q(y)) 
#              where q = [w, ψ, w', ψ', w'', ψ'', w''', ψ''']ᵀ
#              The functions in this module are the 'f' in the above equation
# 
# TODO: It might be better to redo the solution algorithm to be the finite element method
# i.e. one that solve A\b = {u}

module GovDiffEqns

using FLOWMath: linear
using LinearAlgebra

module Steady
"""
Steady differential equations module
All time derivative terms ∂/∂t() = 0 and C(k=0) = 1
"""
export ∂q∂y

function ∂q∂y(q::Array, yⁿ::Float64, foil)
    """
    Compute the derivative of the state vector q with respect to the spatial variable y and location of node yⁿ
    """
    # TODO: DEBUG ALL OF THIS
    # --- First interpolate all necessary values based on spanwise location ---
    y = LinRange(-foil.s, 0, foil.neval)
    clα = linear(y, foil.clα, yⁿ)
    c = linear(y, foil.c, yⁿ)
    b = 0.5 * c # semichord for more readable code
    ab = linear(y, foil.ab, yⁿ)
    eb = linear(y, foil.eb, yⁿ)
    EIₛ = linear(y, foil.EIₛ, yⁿ)
    GJₛ = linear(y, foil.GJₛ, yⁿ)
    Kₛ = linear(y, foil.Kₛ, yⁿ)
    Sₛ = linear(y, foil.Sₛ, yⁿ)
    q[2] += foil.α * π / 360 # update the angle of attack to be total

    # --- Compute governing matrix equations ---
    # NOTE: the convention is [w, ψ]ᵀ for the indexing
    qf = 0.5 * foil.ρ_f * foil.U∞^2 # dynamic pressure
    # Fluid de-stiffening (disturbing)
    K_f = qf * cos(foil.Λ)^2 *
          [
              0.0, -2 * b * clα;
              0.0, -2 * eb * b * clα
          ]

    # Sweep correction matrix
    E_f = qf * sin(foil.Λ) * cos(foil.Λ) * b
    [
        2 * clα, -clα * b * (1 - ab / b);
        clα * b * (1 + ab / b), π * b^2 - 0.5 * clα * b^2 * (1 - (ab / b)^2)
    ]

    # --- Build the linear system ---
    # 4th deriv terms: w'''', ψ''''
    A = (1 / L^4) *
        [
        EIₛ, 0.5 * ab * EIₛ;
        0.5 * ab * EIₛ, Sₛ
    ]
    # 3rd deriv terms: w''', ψ'''
    B = (1 / L^3) *
        [
        0, Kₛ;
        Kₛ, 0
    ]
    # 2nd deriv terms: w'', ψ''
    C = (1 / L^2) *
        [
        0, 0;
        0, GJₛ
    ]
    # 0th deriv terms: w, ψ
    D = K_f
    # 1st deriv terms: w', ψ'
    E = 1 / L * E_f

    b = -(B * q[7:8] + C * q[5:6] + D * q[1:2] + E * q[3:4])

    x = A \ b

    # --- Solution ---
    ∂q∂y = zeros(Float64, 8)
    ∂q∂y[1:6] = q[3:end]
    ∂q∂y[7:8] = x

    return ∂q∂y

end
end # end submodule

using .Steady # expose submodule to module

# module DynamicDiffEqns

# end # end submodule

# using .DynamicDiffEqns

end # end module