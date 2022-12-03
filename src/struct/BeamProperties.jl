"""
Computes the sectional properties of composite beams
"""
module StructProp

# --- Public functions ---
export compute_section_property

struct section_property{T<:Float64}

    """
    Inputs:
    c: chord length
    t: thickness
    ab: 
    ρₛ: density
    E₁: Young's modulus in x direction
    E₂: Young's modulus in y direction
    G₁₂: In-plane Shear modulus 
    ν₁₂: Poisson ratio
    θ: global frame orientation
    """

    c::T
    t::T
    ab::T
    ρₛ::T
    E₁::T
    E₂::T
    G₁₂::T
    ν₁₂::T
    θ::T

end


function compute_section_property(section::section_property)

    """
    Classic laminate theory (CLT) for composite cross section property computation.
    returns:
        EIₛ: scalar, bending stiffness.
        Kₛ: scalar, bend-twist coupling
        GJₛ: scalar, torsion stiffness
        Sₛ: scalar, warping resistance

    NOTE:
    Axes convention is 1 along fiber, 2 transverse in-plane, 3 transverse out-of-plane

    This follows the formulation in 
    'Steady and dynamic hydroelastic behavior of composite lifting surfaces' 
    by Deniz Tolga Akcabaya & Yin Lu Young
    """

    c = section.c
    t = section.t
    ab = section.ab
    ρₛ = section.ρₛ
    E₁ = section.E₁
    E₂ = section.E₂
    G₁₂ = section.G₁₂
    ν₁₂ = section.ν₁₂
    θ = section.θ

    # Compute nu_21 by E2 * nu12 = E1 * nu12
    ν₂₁ = (E₂ / E₁) * ν₁₂

    # Fiber frame
    q₁₁ = E₁ / (1 - ν₁₂ * ν₂₁)
    q₂₂ = E₂ / (1 - ν₁₂ * ν₂₁)
    q₁₂ = q₂₂ * ν₁₂
    q₆₆ = G₁₂

    # Convert to physical frame
    m = cos(θ)
    n = sin(θ)
    q₁₁ₚ = q₁₁ * m^4 + q₂₂ * n^4 + 2 * (q₁₂ + 2 * q₆₆) * m^2 * n^2
    q₂₂ₚ = q₁₁ * n^4 + q₂₂ * m^4 + 2 * (q₁₂ + 2 * q₆₆) * m^2 * n^2
    q₁₂ₚ = (q₁₁ + q₂₂ - 4 * q₆₆) * m^2 * n^2 + q₁₂ * (m^4 + n^4)
    q₁₆ₚ = m * n * (q₁₁ * m^2 - q₂₂ * n^2 - (q₁₂ + 2 * q₆₆) * (m^2 - n^2))
    q₂₆ₚ = m * n * (q₁₁ * n^2 - q₂₂ * m^2 + (q₁₂ + 2 * q₆₆) * (m^2 - n^2))
    q₆₆ₚ = (q₁₁ + q₂₂ - 2 * q₁₂) * m^2 * n^2 + q₆₆ * (m^2 - n^2)^2

    # Flexural stiffnesses D_ij for single layer laminate
    d₁₁ = q₁₁ₚ / 12
    d₂₂ = q₂₂ₚ / 12
    d₁₂ = q₁₂ₚ / 12
    d₁₆ = q₁₆ₚ / 12
    d₂₆ = q₂₆ₚ / 12
    d₆₆ = q₆₆ₚ / 12

    mₛ = ρₛ * c * t
    Iₛ = ρₛ * (c * t^3 / 12 + c^3 * t / 12)

    EIₛ = (d₁₁ - d₁₂^2 / d₂₂) * c * t^3
    Kₛ = 2 * (d₁₆ - d₁₂ * d₂₆ / d₂₂) * c * t^3
    GJₛ = 4 * (d₆₆ - d₂₆^2 / d₂₂) * c * t^3
    Sₛ = EIₛ * ((0.5 * ab)^2 + (c^2 / 12.0))

    # if (Kₛ < 1e-5)
    #     Kₛ = Kₛ + 1e-5
    # end

    return EIₛ, Kₛ, GJₛ, Sₛ, Iₛ, mₛ

end

end # end of module