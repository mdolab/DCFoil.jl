"""
Computes the sectional properties of composite beams
"""
module BeamProperties

# --- Public functions ---
export compute_section_property

struct SectionProperty{T<:Float64}
    c::T # chord length
    t::T # thickness (only needed if not using an airfoil section)
    ab::T # dist from midchord to EA, +ve for EA aft
    ρₛ::T # density
    E₁::T # Young's modulus in-plane fiber longitudinal direction (x)
    E₂::T # Young's modulus in-plane fiber normal direction (y)
    G₁₂::T # In-plane Shear modulus
    ν₁₂::T # Poisson ratio
    θ::T # global fiber frame orientation
	airfoilCoords::Array{T, 2} # airfoil coordinates
end


function compute_section_property(section::SectionProperty, constitutive)
    """
    Orthotropic material uses classic laminate theory (CLT) for 
    composite cross section property computation.

    Outputs
    -------
    	EIₛ: scalar
    		bending stiffness OOP [N - m²]
    	EIₛIP: scalar
    		in-plane (IP) bending stiffness of the element [N - m²]
    	Kₛ: scalar
    		bend-twist coupling [N - m²]
    	GJₛ: scalar
    		torsion stiffness [N - m²]
    	Sₛ: scalar
    		warping resistance [N - m⁴]
    	EAₛ: scalar
               axial stiffness [N]
           Iₛ: scalar
               mass moment of inertia per unit length [kg-m²/m]
           mₛ: scalar
               mass per unit length [kg/m]

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

    # Mass properties
    mₛ = ρₛ * c * t # [kg/m]
    Iₛ = ρₛ * (c * t^3 / 12 + c^3 * t / 12) # [kg-m^2/m]

    # Stiffness properties
    EIₛ = 0.0
    EIₛIP = 0.0
    EAₛ = 0.0
    Kₛ = 0.0
    GJₛ = 0.0
    if (constitutive == "orthotropic") # CLT for a plate
        # Fiber frame
        divPoissonsRatio = 1 / (1 - ν₁₂ * ν₂₁)
        q₁₁ = E₁ * divPoissonsRatio
        q₂₂ = E₂ * divPoissonsRatio
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

        # Flexural stiffnesses D_ij for single layer laminate (M_i = D_ij k_i)
        d₁₁ = q₁₁ₚ / 12
        d₂₂ = q₂₂ₚ / 12
        d₁₂ = q₁₂ₚ / 12
        d₁₆ = q₁₆ₚ / 12
        d₂₆ = q₂₆ₚ / 12
        d₆₆ = q₆₆ₚ / 12

        # Weisshaar and Foist 1985 for a zero chordwise moment beam via composite plate theory (t^3 comes here for computational speedup)
        EIₛ = (d₁₁ - d₁₂^2 / d₂₂) * c * t^3
        Kₛ = 2 * (d₁₆ - d₁₂ * d₂₆ / d₂₂) * c * t^3
        GJₛ = 4 * (d₆₆ - d₂₆^2 / d₂₂) * c * t^3

        # TODO: make these parts more accurate later
        EIₛIP = E₁ * c^3 * t / 12 # in-plane EI for a rectangle
        EAₛ = E₁ * c * t # EA for a rectangle

    elseif (constitutive == "isotropic")
        EIₛ = E₁ * c * t^3 / 12 # EI for a rectangle
        EIₛIP = E₁ * c^3 * t / 12 # EI for a rectangle
        EAₛ = E₁ * c * t # EA for a rectangle
        GJₛ = G₁₂ * π * c^3 * t^3 / (c^2 + t^2) # GJ for an ellipse
        GJₛ = G₁₂ * c^3 * t * 0.333  # GJ for a rectangle
    elseif ("hoang-case")
        # TODO: call CLT function

    end
    Sₛ = EIₛ * ((0.5 * ab)^2 + (c^2 / 12.0))


    # # Bullshit factor to remove spurious modes
    # EIₛIP *= 1e2
    # EAₛ *= 1e2

    # --- Non-rectangular cross-section ---
    # This portion of the code corrects (or replaces) the above computations to consider some of the geometric effects
    # fA, fI , fJ = compute_airfoil_shape_corrections(airfoilCoords; method="xfoil", nChord=nChord)
    # EIₛ *= fI
    # EAₛ *= fA
    # GJₛ *= fJ

    # println("bend stiff: ", EIₛ)
    # println("BTC: ", Kₛ)
    # println("torsion stiff: ", GJₛ)
    # println("warp res: ", Sₛ)
    # println("ext. stiff: ", EAₛ)

    return EIₛ, EIₛIP, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ

end

function compute_airfoil_shape_corrections(airfoilCoords; method="xfoil", nChord=20)
    """
    In the case where one has an idea of the airfoil shape but not the 
    FE model, we can correct based on some airfoil structural theory (or see XFOIL BEND command)
    airfoilCoords: 
    	2D array of airfoil coordinates arranged CCW from TE like
    	[npts, 2]
    	with LE at origin
    Optional arguments:
    nChord - number of chordwise elements if method is "geometric"
    """
    # --- Determine airfoil properties ---
    # tau - thickness-to-chord ratio of the airfoil
    # c - chord
    # eps - camber ratio h/c where h = max {[Zu(x)+ Zℓ(x)]/2}
    # Get max thickness and camber
    npts = size(airfoilCoords, 1)
    airfoilUpper = airfoilCoords[1:Int(npts / 2), :]
    airfoilLower = airfoilCoords[Int(npts / 2)+1:end, :]
    tMax = maximum(airfoilUpper[:, 2] - airfoilLower[:, 2])
    chord = maximum(abs(airfoilCoords[:, 1]))
    tau = tMax / chord
    eps = maximum(0.5 * (airfoilUpper[:, 2] + airfoilLower[:, 2])) / chord
    As = chord * tMax # area of the section
    Is = 1 / 12 * chord * tMax^3 # 2nd area mom for rect
    Js = (chord * tMax^3) / 16 * (16 / 3 - 3.36 * tMax / chord * (1 - (tMax^4) / (12 * chord^4))) # torsion const for rect 4 % err (Roark's Formulas)
    # Default correction factors of 1.0
    fA = 1.0
    fI = 1.0
    fJ = 1.0
    if method == "xfoil"
        A, I = compute_simple_airfoil(chord, tau, eps)
        J = pi * (chord * tMax)^3 / (16 * (chord^2 + tMax^2)) # torsion constant approximated as an ellipse Roark
        fA = A / As
        fI = I / Is
        fJ = J / Js
    elseif method == "geometric"
        # Look at Eirikur's pygeo code
        # Spline upper and lower surfaces and order from LE to TE
        airfoilUpperS =
            airfoilLowerS =
            # --- Loop from LE to TE ---
            # First get area
                A = 0.0
        for iSect in 1:nChord
            A += (airfoilUpperS[iSect] - airfoilLowerS[iSect]) * dx
        end
        # Then get VCA
        zbar = 0.0
        for iSect in 1:nChord
            zbar += 0.5 * (airfoilUpperS[iSect]^2 - airfoilLowerS[iSect]^2) * dx
        end
        # Then compute second area moment
        Ixx = 0.0
        for iSect in 1:nChord
            Ixx += 1 / 3 * ((airfoilUpperS[iSect] - zbar)^3 - (airfoilLowerS[iSect] - zbar)^3) * dx
        end

        fA = A / As
        fI = Ixx / Is
    end

    return fA, fI, fJ
end

function compute_simple_airfoil(chord, tau, eps)
    # From the PDF
    KA = 0.60
    KI = 0.036

    # --- Compute section properties ---
    A = KA * chord^2 * tau
    I = KI * chord^4 * tau * (tau^2 + eps^2)

    return A, I
end

function compute_CLT_multilayer()
end

end # end of module
