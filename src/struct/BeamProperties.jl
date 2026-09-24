"""
@File    :   BeamProperties.jl
@Time    :   2024/01/30
@Author  :   Galen Ng
@Desc    :   Module for computing sectional properties of composite beams or reading from FEM file
"""

using Zygote

# Variational asymptotic method cross-sectional analysis (Hodges 2006). The module lands in the
# namespace that includes this file (FEMMethods and DCFoil), so call it as VAM.<function>.
include("VAM.jl")


struct SectionProperty{TF,TC,T}
    c::TC # chord length
    t::T # thickness (only needed if not using an airfoil section)
    ab::TF # dist from midchord to EA, +ve for EA aft
    ρₛ::TF # density
    E₁::TF # Young's modulus in-plane fiber longitudinal direction (x)
    E₂::TF # Young's modulus in-plane fiber normal direction (y)
    G₁₂::TF # In-plane Shear modulus
    ν₁₂::TF # Poisson ratio
    theta_f::RealOrComplex # global fiber frame orientation (this gets complex-stepped)
    airfoilCoords::Matrix{TF} # airfoil coordinates
end

function compute_section_property(section::SectionProperty, constitutive::String; solverOptions)
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
    theta_f = section.theta_f

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
        m = cos(theta_f)
        n = sin(theta_f)
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

        # GJₛ = G₁₂ * π * c^3 * t^3 / (c^2 + t^2) # GJ for an ellipse
        GJₛ = G₁₂ * c^3 * t * 0.333  # GJ for a rectangle

    elseif ("hoang-case")
        # TODO: call CLT function

    end

    Sₛ = EIₛ * ((0.5 * ab)^2 + (c^2 / 12.0)) # warping resistance for a rectangle


    # # Bullshit factor to remove spurious modes
    # EIₛIP *= 1e2
    # EAₛ *= 1e2

    # --- Non-rectangular cross-section ---
    if haskey(solverOptions, "correct_xsect") && solverOptions["correct_xsect"]
        # This portion of the code corrects (or replaces) the above computations to consider some of the geometric effects
        fA, fI, fJ = compute_airfoil_shape_corrections(section.airfoilCoords, section.t, section.c, section.t / section.c, 0.0; method="xfoil", nChord=20)
        EIₛ *= fI
        EAₛ *= fA
        GJₛ *= fJ
        # println("Airfoil correction factors: $(fA), $(fI), $(fJ)")
        # println("bend stiff: ", EIₛ)
        # println("BTC: ", Kₛ)
        # println("torsion stiff: ", GJₛ)
        # println("warp res: ", Sₛ)
        # println("ext. stiff: ", EAₛ)
    end


    return EIₛ, EIₛIP, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ

end

function generate_section_mesh(section::SectionProperty; nChord::Int=40, nThick::Int=6, order::Int=2)
    """
    Structured quadrilateral mesh of the solid airfoil section for the VAM analysis.

    The unit-chord airfoil coordinates in section.airfoilCoords (counterclockwise from the TE, as
    from generate_naca4dig) are scaled by the chord. The reference axis is the elastic axis at
    x_EA = c/2 + ab from the LE, and the beam frame has x2 pointing toward the LE and x3 up.
    section.t is not used; the thickness comes from the airfoil coordinates.
    """
    coords = section.airfoilCoords
    xMin, xMax = extrema(coords[:, 1])
    if !(xMax - xMin > 0)
        error("Airfoil coordinates have no chordwise extent; compute_beam's placeholder zeros(20, 2) cannot be used for VAM")
    end
    c = section.c
    xEA = c / 2 + section.ab
    xStations = VAM.cosine_stations(xMin, xMax, order * nChord + 1)
    zLower, zUpper = VAM.interpolate_airfoil_surfaces(coords, xStations)
    # x2 = xEA - x decreases along the airfoil, so reverse to ascending x2 for a counterclockwise mesh
    x2Stations = reverse(xEA .- c .* xStations)
    mesh = VAM.generate_structured_quad_mesh(x2Stations, reverse(c .* zLower), reverse(c .* zUpper); nThick=nThick, order=order)
    layup = VAM.uniform_layup(mesh, 1, section.theta_f)
    return mesh, layup
end

function convert_to_beam_constants(properties::VAM.CrossSectionProperties, ab)
    """
    Map the VAM cross-sectional properties onto the eight constants of the COMP2 beam element.

    The axial strain is condensed out (F1 = 0) because the element has no extension-bending
    coupling, so the bending stiffnesses are those about the tension center. The bend-twist
    coupling is K = -S̄23 because the element curvature is w'' while Hodges' κ2 = -w''.
    The warping stiffness follows the element convention S = Ŝ55 + EI ab², with Ŝ55 about the
    shear center (see docs.tex, structural model).

    Returns EIₛ, EIₛIP, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ in the order of compute_section_property.
    """
    stiffness = properties.stiffnessClassical
    condensed = VAM.condense_axial_stiffness(stiffness) # [κ1, κ2, κ3]
    EAₛ = stiffness[1, 1]
    GJₛ = condensed[1, 1]
    EIₛ = condensed[2, 2]
    EIₛIP = condensed[3, 3]
    Kₛ = -condensed[1, 2]
    Sₛ = properties.stiffnessVlasov[5, 5] + EIₛ * ab^2
    mₛ = properties.massPerLength
    Iₛ = properties.momentsOfInertia[1] + properties.momentsOfInertia[2]
    return EIₛ, EIₛIP, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ
end

function analyze_sectionVAM_properties(section::SectionProperty; nChord::Int=40, nThick::Int=6, order::Int=2, ν23=0.4)
    """
    Full VAM cross-sectional analysis of the airfoil section (Hodges 2006, Ch. 4): classical 4x4,
    generalized Timoshenko 6x6, generalized Vlasov 5x5, mass matrix, shear and tension centers.
    The material is transversely isotropic with the 2-3 Poisson ratio ν23 (not in MaterialLibrary).
    """
    mesh, layup = generate_section_mesh(section; nChord=nChord, nThick=nThick, order=order)
    material = VAM.transversely_isotropic_material(section.E₁, section.E₂, section.G₁₂, section.ν₁₂, section.ρₛ; ν23=ν23)
    return VAM.analyze_cross_section(mesh, layup, [material])
end

function analyze_sectionVAM(section::SectionProperty; nChord::Int=40, nThick::Int=6, order::Int=2, ν23=0.4)
    """
    Beam constants of the airfoil section from the variational asymptotic method, in the order of
    compute_section_property: EIₛ, EIₛIP, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ.
    """
    properties = analyze_sectionVAM_properties(section; nChord=nChord, nThick=nThick, order=order, ν23=ν23)
    return convert_to_beam_constants(properties, section.ab)
end

function compute_beam(nNodes::Int64,
    chord, t, ab, ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f,
    constitutive::String; solverOptions=nothing
)
    EIₛ = zeros(RealOrComplex, nNodes)
    Kₛ = zeros(RealOrComplex, nNodes)
    GJₛ = zeros(RealOrComplex, nNodes)
    Sₛ = zeros(RealOrComplex, nNodes)
    Iₛ = zeros(RealOrComplex, nNodes)
    mₛ = zeros(RealOrComplex, nNodes)
    # --- Loop over the span ---
    EI_z = Zygote.Buffer(EIₛ)
    EIIP_z = Zygote.Buffer(EIₛ)
    EA_z = Zygote.Buffer(EIₛ)
    K_z = Zygote.Buffer(Kₛ)
    GJ_z = Zygote.Buffer(GJₛ)
    S_z = Zygote.Buffer(Sₛ)
    I_z = Zygote.Buffer(Iₛ)
    m_z = Zygote.Buffer(mₛ)
    EI_z[:] = EIₛ
    EIIP_z[:] = EIₛ
    EA_z[:] = EIₛ
    K_z[:] = Kₛ
    GJ_z[:] = GJₛ
    S_z[:] = Sₛ
    I_z[:] = Iₛ
    m_z[:] = mₛ

    for ii in 1:nNodes
        section = SectionProperty(chord[ii], t[ii], ab[ii], ρₛ, E₁, E₂, G₁₂, ν₁₂, theta_f, zeros(20, 2))

        # TODO: should probably redo this to be element-based, not node-based
        EI, EIIP, K, GJ, S, EA, I, m = compute_section_property(section, constitutive; solverOptions=solverOptions)
        EI_z[ii] = EI
        EIIP_z[ii] = EIIP
        K_z[ii] = K
        GJ_z[ii] = GJ
        S_z[ii] = S
        EA_z[ii] = EA
        I_z[ii] = I
        m_z[ii] = m
    end

    EIₛ = copy(EI_z)
    EIIPₛ = copy(EIIP_z)
    EAₛ = copy(EA_z)
    Kₛ = copy(K_z)
    GJₛ = copy(GJ_z)
    Sₛ = copy(S_z)
    Iₛ = copy(I_z)
    mₛ = copy(m_z)

    return EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ
end

function compute_airfoil_shape_corrections(airfoilCoords::Matrix, tMax, chord, τ, ε; method="xfoil", nChord=20)
    """
    In the case where one has an idea of the airfoil shape but not the 
    FE model, we can correct based on some airfoil structural theory (or see XFOIL BEND command)

    airfoilCoords: 
    	2D array of airfoil coordinates arranged CCW from TE like
    	[npts, 2]
    	with LE at origin
    τ:
    	thickness-to-chord ratio of the airfoil based on max thickness
    ε:
        camber ratio h/c where h = max {[Zu(x)+ Zℓ(x)]/2}
    Optional arguments:
    nChord - number of chordwise elements if method is "geometric"
    """
    As = chord * tMax # area of the section
    Is = 1 / 12 * chord * tMax^3 # 2nd area mom for rect
    Js = (chord * tMax^3) / 16 * (16 / 3 - 3.36 * tMax / chord * (1 - (tMax^4) / (12 * chord^4))) # torsion const for rect 4 % err (Roark's Formulas)
    # Default correction factors of 1.0
    fA = 1.0
    fI = 1.0
    fJ = 1.0
    if method == "xfoil"
        A, I = compute_simple_airfoil(chord, τ, ε)
        J = pi * (chord * tMax)^3 / (16 * (chord^2 + tMax^2)) # torsion constant approximated as an ellipse Roark
        fA = A / As
        fI = I / Is
        fJ = J / Js
    elseif method == "geometric"
        # --- Determine airfoil properties ---
        # tau - thickness-to-chord ratio of the airfoil
        # c - chord
        # eps - camber ratio h/c where h = max {[Zu(x)+ Zℓ(x)]/2}
        # Get max thickness and camber
        npts = size(airfoilCoords, 1)
        airfoilUpper = airfoilCoords[1:Int(npts / 2), :]
        airfoilLower = airfoilCoords[Int(npts / 2)+1:end, :]
        tMax = maximum(airfoilUpper[:, 2] - airfoilLower[:, 2])
        chord = maximum(abs.(airfoilCoords[:, 1]))
        tau = tMax / chord
        eps = maximum(0.5 * (airfoilUpper[:, 2] + airfoilLower[:, 2])) / chord
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
    """
    Compute the area and second moment of inertia of a simple airfoil
    From the PDF https://ocw.mit.edu/courses/16-01-unified-engineering-i-ii-iii-iv-fall-2005-spring-2006/pages/systems-labs-06/
    """
    KA = 0.60
    KI = 0.036

    # --- Compute section properties ---
    A = KA * chord^2 * tau
    I = KI * chord^4 * tau * (tau^2 + eps^2)

    return A, I
end

function compute_CLT_multilayer()
    # TODO: 
end

