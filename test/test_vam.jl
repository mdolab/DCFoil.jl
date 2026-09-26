# --- Julia 1.11---
"""
@File    :   test_vam.jl
@Time    :   2026/09/23
@Author  :   Galen Ng
@Desc    :   Tests for the VAM cross-sectional analysis (src/struct/VAM.jl).
             Every test returns the maximum over its checks of (error / tolerance), so the test
             passes when the returned value is <= 1. Yes/no checks contribute 0 or Inf.
             References: Hodges (2006) Nonlinear Composite Beam Theory Ch. 7 and closed forms.
"""

using LinearAlgebra

include("../src/struct/VAM.jl")
using .VAM

# ==============================================================================
#                         Helpers
# ==============================================================================
normalized_error(value, reference, tolerance) = abs(value / reference - 1) / tolerance
normalized_absolute_error(value, reference, tolerance) = abs(value - reference) / tolerance
passes(condition::Bool) = condition ? 0.0 : Inf

"""
Saint-Venant torsion factor β for a rectangle with side ratio short/long (Hodges eq. 7.67), so
that J = (short)³ (long) β / 3.
"""
function rectangle_torsion_factor(ratio)
    series = sum(tanh(n * pi / (2 * ratio)) / n^5 for n in 1:2:401)
    return 1 - 192 * ratio / pi^5 * series
end

"""
Inverse shear correction factors of an isotropic rectangle 2a x 2b (Hodges eq. 7.68), ρ = a / b.
"""
function rectangle_shear_factors(a, b, ν)
    ρ = a / b
    inv_c2 = 6 / 5 + (ν / (1 + ν))^2 * ρ^-4 * (1 / 5 - 18 / (ρ * pi^5) * sum(tanh(m * pi * ρ) / m^5 for m in 1:400))
    inv_c3 = 6 / 5 + (ν / (1 + ν))^2 * ρ^4 * (1 / 5 - 18ρ / pi^5 * sum(tanh(n * pi / ρ) / n^5 for n in 1:400))
    return 1 / inv_c2, 1 / inv_c3
end

"""
Exact shear stiffnesses of an isotropic ellipse with semi-axes a (x2) and b (x3) (Hodges eq. 7.61).
"""
function ellipse_shear_stiffnesses(a, b, ν, G)
    area = pi * a * b
    S2 = 3 * a^2 * (3a^2 + b^2) * (1 + ν)^2 * G * area / (2 * (b^4 * ν^2 + 5a^4 * (1 + ν)^2 + 2a^2 * b^2 * (1 + ν)^2))
    S3 = 3 * b^2 * (a^2 + 3b^2) * (1 + ν)^2 * G * area / (2 * (a^4 * ν^2 + 5b^4 * (1 + ν)^2 + 2a^2 * b^2 * (1 + ν)^2))
    return S2, S3
end

function rectangle_mesh(width, height, nChord, nThick; order=2, shift=(0.0, 0.0), cosine=false)
    nStations = order * nChord + 1
    x2 = cosine ? cosine_stations(-width / 2, width / 2, nStations) : collect(range(-width / 2, width / 2, length=nStations))
    x2 = x2 .+ shift[1]
    zUpper = fill(height / 2 + shift[2], nStations)
    zLower = fill(-height / 2 + shift[2], nStations)
    return generate_structured_quad_mesh(x2, zLower, zUpper; nThick=nThick, order=order)
end

function ellipse_mesh(a, b, nChord, nThick; order=2)
    x2 = cosine_stations(-a, a, order * nChord + 1)
    zUpper = [b * sqrt(max(0.0, 1 - (x / a)^2)) for x in x2]
    return generate_structured_quad_mesh(x2, -zUpper, zUpper; nThick=nThick, order=order)
end

"""
NACA four-digit thickness distribution with the open trailing edge used by generate_naca4dig.
"""
naca_thickness(x, toc) = 5 * toc * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x^2 + 0.2843 * x^3 - 0.1015 * x^4)

"""
NACA00xx mesh of chord c with the reference axis at midchord and x2 positive toward the leading edge.
"""
function naca_mesh(c, toc, nChord, nThick; order=2)
    x2 = cosine_stations(-c / 2, c / 2, order * nChord + 1)
    xFromLE = c / 2 .- x2
    zUpper = [c * naca_thickness(x / c, toc) for x in xFromLE]
    return generate_structured_quad_mesh(x2, -zUpper, zUpper; nThick=nThick, order=order)
end

# DCFoil "cfrp" constants from MaterialLibrary.jl
const CFRP = transversely_isotropic_material(117.8e9, 13.4e9, 3.9e9, 0.25, 1590.0)

"""
Beam constants of a thin plate c x t from the classical lamination theory formulas used in
compute_section_property (Weisshaar and Foist 1985).
"""
function clt_plate_constants(E1, E2, G12, ν12, c, t, θ)
    ν21 = E2 / E1 * ν12
    divPoisson = 1 / (1 - ν12 * ν21)
    q11 = E1 * divPoisson
    q22 = E2 * divPoisson
    q12 = q22 * ν12
    q66 = G12
    m, n = cos(θ), sin(θ)
    q11p = q11 * m^4 + q22 * n^4 + 2 * (q12 + 2q66) * m^2 * n^2
    q22p = q11 * n^4 + q22 * m^4 + 2 * (q12 + 2q66) * m^2 * n^2
    q12p = (q11 + q22 - 4q66) * m^2 * n^2 + q12 * (m^4 + n^4)
    q16p = m * n * (q11 * m^2 - q22 * n^2 - (q12 + 2q66) * (m^2 - n^2))
    q26p = m * n * (q11 * n^2 - q22 * m^2 + (q12 + 2q66) * (m^2 - n^2))
    q66p = (q11 + q22 - 2q12) * m^2 * n^2 + q66 * (m^2 - n^2)^2
    d11, d22, d12, d16, d26, d66 = q11p / 12, q22p / 12, q12p / 12, q16p / 12, q26p / 12, q66p / 12
    EI = (d11 - d12^2 / d22) * c * t^3
    K = 2 * (d16 - d12 * d26 / d22) * c * t^3
    GJ = 4 * (d66 - d26^2 / d22) * c * t^3
    return EI, K, GJ
end

# ==============================================================================
#                         Tests
# ==============================================================================
"""
Material stiffness rotation: isotropic invariance, the +45 degree shear-normal coupling sign,
and rotating back.
"""
function test_VAM_material_rotation()
    errors = Float64[]
    iso = isotropic_material(1.0, 0.3, 1.0)
    Diso = build_material_stiffness(iso)
    println("D_isotropic:")
    show(stdout, "text/plain", Diso)
    println()
    push!(errors, norm(rotate_material_stiffness(Diso, ply_rotation(0.7)) - Diso) / norm(Diso) / 1e-12)
    D = build_material_stiffness(CFRP)
    D45 = rotate_material_stiffness(D, ply_rotation(pi / 4))
    println("D")
    show(stdout, "text/plain", D)
    println()
    println("D45")
    show(stdout, "text/plain", D45)
    println()
    # At 45 degrees the beam-frame coupling C_1112 equals (C_1111 - C_2222) / 4 and is positive
    push!(errors, normalized_error(D45[1, 2], (D[1, 1] - D[4, 4]) / 4, 1e-12))
    push!(errors, passes(D45[1, 2] > 0))
    push!(errors, norm(rotate_material_stiffness(D45, ply_rotation(-pi / 4)) - D) / norm(D) / 1e-12)
    return maximum(errors)
end

"""
Isotropic ellipse and circle against the closed forms of Hodges eqs. (7.60) and (7.61).
"""
function test_VAM_ellipse()
    errors = Float64[]
    E = 1.0
    for (a, b, ν) in [(2.0, 1.0, 0.3), (2.0, 1.0, 0.0), (1.0, 1.0, 0.3), (1.0, 1.0, 0.0)]
        G = E / (2 * (1 + ν))
        mesh = ellipse_mesh(a, b, 40, 8)
        props = analyze_cross_section(mesh, uniform_layup(mesh, 1, 0.0), [isotropic_material(E, ν, 1.0)])
        S = props.stiffnessClassical
        area = pi * a * b
        push!(errors, normalized_error(S[1, 1], E * area, 1e-5))
        push!(errors, normalized_error(S[2, 2], G * pi * a^3 * b^3 / (a^2 + b^2), 1e-5))
        push!(errors, normalized_error(S[3, 3], E * pi / 4 * a * b^3, 1e-5))
        push!(errors, normalized_error(S[4, 4], E * pi / 4 * a^3 * b, 1e-5))
        push!(errors, maximum(abs.(S - Diagonal(S))) / maximum(abs.(S)) / 1e-10)
        S2, S3 = ellipse_shear_stiffnesses(a, b, ν, G)
        S6 = props.stiffnessTimoshenko
        push!(errors, normalized_error(S6[2, 2], S2, 1e-4))
        push!(errors, normalized_error(S6[3, 3], S3, 1e-4))
        push!(errors, norm(props.shearCenter) / a / 1e-10)
        push!(errors, norm(props.tensionCenter) / a / 1e-10)
        push!(errors, check_timoshenko_consistency(S6, S) / 1e-10)
        if a == b
            # Circle: shear factor 6 (1 + ν)^2 / (7 + 14 ν + 8 ν^2)
            factor = 6 * (1 + ν)^2 / (7 + 14ν + 8ν^2)
            push!(errors, normalized_error(S6[2, 2] / (G * area), factor, 1e-4))
        end
    end
    return maximum(errors)
end

"""
Isotropic rectangle 1 x 2 (x2 x x3): exact shear correction factors (Hodges eq. 7.68, book p. 229
gives c2 = 0.784442) and the Saint-Venant torsion constant (eq. 7.67).
"""
function test_VAM_rectangleShear()
    errors = Float64[]
    a, b = 0.5, 1.0
    G = 1e9
    for ν in [0.3, 0.0]
        E = 2G * (1 + ν)
        mesh = rectangle_mesh(2a, 2b, 10, 20)
        props = analyze_cross_section(mesh, uniform_layup(mesh, 1, 0.0), [isotropic_material(E, ν, 1.0)])
        S = props.stiffnessClassical
        S6 = props.stiffnessTimoshenko
        area = 4a * b
        c2, c3 = rectangle_shear_factors(a, b, ν)
        push!(errors, normalized_error(S6[2, 2] / (G * area), c2, 5e-5))
        push!(errors, normalized_error(S6[3, 3] / (G * area), c3, 5e-5))
        if ν == 0.3
            push!(errors, normalized_error(S6[2, 2] / (G * area), 0.784442, 5e-5))
        else
            push!(errors, normalized_error(S6[2, 2] / (G * area), 5 / 6, 5e-5))
            push!(errors, normalized_error(S6[3, 3] / (G * area), 5 / 6, 5e-5))
        end
        J = (2a)^3 * (2b) * rectangle_torsion_factor(a / b) / 3
        push!(errors, normalized_error(S[2, 2], G * J, 5e-5))
        push!(errors, normalized_error(S[1, 1], E * area, 1e-10))
        push!(errors, normalized_error(S[3, 3], E * 4a * b^3 / 3, 1e-10))
        push!(errors, normalized_error(S[4, 4], E * 4a^3 * b / 3, 1e-10))
    end
    return maximum(errors)
end

"""
Isotropic strip b = 0.953 in, h = 0.03 in, E = 1e7 psi, ν = 0.3 against VABS (Hodges Table 7.9)
and the analytical warping rigidity E h^3 b^3 / (144 (1 - ν^2)) (eq. 7.81).
"""
function test_VAM_isotropicStrip()
    errors = Float64[]
    b, h = 0.953, 0.03
    E, ν = 1e7, 0.3
    mesh = rectangle_mesh(b, h, 60, 2; cosine=true)
    props = analyze_cross_section(mesh, uniform_layup(mesh, 1, 0.0), [isotropic_material(E, ν, 1.0)])
    push!(errors, normalized_error(props.stiffnessClassical[2, 2], 32.3418, 5e-4))
    push!(errors, normalized_error(props.stiffnessVlasov[5, 5], 1.78635, 5e-4))
    push!(errors, normalized_error(props.stiffnessVlasov[5, 5], E * h^3 * b^3 / (144 * (1 - ν^2)), 3e-3))
    return maximum(errors)
end

"""
Orthotropic and anisotropic strips (Hodges Table 7.8 material) against VABS Tables 7.10 and 7.11.
"""
function test_VAM_compositeStrip()
    errors = Float64[]
    b, h = 0.953, 0.03
    material = Material3D(E1=20.59e6, E2=1.42e6, E3=1.42e6, G12=8.7e5, G13=8.7e5, G23=6.96e5,
        ν12=0.42, ν13=0.42, ν23=0.42, ρ=1.0)
    mesh = rectangle_mesh(b, h, 60, 2; cosine=true)
    for (θ, S22ref, S55ref, tol22, tol55) in [(0.0, 7.31562, 3.39251, 5e-4, 2e-4), (deg2rad(15), 7.96547, 1.82499, 2e-4, 5e-4)]
        props = analyze_cross_section(mesh, uniform_layup(mesh, 1, θ), [material])
        Sv = props.stiffnessVlasov
        effectiveTorsion = Sv[2, 2] - Sv[2, 3]^2 / Sv[3, 3]
        push!(errors, normalized_error(effectiveTorsion, S22ref, tol22))
        push!(errors, normalized_error(Sv[5, 5], S55ref, tol55))
    end
    return maximum(errors)
end

"""
Translation of the reference axis and a two-layer section: extension-bending coupling, invariance
of the condensed stiffnesses, shear and tension center locations, and material indexing.
"""
function test_VAM_translationAndLayers()
    errors = Float64[]
    a, b = 0.5, 1.0
    E, ν = 2.6e9, 0.3
    material = isotropic_material(E, ν, 1.0)
    mesh = rectangle_mesh(2a, 2b, 10, 20)
    base = analyze_cross_section(mesh, uniform_layup(mesh, 1, 0.0), [material])
    d = (0.3, 0.2)
    meshShifted = rectangle_mesh(2a, 2b, 10, 20; shift=d)
    shifted = analyze_cross_section(meshShifted, uniform_layup(meshShifted, 1, 0.0), [material])
    S = shifted.stiffnessClassical
    area = 4a * b
    push!(errors, normalized_error(S[1, 3], E * area * d[2], 1e-8))
    push!(errors, normalized_error(S[1, 4], -E * area * d[1], 1e-8))
    condensedBase = condense_axial_stiffness(base.stiffnessClassical)
    push!(errors, norm(condense_axial_stiffness(S) - condensedBase) / norm(condensedBase) / 1e-8)
    push!(errors, norm(shifted.stiffnessTimoshenko[2:3, 2:3] - base.stiffnessTimoshenko[2:3, 2:3]) / norm(base.stiffnessTimoshenko[2:3, 2:3]) / 1e-8)
    push!(errors, normalized_absolute_error(shifted.shearCenter[1], d[1], 1e-8 * b))
    push!(errors, normalized_absolute_error(shifted.shearCenter[2], d[2], 1e-8 * b))
    push!(errors, normalized_absolute_error(shifted.tensionCenter[1], d[1], 1e-8 * b))
    push!(errors, normalized_absolute_error(shifted.tensionCenter[2], d[2], 1e-8 * b))
    push!(errors, normalized_error(shifted.stiffnessVlasov[5, 5], base.stiffnessVlasov[5, 5], 1e-8))

    # Two isotropic layers of different modulus and density split at x3 = 0
    E1, E2 = 1.0e9, 3.0e9
    ρ1, ρ2 = 1.0, 2.0
    materials = [isotropic_material(E1, ν, ρ1), isotropic_material(E2, ν, ρ2)]
    nElem = size(mesh.elemConn, 1)
    elemMaterial = [sum(mesh.nodeCoords[mesh.elemConn[e, :], 2]) < 0 ? 1 : 2 for e in 1:nElem]
    layered = analyze_cross_section(mesh, SectionLayup(elemMaterial, zeros(nElem)), materials)
    SL = layered.stiffnessClassical
    A1 = 2a * b
    z1, z2 = -b / 2, b / 2
    I1 = 2a * b^3 / 12
    push!(errors, normalized_error(SL[1, 1], (E1 + E2) * A1, 1e-8))
    push!(errors, normalized_error(SL[1, 3], (E1 * z1 + E2 * z2) * A1, 1e-8))
    push!(errors, normalized_error(SL[3, 3], E1 * (I1 + A1 * z1^2) + E2 * (I1 + A1 * z2^2), 1e-8))
    push!(errors, normalized_absolute_error(layered.tensionCenter[2], (E1 * z1 + E2 * z2) / (E1 + E2), 1e-8 * b))
    push!(errors, normalized_absolute_error(layered.tensionCenter[1], 0.0, 1e-8 * b))
    push!(errors, normalized_absolute_error(layered.massCenter[2], (ρ1 * z1 + ρ2 * z2) / (ρ1 + ρ2), 1e-8 * b))
    push!(errors, normalized_error(layered.massPerLength, (ρ1 + ρ2) * A1, 1e-8))
    return maximum(errors)
end

"""
Thin CFRP plate (c / t = 200) against the classical lamination theory beam constants of
compute_section_property, including the sign of the bend-twist coupling K = -S̄23.
"""
function test_VAM_thinPlateCLT()
    errors = Float64[]
    c = 0.1
    t = c / 200
    mesh = rectangle_mesh(c, t, 60, 2; cosine=true)
    for θ in [0.0, deg2rad(15), deg2rad(30)]
        props = analyze_cross_section(mesh, uniform_layup(mesh, 1, θ), [CFRP])
        Sc = condense_axial_stiffness(props.stiffnessClassical) # [κ1, κ2, κ3]
        GJ, EI, K = Sc[1, 1], Sc[2, 2], -Sc[1, 2]
        EIclt, Kclt, GJclt = clt_plate_constants(CFRP.E1, CFRP.E2, CFRP.G12, CFRP.ν12, c, t, θ)
        if θ == 0
            push!(errors, normalized_error(EI, EIclt, 1e-4))
            push!(errors, abs(K) / sqrt(GJ * EI) / 1e-10)
            push!(errors, normalized_error(GJ, CFRP.G12 * c * t^3 * rectangle_torsion_factor(t / c) / 3, 5e-4))
        else
            push!(errors, normalized_error(EI, EIclt, 1e-2))
            push!(errors, normalized_error(K, Kclt, 1e-2)) # also fixes the sign
            push!(errors, normalized_error(GJ, GJclt, 1e-2))
        end
    end
    return maximum(errors)
end

"""
Solid NACA0009 in unidirectional CFRP, chord 0.1 m, reference axis at midchord.
References at zero fiber angle are exact quadratures of the thickness polynomial (uniaxial stress
is exact for fibers along the beam), the Prandtl bounds for torsion, and the thin-section shear
center estimate. At 15 degrees the thin-strip CLT integrals give the expected magnitudes.
"""
function test_VAM_NACA0009()
    errors = Float64[]
    c = 0.1
    toc = 0.09
    E1 = CFRP.E1
    mesh = naca_mesh(c, toc, 40, 8)
    props = analyze_cross_section(mesh, uniform_layup(mesh, 1, 0.0), [CFRP])
    S = props.stiffnessClassical
    # Exact quadrature references (area 0.0616575 c^2, centroid 0.4204355 c from the LE)
    areaRef = 6.165750e-4
    x2Centroid = c / 2 - 0.4204355 * c
    push!(errors, normalized_error(props.area, areaRef, 1e-5))
    push!(errors, normalized_absolute_error(props.centroid[1], x2Centroid, 1e-5 * c))
    push!(errors, normalized_absolute_error(props.centroid[2], 0.0, 1e-10 * c))
    push!(errors, normalized_error(S[1, 1], E1 * areaRef, 1e-5))
    push!(errors, normalized_error(S[3, 3], 338.4163, 1e-5))
    push!(errors, normalized_error(S[4, 4], 44694.37, 1e-5))
    push!(errors, normalized_error(S[1, 4], -E1 * areaRef * x2Centroid, 1e-5))
    condensed = condense_axial_stiffness(S)
    push!(errors, normalized_error(condensed[3, 3], 40096.36, 1e-5))
    # Torsion: displacement FE converges from above to a value between the Prandtl lower bound
    # G ∫ (t^3/3 - t^3 t'^2/4) dx and the thin-section estimate G ∫ t^3/3 dx
    GJ = S[2, 2]
    push!(errors, passes(1.0 <= GJ / 44.2444 <= 1.005))
    push!(errors, passes(GJ < 44.8157))
    couplings = [S[1, 2], S[1, 3], S[2, 3], S[2, 4], S[3, 4]]
    scales = [S[1, 1] * S[2, 2], S[1, 1] * S[3, 3], S[2, 2] * S[3, 3], S[2, 2] * S[4, 4], S[3, 3] * S[4, 4]]
    push!(errors, maximum(abs.(couplings) ./ sqrt.(scales)) / 1e-10)
    push!(errors, normalized_error(props.massPerLength, 0.9803543, 1e-5))
    push!(errors, normalized_error(props.momentsOfInertia[1], 4.567758e-6, 1e-5))
    push!(errors, normalized_error(props.momentsOfInertia[2], 6.032602e-4, 1e-5))
    push!(errors, normalized_absolute_error(props.tensionCenter[1], x2Centroid, 1e-6 * c))
    push!(errors, normalized_absolute_error(props.massCenter[1], x2Centroid, 1e-6 * c))
    shearCenterFromLE = (c / 2 - props.shearCenter[1]) / c
    push!(errors, passes(0.34 <= shearCenterFromLE <= 0.40))
    push!(errors, passes(props.shearCenter[1] > props.centroid[1])) # forward of the centroid
    push!(errors, normalized_absolute_error(props.shearCenter[2], 0.0, 1e-8 * c))
    push!(errors, check_timoshenko_consistency(props.stiffnessTimoshenko, S) / 1e-10)

    # Fiber angle: antisymmetric bend-twist coupling, positive K (nose-down under upward bending)
    # and magnitudes near the thin-strip CLT integrals EI2 = 278.9, K = 138.4, GJ = 122.3 N m^2
    coarse = naca_mesh(c, toc, 20, 4)
    plus = condense_axial_stiffness(analyze_cross_section(coarse, uniform_layup(coarse, 1, deg2rad(15)), [CFRP]).stiffnessClassical)
    minus = condense_axial_stiffness(analyze_cross_section(coarse, uniform_layup(coarse, 1, deg2rad(-15)), [CFRP]).stiffnessClassical)
    push!(errors, normalized_error(plus[1, 2], -minus[1, 2], 1e-10))
    push!(errors, norm(plus - Diagonal(plus) - (-(minus - Diagonal(minus)))) / norm(plus) / 1e-10)
    K = -plus[1, 2]
    push!(errors, passes(K > 0))
    push!(errors, normalized_error(plus[2, 2], 278.9, 5e-2))
    push!(errors, normalized_error(K, 138.4, 5e-2))
    push!(errors, normalized_error(plus[1, 1], 122.3, 5e-2))
    # Mesh convergence of the coarse mesh used above
    coarseZero = analyze_cross_section(coarse, uniform_layup(coarse, 1, 0.0), [CFRP]).stiffnessClassical
    push!(errors, norm(coarseZero - S) / norm(S) / 1e-5)
    return maximum(errors)
end
