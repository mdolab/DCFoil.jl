# --- Julia 1.11---
"""
@File    :   test_strainbeam.jl
@Time    :   2026/09/24
@Author  :   Galen Ng
@Desc    :   Tests for the geometrically exact strain-based beam (src/struct/StrainBeam.jl) fed by
             the VAM section properties. Every test returns the maximum over its checks of
             (error / tolerance), so the test passes when the returned value is <= 1.
             References: Hodges (2006) Ch. 5, Yu et al. (2005), Bisshopp and Drucker (1945),
             GXBeam.jl conventions (McDonnell and Ning 2022).
"""

using LinearAlgebra
using StaticArrays

# test_vam.jl provides Main.VAM, CFRP, naca_mesh, rectangle_mesh and the error helpers
if !isdefined(Main, :VAM)
    include("test_vam.jl")
end
include("../src/struct/StrainBeam.jl")
using .StrainBeam

const HAS_PLOTS = try
    @eval using Plots
    true
catch err
    @warn "Plots could not be loaded; StrainBeam figures are skipped" exception = (err, catch_backtrace())
    false
end

# ==============================================================================
#                         Section builders
# ==============================================================================
"""
Isotropic-like uncoupled section: extension EA, shear GA, torsion GJ, bending EI2 (flap, about x2) and
EI3 (chordwise, about x3), mass per length μ with inertias i2, i3.
"""
function isotropic_section(; EA=1e7, GA=1e7, GJ=1e5, EI2=1e5, EI3=1e5, μ=1.0, i2=1e-3, i3=1e-3)
    S = Matrix(Diagonal([EA, GA, GA, GJ, EI2, EI3]))
    M = Matrix(Diagonal([μ, μ, μ, i2 + i3, i2, i3]))
    return BeamSection(S, M)
end

"""
Carbon-fiber NACA0009 section of chord 0.1 m at fiber angle θf, exactly as test_VAM_NACA0009 builds
it (reference axis at midchord, x2 toward the leading edge). Returns the BeamSection (with the Vlasov
warping rigidity when `warping` is `VlasovWarping()`) and the VAM properties.
"""
function naca0009_section(θf; nChord=20, nThick=4, warping=StrainBeam.NoWarping())
    c = 0.1
    mesh = naca_mesh(c, 0.09, nChord, nThick)
    properties = analyze_cross_section(mesh, uniform_layup(mesh, 1, θf), [CFRP])
    return BeamSection(properties, warping), properties
end

"""
Isotropic strip b = 0.953 in, h = 0.03 in, E = 1e7 psi, ν = 0.3 of Hodges (2006) Table 7.9
(GJ = 32.3418 lb in², Ŝ55 = 1.78635 lb in⁴), with the Vlasov warping rigidity. Units lb, in.
"""
function vlasov_strip_section(; warping=StrainBeam.VlasovWarping())
    b, h = 0.953, 0.03
    E, ν = 1e7, 0.3
    mesh = rectangle_mesh(b, h, 60, 2; cosine=true)
    properties = analyze_cross_section(mesh, uniform_layup(mesh, 1, 0.0), [isotropic_material(E, ν, 1.0)])
    return BeamSection(properties, warping), properties
end

"""
Axis-angle rotation tensor (global components) rotating vectors by φ about the unit axis n.
"""
function axis_angle_rotation(φ, n)
    ñ = StrainBeam.tilde(SVector{3}(n))
    return I + sin(φ) * ñ + (1 - cos(φ)) * ñ^2
end

"""
Cantilever model along +y with a clamped root and one tip condition.
"""
function cantilever_model(section, L, nElem, tipCondition; direction=[0.0, 1.0, 0.0], sectionNormal=nothing, gravity=zeros(3))
    assembly = make_cantilever_assembly(L, nElem, section; direction=direction, sectionNormal=sectionNormal)
    conditions = Dict(1 => clamped(), nElem + 1 => tipCondition)
    return StaticModel(assembly, conditions; gravity=gravity)
end

# ==============================================================================
#                         Tests
# ==============================================================================
"""
Wiener-Milenkovic rotation algebra: C(θ) is the transpose of the axis-angle tensor with
φ = 4 atan(|θ|/4), Q Q⁻¹ = I, parameters are recovered from the matrix, composition matches the matrix
product, the rescaled representation of a rotation beyond 180 degrees is the same rotation, and Q maps
the parameter rate to the material angular velocity axial(C Ċᵀ).
"""
function test_StrainBeam_rotations()
    errors = Float64[]
    for θ in [SVector(0.3, -0.2, 0.5), SVector(1e-3, 2e-3, -1e-3), SVector(2.5, 1.0, -1.5)]
        θnorm = norm(θ)
        φ = 4 * atan(θnorm / 4)
        R = axis_angle_rotation(φ, θ / θnorm)
        C = wiener_milenkovic_rotation(θ)
        push!(errors, norm(C - transpose(R)) / 1e-12)
        push!(errors, norm(rotation_tangent(θ) * rotation_tangent_inverse(θ) - I) / 1e-12)
        push!(errors, norm(rotation_parameters(C) - θ) / 1e-10)
        # Q maps a parameter rate not parallel to θ to the material angular velocity axial(C Ċᵀ),
        # which is what the compatibility residual assumes (κ_B = Cabᵀ Q θ′)
        δ = SVector(0.4, 0.7, -0.2)
        δ = δ - dot(δ, θ) * θ / dot(θ, θ)
        h = 1e-6
        Ċ = (wiener_milenkovic_rotation(θ + h * δ) - wiener_milenkovic_rotation(θ - h * δ)) / (2h)
        W = C * transpose(Ċ)
        ωMaterial = SVector(W[3, 2], W[1, 3], W[2, 1])
        push!(errors, norm(ωMaterial - rotation_tangent(θ) * δ) / norm(δ) / 1e-6)
    end
    # 90 degrees about e1 followed by 90 degrees about e2 (4 tan(φ/4) with φ = π/2)
    θ1 = 4 * tan(pi / 8) * SVector(1.0, 0.0, 0.0)
    θ2 = 4 * tan(pi / 8) * SVector(0.0, 1.0, 0.0)
    θ12 = compose_rotation_parameters(θ1, θ2)
    push!(errors, norm(wiener_milenkovic_rotation(θ12) - wiener_milenkovic_rotation(θ1) * wiener_milenkovic_rotation(θ2)) / 1e-12)
    push!(errors, norm(transpose(wiener_milenkovic_rotation(θ12)) - axis_angle_rotation(pi / 2, [0, 1, 0]) * axis_angle_rotation(pi / 2, [1, 0, 0])) / 1e-12)
    # Beyond 180 degrees: |θ| = 5 is the rotation by φ = 4 atan(5/4) about n, and the rescaled
    # parameters (|c*| = 16/5 about -n) are the same rotation
    θbig = SVector(0.0, 0.0, 5.0)
    Rbig = axis_angle_rotation(4 * atan(5 / 4), [0, 0, 1])
    push!(errors, norm(wiener_milenkovic_rotation(θbig) - transpose(Rbig)) / 1e-12)
    push!(errors, norm(wiener_milenkovic_rotation(rescale_rotation_parameters(θbig)) - transpose(Rbig)) / 1e-12)
    push!(errors, normalized_error(norm(rescale_rotation_parameters(θbig)), 16 / 5, 1e-12))
    # Twist angle: pure rotation about the beam axis is recovered exactly, bending alone gives zero
    push!(errors, normalized_absolute_error(twist_angle(axis_angle_rotation(0.3, [1, 0, 0])), 0.3, 1e-12))
    push!(errors, normalized_absolute_error(twist_angle(axis_angle_rotation(0.7, [0, 1, 0])), 0.0, 1e-12))
    return maximum(errors)
end

"""
ForwardDiff Jacobian against central finite differences at the undeformed state and at a deformed
state of the CFRP NACA0009 cantilever (anisotropic section with all couplings active).
"""
function test_StrainBeam_jacobian()
    errors = Float64[]
    section, _ = naca0009_section(deg2rad(15))
    L = 1.0
    nElem = 6
    model = cantilever_model(section, L, nElem, dead_load([0.0, 0.0, 40.0], [0.0, 0.0, 0.0]))
    x0 = zeros(model.indices.nStates)
    for x in [x0, solve_static(model; nLoadSteps=4).x]
        Jad = compute_jacobian(x, model, 1.0; mode="FAD")
        Jfd = compute_jacobian(x, model, 1.0; mode="FiDi")
        push!(errors, norm(Jad - Jfd) / norm(Jfd) / 1e-6)
    end
    # Same with the Vlasov warping unknowns present
    sectionVlasov, _ = naca0009_section(deg2rad(15); warping=StrainBeam.VlasovWarping())
    modelVlasov = cantilever_model(sectionVlasov, L, nElem, dead_load([0.0, 0.0, 40.0], [0.0, 2.0, 0.0]))
    xV = solve_static(modelVlasov; nLoadSteps=4).x
    push!(errors, norm(compute_jacobian(xV, modelVlasov, 1.0; mode="FAD") - compute_jacobian(xV, modelVlasov, 1.0; mode="FiDi")) /
                  norm(compute_jacobian(xV, modelVlasov, 1.0; mode="FiDi")) / 1e-6)
    return maximum(errors)
end

"""
Cantilever rolled up by a tip moment M = λ π EI / L about global z: circular arc of radius ρ = EI/M
with tip position (-ρ(1 - cos θ), ρ sin θ), θ = M L / EI (Hodges 2006 Ch. 5). λ = 1.6 rotates the tip
by 288 degrees; the full circle (λ = 2) is the 360 degree singularity of the parametrization.
"""
function test_StrainBeam_tipMoment()
    errors = Float64[]
    EI = 1e5
    L = 10.0
    nElem = 32
    section = isotropic_section(EI2=EI, EI3=EI)
    for λ in [0.4, 0.8, 1.2, 1.6]
        Mtip = λ * pi * EI / L
        model = cantilever_model(section, L, nElem, dead_load([0.0, 0.0, 0.0], [0.0, 0.0, Mtip]))
        solution = solve_static(model; nLoadSteps=8, tol=1e-10)
        θ = Mtip * L / EI
        ρ = EI / Mtip
        reference = [-ρ * (1 - cos(θ)), ρ * sin(θ) - L, 0.0]
        push!(errors, norm(solution.displacements[end, :] - reference) / L / 2e-3)
        push!(errors, normalized_error(solution.curvatures[end, 3], Mtip / EI, 1e-10))
        push!(errors, normalized_absolute_error(solution.twist[end], 0.0, 1e-8))
        push!(errors, passes(all(solution.iterations .<= 8)))
    end
    return maximum(errors)
end

"""
Mesh convergence of the tip-moment case at λ = 1.2: the midpoint-rule element is second order.
"""
function test_StrainBeam_convergence()
    errors = Float64[]
    EI = 1e5
    L = 10.0
    Mtip = 1.2 * pi * EI / L
    θ = Mtip * L / EI
    ρ = EI / Mtip
    reference = [-ρ * (1 - cos(θ)), ρ * sin(θ) - L, 0.0]
    section = isotropic_section(EI2=EI, EI3=EI)
    counts = [4, 8, 16, 32]
    tipErrors = Float64[]
    for nElem in counts
        model = cantilever_model(section, L, nElem, dead_load([0.0, 0.0, 0.0], [0.0, 0.0, Mtip]))
        solution = solve_static(model; nLoadSteps=8, tol=1e-11)
        push!(tipErrors, norm(solution.displacements[end, :] - reference))
    end
    slope = (log(tipErrors[end]) - log(tipErrors[1])) / (log(counts[end]) - log(counts[1]))
    println("StrainBeam tip-moment convergence: errors = $(tipErrors), slope = $(slope)")
    push!(errors, passes(slope <= -1.8))
    push!(errors, passes(all(diff(tipErrors) .< 0)))
    return maximum(errors)
end

"""
Frame handling: rigidly rotating the whole problem (beam direction, section normal, load and a tip
torque) leaves the deformed-frame loads, strains and twist unchanged and rotates the displacements,
for a section with unequal bending stiffnesses. The default minimal-rotation frame is checked to
reproduce the DCFoil convention (e2 toward -x, e3 = +z for a beam along +y).
"""
function test_StrainBeam_frameInvariance()
    errors = Float64[]
    L = 5.0
    nElem = 12
    force = [30.0, 0.0, 200.0]
    torque = [0.0, 400.0, 0.0]
    section = isotropic_section(EI2=1e5, EI3=4e5, GJ=2e5)
    referenceModel = cantilever_model(section, L, nElem, dead_load(force, torque))
    push!(errors, norm(referenceModel.assembly.elementFrames[1] - [0 -1 0; 1 0 0; 0 0 1]) / 1e-14)
    reference = solve_static(referenceModel; nLoadSteps=5)
    push!(errors, passes(abs(reference.twist[end]) > 1e-3)) # the torque makes the case genuinely 3D
    for R in [axis_angle_rotation(0.9, [1, 0, 0]), axis_angle_rotation(1.1, [0.3, -0.5, 0.8] / norm([0.3, -0.5, 0.8]))]
        rotatedModel = cantilever_model(section, L, nElem, dead_load(R * force, R * torque); direction=R * [0.0, 1.0, 0.0], sectionNormal=R * [0.0, 0.0, 1.0])
        push!(errors, norm(rotatedModel.assembly.elementFrames[1] - R * referenceModel.assembly.elementFrames[1]) / 1e-12)
        rotated = solve_static(rotatedModel; nLoadSteps=5)
        push!(errors, norm(rotated.forces - reference.forces) / norm(reference.forces) / 1e-8)
        push!(errors, norm(rotated.moments - reference.moments) / norm(reference.moments) / 1e-8)
        push!(errors, norm(rotated.curvatures - reference.curvatures) / norm(reference.curvatures) / 1e-8)
        push!(errors, norm(rotated.displacements - reference.displacements * transpose(R)) / norm(reference.displacements) / 1e-8)
        push!(errors, norm(rotated.twist - reference.twist) / norm(reference.twist) / 1e-8)
    end
    return maximum(errors)
end

# ==============================================================================
#                         CFRP NACA0009 cantilever: spanwise results
# ==============================================================================
using DelimitedFiles

"""
Write a matrix with a header row as CSV.
"""
function write_csv(path, header, data)
    open(path, "w") do io
        writedlm(io, permutedims(header), ',')
        writedlm(io, data, ',')
    end
    return path
end

"""
Spanwise arrays of a solution in the beam frame: nodal (span coordinate s, u1 axial, u2 chordwise
toward the LE, u3 flap, twist, twist rate) and element-midpoint (s, F1..F3, M1..M3, γ11, 2γ12, 2γ13,
κ1..κ3, bimoment, Saint-Venant torque) tables.
"""
function spanwise_tables(solution, model)
    assembly = model.assembly
    Cab = assembly.elementFrames[1]
    s = assembly.nodeCoords * (Cab * [1.0, 0.0, 0.0])
    beamDisplacements = solution.displacements * Cab # rows u_bᵀ = u_aᵀ Cab
    nodal = hcat(s, beamDisplacements, solution.twist, solution.twistRate)
    sMid = [(s[assembly.elemConn[e, 1]] + s[assembly.elemConn[e, 2]]) / 2 for e in 1:size(assembly.elemConn, 1)]
    elements = hcat(sMid, solution.forces, solution.moments, solution.strains, solution.curvatures, solution.bimoment, solution.saintVenantTorque)
    return nodal, elements
end

const NODAL_HEADER = ["s", "u1", "u2", "u3", "twist", "twistRate"]
const ELEMENT_HEADER = ["s", "F1", "F2", "F3", "M1", "M2", "M3", "gamma11", "gamma12", "gamma13", "kappa1", "kappa2", "kappa3", "bimoment", "saintVenantTorque"]

"""
Figures of the spanwise results with Plots (only when it loaded): deflections and twist, curvatures
and strains, internal loads, and the load-deflection history, nonlinear against linear.
"""
function plot_strainbeam_spanwise(nodalNonlinear, nodalLinear, elementsNonlinear, elementsLinear, history, outputPaths)
    p1 = plot(xlabel="span s [m]", ylabel="displacement [m]", title="Deflections", legend=:topleft)
    for (j, name) in zip(2:4, ["u1 axial", "u2 chordwise", "u3 flap"])
        plot!(p1, nodalNonlinear[:, 1], nodalNonlinear[:, j], label="$(name) nonlinear", lw=2)
        plot!(p1, nodalLinear[:, 1], nodalLinear[:, j], label="$(name) linear", ls=:dash)
    end
    p2 = plot(xlabel="span s [m]", ylabel="twist [deg]", title="Twist (nose-down negative)", legend=:bottomleft)
    plot!(p2, nodalNonlinear[:, 1], rad2deg.(nodalNonlinear[:, 5]), label="nonlinear", lw=2)
    plot!(p2, nodalLinear[:, 1], rad2deg.(nodalLinear[:, 5]), label="linear", ls=:dash)
    p3 = plot(xlabel="span s [m]", ylabel="curvature [1/m]", title="Curvatures and axial strain", legend=:topright)
    for (j, name) in zip(11:13, ["κ1", "κ2", "κ3"])
        plot!(p3, elementsNonlinear[:, 1], elementsNonlinear[:, j], label="$(name) nonlinear", lw=2)
        plot!(p3, elementsLinear[:, 1], elementsLinear[:, j], label="$(name) linear", ls=:dash)
    end
    plot!(twinx(p3), elementsNonlinear[:, 1], elementsNonlinear[:, 8], label="γ11 nonlinear", color=:black, ylabel="γ11")
    p4 = plot(xlabel="span s [m]", ylabel="load [N, N m]", title="Internal loads (deformed frame)", legend=:topright)
    for (j, name) in zip(2:7, ["F1", "F2", "F3", "M1", "M2", "M3"])
        plot!(p4, elementsNonlinear[:, 1], elementsNonlinear[:, j], label="$(name) nonlinear", lw=2)
        plot!(p4, elementsLinear[:, 1], elementsLinear[:, j], label="$(name) linear", ls=:dash)
    end
    p5 = plot(xlabel="span s [m]", ylabel="bimoment [N m²]", title="Vlasov bimoment and twist rate", legend=:topright)
    plot!(p5, elementsNonlinear[:, 1], elementsNonlinear[:, 14], label="Mω", lw=2)
    plot!(twinx(p5), nodalNonlinear[:, 1], nodalNonlinear[:, 6], label="κ1 = twist rate", color=:red, ylabel="κ1 [1/m]")
    p6 = plot(xlabel="load factor", ylabel="tip flap deflection [m]", title="Tip response", legend=:topleft)
    plot!(p6, history[:, 1], history[:, 2], label="u3 nonlinear", lw=2, marker=:circle)
    plot!(p6, history[:, 1], history[:, 3], label="u3 linear", ls=:dash)
    plot!(twinx(p6), history[:, 1], rad2deg.(history[:, 4]), label="twist nonlinear [deg]", color=:green, ylabel="tip twist [deg]")
    figure = plot(p1, p2, p3, p4, p5, p6, layout=(3, 2), size=(1400, 1500), left_margin=8Plots.mm, bottom_margin=6Plots.mm)
    for path in outputPaths
        mkpath(dirname(path))
        savefig(figure, path)
    end
    return figure
end

"""
Carbon-fiber NACA0009 cantilever (chord 0.1 m, span 1 m, +15 degree fibers, midchord reference axis)
under an upward tip force stepped into the large-deflection regime, with the Vlasov warping term on.
Checks: agreement with the linearized solution at small load, geometric stiffening under the tip force
(nonlinear deflection below linear, monotone with load), axial foreshortening, nose-down twist of the
+15 degree layup, strain energy against the external work along the load path, the natural bimoment
condition at the tip, and the restrained-warping sensitivity of the tip twist. Writes the spanwise
tables as CSV to `outputDir` and the figures when Plots is available.
"""
function test_StrainBeam_NACA0009(; outputDir=joinpath(@__DIR__, "..", "OUTPUT", "strainbeam"), figureDir=joinpath(@__DIR__, "..", "docs", "figures"))
    errors = Float64[]
    L = 1.0
    nElem = 40
    Ptip = 300.0 # N upward, about 0.3 L of linear tip deflection
    section, properties = naca0009_section(deg2rad(15); warping=StrainBeam.VlasovWarping())
    sectionNoWarping = BeamSection(properties, StrainBeam.NoWarping())
    # Cluster elements toward the root to resolve the warping boundary layer 1/λ ≈ 3 cm
    stations = L .* (1 .- cos.(range(0, pi / 2, length=nElem + 1)))
    tip = dead_load([0.0, 0.0, Ptip], zeros(3))
    assembly = make_cantilever_assembly(L, nElem, section; stations=stations)
    model = StaticModel(assembly, Dict(1 => clamped(), nElem + 1 => tip))
    nonlinear = solve_static(model; nLoadSteps=40, tol=1e-9)
    linear = linear_solve(model)
    println("StrainBeam NACA0009: GJ = $(section.stiffness[4, 4]), EI2 = $(section.stiffness[5, 5]), Ŝ55 = $(section.warpingStiffness), " *
            "1/λ = $(sqrt(section.warpingStiffness / section.stiffness[4, 4])) m, nStates = $(model.indices.nStates)")

    # (1) Small load: the nonlinear solve reproduces the linearized solution
    smallModel = StaticModel(assembly, Dict(1 => clamped(), nElem + 1 => dead_load([0.0, 0.0, Ptip / 1000], zeros(3))))
    smallNonlinear = solve_static(smallModel; nLoadSteps=2)
    smallLinear = linear_solve(smallModel)
    push!(errors, norm(smallNonlinear.displacements - smallLinear.displacements) / norm(smallLinear.displacements) / 1e-3)
    push!(errors, normalized_error(smallNonlinear.twist[end], smallLinear.twist[end], 1e-3))
    # Clapeyron at small load: external work ½ P u3 equals the strain energy
    _, energySmall = work_energy_balance(smallLinear, smallModel)
    push!(errors, normalized_error(energySmall, Ptip / 1000 * smallLinear.displacements[end, 3] / 2, 1e-6))

    # (2) Geometric stiffening: nonlinear tip deflection below linear and monotone in the load
    tipHistory = [nodal_displacements(state, model, λ)[end, 3] for (λ, state) in zip(nonlinear.loadFactors, nonlinear.stateHistory)]
    push!(errors, passes(nonlinear.displacements[end, 3] < linear.displacements[end, 3]))
    push!(errors, passes(all(diff(tipHistory) .> 0)))
    push!(errors, passes(nonlinear.displacements[end, 3] > 0.2L)) # genuinely large deflection
    # (3) Foreshortening along the beam (-y) and nose-down twist of the +15 degree layup
    push!(errors, passes(nonlinear.displacements[end, 2] < -1e-3L))
    push!(errors, passes(linear.twist[end] < 0 && nonlinear.twist[end] < 0))
    # (4) Energy balance along the load path (trapezoidal in 40 steps)
    externalWork, energy = work_energy_balance(nonlinear, model)
    push!(errors, normalized_error(energy, externalWork, 2e-3))
    # (5) Bimoment vanishes at the free tip and the warping restraint reduces the twist a little
    push!(errors, passes(abs(nonlinear.bimoment[end]) < 0.05 * maximum(abs.(nonlinear.bimoment))))
    push!(errors, passes(abs(nonlinear.twistRate[1]) < 1e-12)) # restrained warping at the clamped root
    modelNoWarping = StaticModel(make_cantilever_assembly(L, nElem, sectionNoWarping; stations=stations), Dict(1 => clamped(), nElem + 1 => tip))
    noWarping = solve_static(modelNoWarping; nLoadSteps=40)
    twistReduction = 1 - nonlinear.twist[end] / noWarping.twist[end]
    println("StrainBeam NACA0009: tip u3 nonlinear $(nonlinear.displacements[end, 3]) m vs linear $(linear.displacements[end, 3]) m, " *
            "tip twist $(rad2deg(nonlinear.twist[end])) deg, Vlasov twist reduction $(twistReduction)")
    push!(errors, passes(0 < twistReduction < 0.1))

    # Spanwise tables, history and figures
    nodalNonlinear, elementsNonlinear = spanwise_tables(nonlinear, model)
    nodalLinear, elementsLinear = spanwise_tables(linear, model)
    history = zeros(length(nonlinear.loadFactors), 4)
    for (k, (λ, state)) in enumerate(zip(nonlinear.loadFactors, nonlinear.stateHistory))
        step = StrainBeam.build_solution(state, model, λ, [λ], [state], [0], [0.0])
        history[k, :] .= [λ, step.displacements[end, 3], λ * linear.displacements[end, 3], step.twist[end]]
    end
    mkpath(outputDir)
    write_csv(joinpath(outputDir, "naca0009_nodal_nonlinear.csv"), NODAL_HEADER, nodalNonlinear)
    write_csv(joinpath(outputDir, "naca0009_nodal_linear.csv"), NODAL_HEADER, nodalLinear)
    write_csv(joinpath(outputDir, "naca0009_elements_nonlinear.csv"), ELEMENT_HEADER, elementsNonlinear)
    write_csv(joinpath(outputDir, "naca0009_elements_linear.csv"), ELEMENT_HEADER, elementsLinear)
    write_csv(joinpath(outputDir, "naca0009_load_history.csv"), ["loadFactor", "tipU3Nonlinear", "tipU3Linear", "tipTwistNonlinear"], history)
    if HAS_PLOTS
        plot_strainbeam_spanwise(nodalNonlinear, nodalLinear, elementsNonlinear, elementsLinear, history,
            ["test_strainbeam_naca0009_spanwise.pdf", joinpath(figureDir, "strainbeam_naca0009_spanwise.pdf")])
    else
        println("StrainBeam NACA0009: Plots unavailable, figures skipped; run test/plot_strainbeam.py on $(outputDir)")
    end
    return maximum(errors)
end

# ==============================================================================
#                         Closed-form benchmarks
# ==============================================================================
"""
Bisshopp and Drucker (1945) large-deflection cantilever under a transverse dead tip force P (the
elastica), α = P L² / EI = 1, 2, 5, 10. The references were re-derived by RK4 shooting on
θ″(ξ) = -α cos θ, θ(0) = 0, θ′(1) = 0 and agree with Bisshopp and Drucker / Mattiasson (1981) to five
digits:
    α       δ/L        Δ/L       θ_tip [rad]
    1     0.30172    0.05643      0.46135
    2     0.49346    0.16064      0.78175
    5     0.71379    0.38763      1.21537
    10    0.81061    0.55500      1.43029
EA and GA are 1e9 so shear and extension (γ ~ F/EA ~ 1e-5) stay below the inextensible, shear-rigid
elastica. The external work is path dependent for a dead load on a nonlinear structure
(F_tip·δ / U = 2.10, 2.34, 3.19, 4.52 at α = 1, 2, 5, 10 from the elastica itself), so the balance uses
the trapezoidal integral over the load steps; its error bottoms out near 4e-6 at 300 steps (truncation
falls as 1/n², round-off in the increments grows beyond about 400 steps), hence the 1e-5 tolerance.
"""
function test_StrainBeam_tipForce()
    errors = Float64[]
    EI = 1e5
    L = 10.0
    nElem = 32
    section = isotropic_section(EA=1e9, GA=1e9, EI2=EI, EI3=EI)
    references = Dict(
        1.0 => (δ=0.30172, Δ=0.05643, θtip=0.46135),
        2.0 => (δ=0.49346, Δ=0.16064, θtip=0.78175),
        5.0 => (δ=0.71379, Δ=0.38763, θtip=1.21537),
        10.0 => (δ=0.81061, Δ=0.55500, θtip=1.43029),
    )
    e1 = SVector(1.0, 0.0, 0.0)
    for α in [1.0, 2.0, 5.0, 10.0]
        P = α * EI / L^2
        model = cantilever_model(section, L, nElem, dead_load([0.0, 0.0, P], zeros(3)))
        solution = solve_static(model; nLoadSteps=300, tol=1e-11, maxIters=40)
        reference = references[α]
        δ = solution.displacements[end, 3]
        Δ = -solution.displacements[end, 2]
        push!(errors, normalized_error(δ / L, reference.δ, 5e-3))
        push!(errors, normalized_error(Δ / L, reference.Δ, 5e-3))
        # Tip rotation: angle of the deformed axis R Cab e1 from the undeformed axis in the bending plane
        Cab = model.assembly.elementFrames[end]
        deformedAxis = solution.rotations[end] * (Cab * e1)
        push!(errors, normalized_error(atan(deformedAxis[3], deformedAxis[2]), reference.θtip, 5e-3))
        externalWork, internalEnergy = work_energy_balance(solution, model)
        push!(errors, normalized_error(externalWork, internalEnergy, 1e-5))
    end
    return maximum(errors)
end

"""
Small-load linear limit of the CFRP NACA0009 cantilever (chord 0.1 m, L = 1 m along +y, 40 elements)
under a small upward dead tip force P e3 against (1) the exact statically determinate linear solution
built from the section's generalized Timoshenko compliance, (2) the second-order foreshortening the
linear solution cannot have, and (3) the Euler-Bernoulli coupled closed form from the condensed
classical stiffness, the level of theory behind DCFoil's linear COMP2 element.

Sign convention (right-handed e1 along the span, e2 = -x toward the leading edge, e3 = +z): the moment
of the tip force P e3 about a station at arc length x is (r_tip - r_x) × F = (L - x) e1 × P e3
= -(L - x) P e2, since e1 × e3 = -e2. So F(x) = P e3 and M(x) = -(L - x) P e2, zero at the tip and
largest at the clamped root; an upward tip force bends the beam concave up, a negative M2.
"""
function test_StrainBeam_linearLimit()
    errors = Float64[]
    section, properties = naca0009_section(deg2rad(15))
    L = 1.0
    nElem = 40
    P = 0.35 # N, about 1.0e-3 L of tip deflection for this section
    model = cantilever_model(section, L, nElem, dead_load([0.0, 0.0, P], zeros(3)))
    solution = linear_solve(model)

    # (1) Exact linear solution: [F; M](x) = s0 + s1 x with s0 = [P e3; -L P e2] and s1 = [0; P e2],
    # so with constant compliance Φ the strains are affine, θ = ∫κ quadratic and u = ∫(γ + θ × e1) cubic
    Cab = StrainBeam.undeformed_frame([0.0, 1.0, 0.0])
    Φ = section.compliance
    e1 = SVector(1.0, 0.0, 0.0)
    s0 = Φ * SVector(0.0, 0.0, P, 0.0, -L * P, 0.0)
    s1 = Φ * SVector(0.0, 0.0, 0.0, 0.0, P, 0.0)
    γ0, γ1 = SVector(s0[1], s0[2], s0[3]), SVector(s1[1], s1[2], s1[3])
    κ0, κ1 = SVector(s0[4], s0[5], s0[6]), SVector(s1[4], s1[5], s1[6])
    γb(x) = γ0 + γ1 * x
    κb(x) = κ0 + κ1 * x
    θb(x) = κ0 * x + κ1 * x^2 / 2
    ub(x) = γ0 * x + (γ1 + cross(κ0, e1)) * x^2 / 2 + (cross(κ1, e1) / 2) * x^3 / 3
    stations = collect(range(0, L, length=nElem + 1))
    uTipScale = norm(ub(L))
    θTipScale = norm(θb(L))
    for (i, x) in enumerate(stations)
        push!(errors, norm(solution.displacements[i, :] - Cab * ub(x)) / uTipScale / 1e-3)
        # Small rotations: the WM parameters are the global rotation vector Cab θb to leading order
        push!(errors, norm(solution.rotationParameters[i, :] - Cab * θb(x)) / θTipScale / 1e-3)
    end
    # Element loads are exact per element (statically determinate), so strains match the affine
    # closed form at the element midpoints to machine precision
    for e in 1:nElem
        xMid = (stations[e] + stations[e+1]) / 2
        push!(errors, norm(solution.strains[e, :] - γb(xMid)) / norm(γb(xMid)) / 1e-6)
        push!(errors, norm(solution.curvatures[e, :] - κb(xMid)) / norm(κb(xMid)) / 1e-6)
    end
    # Nose-down twist for +15 degree fibers under an upward tip force
    push!(errors, passes(solution.curvatures[end, 1] < 0))
    push!(errors, passes(solution.twist[end] < 0))

    # (2) The nonlinear solve at the same small load agrees to second order in the rotations for the
    # flap deflection and the twist, and its axial displacement is the foreshortening -½ ∫ |θ × e1|² dx
    nonlinearSmall = solve_static(model; nLoadSteps=4)
    push!(errors, normalized_error(nonlinearSmall.displacements[end, 3], solution.displacements[end, 3], 1e-4))
    push!(errors, normalized_error(nonlinearSmall.twist[end], solution.twist[end], 1e-4))
    xFine = range(0, L, length=4001)
    foreshortening = -sum(norm(cross(θb(x), e1))^2 for x in xFine) * step(xFine) / 2
    push!(errors, normalized_error(nonlinearSmall.displacements[end, 2], foreshortening, 5e-2))
    # The nonlinear/linear gap grows with load: at 100 P it exceeds 1%
    model100 = cantilever_model(section, L, nElem, dead_load([0.0, 0.0, 100P], zeros(3)))
    gap100 = norm(solve_static(model100; nLoadSteps=20).displacements[end, :] - linear_solve(model100).displacements[end, :]) /
             norm(linear_solve(model100).displacements[end, :])
    push!(errors, passes(gap100 > 0.01))

    # (3) Euler-Bernoulli coupled closed form from the condensed classical 3x3 [κ1, κ2, κ3]:
    # κ2 = Φc22 M2 with M2 = -P(L - x) and κ2 = -w″ gives w(L) = P L³ Φc22 / 3; κ1 = Φc12 M2 gives
    # φ(L) = -Φc12 P L² / 2. The 5% gap to the Timoshenko result is transverse shear amplified by the
    # strong bend-twist-shear coupling of the +15 degree layup (S45 / sqrt(S44 S55) ≈ -0.71); at 0
    # degrees the same comparison is within 0.5%, the naive shear estimate 3 EI / (GA L²).
    Φc = inv(Main.VAM.condense_axial_stiffness(properties.stiffnessClassical))
    wEB = P * L^3 * Φc[2, 2] / 3
    φEB = -Φc[1, 2] * P * L^2 / 2
    println("StrainBeam linear limit: P = $(P) N, tip deflection/L = $(solution.displacements[end, 3] / L); " *
            "Timoshenko vs Euler-Bernoulli deflection ratio $(solution.displacements[end, 3] / wEB), twist ratio $(solution.twist[end] / φEB)")
    push!(errors, normalized_error(solution.displacements[end, 3], wEB, 7e-2))
    push!(errors, normalized_error(solution.twist[end], φEB, 7e-2))
    return maximum(errors)
end

"""
Restrained-warping torsion of the Hodges (2006) Table 7.9 isotropic strip: tip torque T about the
beam axis on a clamped cantilever of length L = 2 in (λL ≈ 8.5), against the closed form of Yu et al.
(2005) Eqs. 6-10 with Mω = ∂U/∂κ1′ and total torque M1 - Mω′ = T:
    θ′(x) = (T/GJ) [1 - cosh(λ(L-x))/cosh(λL)],  θ(x) = (T/GJ) [x - tanh(λL)/λ + sinh(λ(L-x))/(λ cosh(λL))],
    Mω(x) = (T/λ) sinh(λ(L-x))/cosh(λL),  M1,SV(x) = GJ θ′(x),  total torque = T,
λ = sqrt(GJ/Ŝ55). GJ and Ŝ55 come from the section (checked against 32.3418 lb in² and 1.78635 lb in⁴
at the 5e-4 the VAM strip test accepts). The mesh is graded s = L ξ² toward the root so about 11 of
the 32 elements resolve the boundary layer 1/λ ≈ 0.235 in; a uniform mesh fails the Saint-Venant
torque check by 40%.
"""
function test_StrainBeam_vlasovTorsion()
    errors = Float64[]
    section, _ = vlasov_strip_section()
    GJ = section.stiffness[4, 4]
    S55 = section.warpingStiffness
    push!(errors, normalized_error(GJ, 32.3418, 5e-4))
    push!(errors, normalized_error(S55, 1.78635, 5e-4))
    L = 2.0
    λ = sqrt(GJ / S55)
    λL = λ * L
    T = 0.05 * GJ / (L - tanh(λL) / λ) # tip twist about 0.05 rad, linear regime
    nElem = 32
    stations = L .* (collect(0:nElem) ./ nElem) .^ 2
    assembly = make_cantilever_assembly(L, nElem, section; stations=stations)
    tip = dead_load(zeros(3), [0.0, T, 0.0])
    solution = solve_static(StaticModel(assembly, Dict(1 => clamped(), nElem + 1 => tip)); nLoadSteps=1, tol=1e-11)

    θref(x) = (T / GJ) * (x - tanh(λL) / λ + sinh(λ * (L - x)) / (λ * cosh(λL)))
    θprimeRef(x) = (T / GJ) * (1 - cosh(λ * (L - x)) / cosh(λL))
    Mωref(x) = (T / λ) * sinh(λ * (L - x)) / cosh(λL)
    push!(errors, normalized_error(solution.twist[end], θref(L), 5e-3))
    nodeArc = assembly.nodeCoords[:, 2]
    for i in eachindex(nodeArc)
        push!(errors, normalized_absolute_error(solution.twistRate[i], θprimeRef(nodeArc[i]), 5e-3 * T / GJ))
    end
    for e in 1:nElem
        xMid = (nodeArc[assembly.elemConn[e, 1]] + nodeArc[assembly.elemConn[e, 2]]) / 2
        push!(errors, normalized_absolute_error(solution.bimoment[e], Mωref(xMid), 1e-2 * Mωref(0.0)))
        push!(errors, normalized_error(solution.moments[e, 1], T, 1e-8)) # total torque everywhere
        push!(errors, normalized_absolute_error(solution.saintVenantTorque[e], GJ * θprimeRef(xMid), 5e-3 * T))
    end
    svRatio = 1 - tanh(λL) / λL
    println("StrainBeam Vlasov strip: λ = $(λ) 1/in, θ(L)/(T L/GJ) = $(solution.twist[end] / (T * L / GJ)) (closed form $(svRatio))")
    push!(errors, normalized_error(solution.twist[end] / (T * L / GJ), svRatio, 5e-3))
    # Without warping the strip is in unrestrained Saint-Venant torsion, θ(L) = T L / GJ
    sectionNoWarping, _ = vlasov_strip_section(warping=StrainBeam.NoWarping())
    noWarping = solve_static(StaticModel(make_cantilever_assembly(L, nElem, sectionNoWarping; stations=stations),
            Dict(1 => clamped(), nElem + 1 => tip)); nLoadSteps=1, tol=1e-11)
    push!(errors, normalized_error(noWarping.twist[end], T * L / GJ, 1e-6))
    return maximum(errors)
end
