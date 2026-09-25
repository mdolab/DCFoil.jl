# --- Julia 1.11---
"""
@File    :   StrainBeam.jl
@Time    :   2026/09/24
@Author  :   Galen Ng
@Desc    :   Geometrically exact, strain-based nonlinear static beam model fed by the VAM section
             properties. The 1D theory is the intrinsic formulation of Hodges (2006) "Nonlinear
             Composite Beam Theory" Ch. 5 with the mixed variational discretization of Hodges (1990)
             as used in GEBT (Yu and Blair 2012): two-node elements carrying constant internal loads
             F, M in the deformed frame and nodes carrying displacements u and Wiener-Milenkovic
             rotation parameters θ. The generalized Vlasov restrained-warping term of Yu, Hodges,
             Volovoi and Fuchs (2005) is included as one extra conjugate pair (twist rate κ1 at the
             nodes, bimoment Mω on the elements). Conventions follow GXBeam.jl (McDonnell and Ning 2022).
             Nodal rotations are raw Wiener-Milenkovic parameters, exact for rotations below 360 degrees.

             Frames: a = global (x flow, y span, z up); b = undeformed element frame with e1 along
             the element, e2 toward the leading edge for a starboard wing, e3 up; B = deformed frame.
             Cab has the b basis vectors as columns in global components (v_a = Cab v_b). C(θ) is the
             Wiener-Milenkovic matrix of the nodal rotation; Cᵀ Cab maps B components to a components.

             Section input: the VAM generalized Timoshenko 6x6 stiffness in the ordering
             [γ11, 2γ12, 2γ13, κ1, κ2, κ3] conjugate to [F1, F2, F3, M1, M2, M3], the 6x6 mass matrix
             in [V1, V2, V3, Ω1, Ω2, Ω3], and optionally the Vlasov warping rigidity Ŝ55 = EΓ.
"""

module StrainBeam

using LinearAlgebra
using StaticArrays
using ForwardDiff

export BeamSection, BeamAssembly, PointCondition, DistributedLoad, StaticModel, StaticSolution
export make_cantilever_assembly, undeformed_frame
export clamped, dead_load, follower_load, prescribed_displacement, distributed_load
export tilde, wiener_milenkovic_rotation, rotation_tangent, rotation_tangent_inverse
export rescale_rotation_parameters, rotation_parameters, compose_rotation_parameters, twist_angle
export state_indices, compute_residual, compute_jacobian
export solve_static, linear_solve, newton_solve, work_energy_balance, to_fem_states, nodal_displacements

const E1 = SVector(1.0, 0.0, 0.0)
const E2 = SVector(0.0, 1.0, 0.0)
const E3 = SVector(0.0, 0.0, 1.0)
const I3 = SMatrix{3,3}(1.0I)

# ==============================================================================
#                         Rotations (Wiener-Milenkovic parameters)
# ==============================================================================
"""
Skew-symmetric matrix of a 3-vector so that tilde(a) * b = a × b.
"""
tilde(a) = @SMatrix [0 -a[3] a[2]; a[3] 0 -a[1]; -a[2] a[1] 0]

"""
Wiener-Milenkovic parameters with |c| > 4 (rotation angle beyond 180 degrees) mapped onto the
equivalent parameters of the same rotation, c* = -16 c / (cᵀc): the rotation by φ about n equals the
rotation by 2π - φ about -n, and 4 tan((2π - φ)/4) = 16 / (4 tan(φ/4)) (Bauchau 2011). This is a
post-processing utility: the residual works on the raw nodal parameters, which represent every
rotation below 360 degrees exactly, so rescaling inside the kinematics would make the parameter
increments of the compatibility residual inconsistent with the stored unknowns.
"""
function rescale_rotation_parameters(θ)
    θnormSquared = θ' * θ
    if ForwardDiff.value(θnormSquared) > 16
        return -(16 / θnormSquared) * θ
    end
    return θ
end

"""
Wiener-Milenkovic rotation matrix C(c) with c = 4 tan(φ/4) n and c0 = 2 - cᵀc/8,
C = [(c0² - cᵀc) I + 2 c cᵀ - 2 c0 c̃] / (4 - c0)². For small angles C ≈ I - c̃, so C is the
transpose of the rotation tensor R = exp(φ ñ) that rotates vectors by φ about n. The rational form
is exact for every |c| < ∞, i.e. for rotation angles below 360 degrees.
"""
function wiener_milenkovic_rotation(c)
    c0 = 2 - c' * c / 8
    return ((c0^2 - c' * c) * I3 + 2 * c * c' - 2 * c0 * tilde(c)) / (4 - c0)^2
end

"""
Tangent matrix Q(c) mapping the parameter rate to the material angular velocity, Q c′ = axial(C Ċᵀ),
so that the curvature in the deformed frame is κ = Cᵀ_ab Q θ′; Q = [(4 - cᵀc/4) I - 2 c̃ + c cᵀ/2] / (4 - c0)².
"""
function rotation_tangent(c)
    c0 = 2 - c' * c / 8
    return ((4 - c' * c / 4) * I3 - 2 * tilde(c) + c * c' / 2) / (4 - c0)^2
end

"""
Closed-form inverse of the tangent matrix, Q⁻¹ = (1 - cᵀc/16) I + c̃/2 + c cᵀ/8.
"""
function rotation_tangent_inverse(c)
    return (1 - c' * c / 16) * I3 + tilde(c) / 2 + c * c' / 8
end

"""
Wiener-Milenkovic parameters of the rotation whose matrix is C (inverse of
wiener_milenkovic_rotation) using the angle from the trace and the axis from the skew part.
"""
function rotation_parameters(C::AbstractMatrix)
    R = transpose(C)
    cosφ = clamp((tr(R) - 1) / 2, -1.0, 1.0)
    φ = acos(cosφ)
    if φ < 1e-12
        return SVector(0.0, 0.0, 0.0)
    end
    axial = SVector(R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2])
    n = axial / (2 * sin(φ))
    return 4 * tan(φ / 4) * n
end

"""
Parameters of the rotation obtained by applying θFirst and then θSecond. Since C is the transpose of
the rotation tensor, the total matrix is C(θFirst) C(θSecond).
"""
function compose_rotation_parameters(θFirst, θSecond)
    return rotation_parameters(wiener_milenkovic_rotation(θFirst) * wiener_milenkovic_rotation(θSecond))
end

"""
Minimal rotation tensor (global components) taking the unit vector `from` onto the unit vector `to`.
"""
function minimal_rotation(from, to)
    axis = cross(from, to)
    sinφ = norm(axis)
    cosφ = dot(from, to)
    if sinφ < 1e-14
        if cosφ > 0
            return I3
        end
        # Antiparallel: rotate by π about any axis perpendicular to `from`
        perpendicular = abs(from[3]) < 0.9 ? cross(from, E3) : cross(from, E2)
        n = perpendicular / norm(perpendicular)
        return -I3 + 2 * n * n'
    end
    n̂ = tilde(axis / sinφ)
    return I3 + sinφ * n̂ + (1 - cosφ) * n̂^2
end

"""
Twist of a deformed section: the rotation angle about the deformed beam axis that remains after
removing the minimal rotation aligning e1 with the deformed axis. `Rb` is the deformation rotation
tensor expressed in the undeformed element frame b, Rb = Cabᵀ Cᵀ Cab.
"""
function twist_angle(Rb::AbstractMatrix)
    deformedAxis = Rb * E1
    Rbend = minimal_rotation(E1, deformedAxis)
    Rtwist = transpose(Rbend) * Rb
    return atan(Rtwist[3, 2], Rtwist[2, 2])
end

# ==============================================================================
#                         Sections, assembly, loads
# ==============================================================================
"""
Constitutive data of one section. `stiffness` is the VAM generalized Timoshenko 6x6 in
[γ11, 2γ12, 2γ13, κ1, κ2, κ3]; `mass` is the 6x6 mass matrix in [V1, V2, V3, Ω1, Ω2, Ω3];
`warpingStiffness` is the Vlasov Ŝ55 = EΓ about the shear center (0 disables restrained warping).
The reduced blocks serve the Vlasov path, where κ1 is an independent field: with r = [1, 2, 3, 5, 6],
[γ; κ2; κ3] = S_rr⁻¹ ([F; M2; M3] - S_r1 κ1) and the Saint-Venant torque is M1 = S_1r [γ; κ2; κ3] + S_11 κ1.
"""
struct BeamSection{T}
    stiffness::SMatrix{6,6,T,36}
    compliance::SMatrix{6,6,T,36}
    mass::SMatrix{6,6,T,36}
    warpingStiffness::T
    reducedCompliance::SMatrix{5,5,T,25}
    torsionCoupling::SVector{5,T} # S_r1 = S_1rᵀ
    torsionStiffness::T # S_11 (the κ1 diagonal, GJ-like)
end

const REDUCED_INDICES = SVector(1, 2, 3, 5, 6)
const TORSION_INDEX = 4

function BeamSection(stiffnessTimoshenko::AbstractMatrix, massMatrix::AbstractMatrix; warpingStiffness=0.0)
    T = float(promote_type(eltype(stiffnessTimoshenko), eltype(massMatrix), typeof(warpingStiffness)))
    S = SMatrix{6,6,T}(stiffnessTimoshenko)
    Srr = S[REDUCED_INDICES, REDUCED_INDICES]
    return BeamSection{T}(S, inv(S), SMatrix{6,6,T}(massMatrix), T(warpingStiffness), inv(Srr),
        S[REDUCED_INDICES, TORSION_INDEX], S[TORSION_INDEX, TORSION_INDEX])
end

"""
Section from a VAM `CrossSectionProperties` (duck-typed so both module instances work). Pass
`includeWarping=VlasovWarping()` to take Ŝ55 from the Vlasov 5x5.
"""
struct VlasovWarping end
struct NoWarping end

function BeamSection(properties, ::VlasovWarping)
    return BeamSection(properties.stiffnessTimoshenko, properties.massMatrix; warpingStiffness=properties.stiffnessVlasov[5, 5])
end
BeamSection(properties, ::NoWarping) = BeamSection(properties.stiffnessTimoshenko, properties.massMatrix)

has_warping(section::BeamSection) = section.warpingStiffness > 0

"""
Undeformed element frame Cab (columns = b basis in global components) for an element along
`direction`. With `sectionNormal` given, e3 is that vector made orthogonal to the element (the VAM x3
direction) and e2 = e3 × e1; without it the frame is the minimal rotation taking the global x axis
onto the element, which for an element along +y gives e2 = -x (toward the leading edge) and e3 = +z,
the DCFoil convention.
"""
function undeformed_frame(direction::AbstractVector; sectionNormal=nothing)
    e1 = SVector{3}(direction) / norm(direction)
    if sectionNormal === nothing
        return minimal_rotation(E1, e1)
    end
    normal = SVector{3}(sectionNormal)
    e3 = normal - dot(normal, e1) * e1
    norm(e3) > 1e-12 || error("sectionNormal must not be parallel to the element direction")
    e3 = e3 / norm(e3)
    e2 = cross(e3, e1)
    return hcat(e1, e2, e3)
end

"""
Straight beam of nNodes = nElem + 1 nodes with elements `elemConn` [nElem, 2] and one section per element.
"""
struct BeamAssembly{T}
    nodeCoords::Matrix{T} # [nNodes, 3] global
    elemConn::Matrix{Int} # [nElem, 2]
    sections::Vector{BeamSection{T}}
    elementFrames::Vector{SMatrix{3,3,T,9}} # Cab per element
    lengths::Vector{T}
end

"""
`sectionNormal` fixes the section x3 direction of every element (see `undeformed_frame`); `nothing`
selects the minimal-rotation frame.
"""
function BeamAssembly(nodeCoords::AbstractMatrix, elemConn::AbstractMatrix{Int}, sections::AbstractVector{<:BeamSection}; sectionNormal=nothing)
    nElem = size(elemConn, 1)
    length(sections) == nElem || error("Need one section per element, got $(length(sections)) for $(nElem) elements")
    T = float(promote_type(eltype(nodeCoords), eltype(sections[1].stiffness)))
    frames = SMatrix{3,3,T,9}[]
    lengths = T[]
    for e in 1:nElem
        dR = SVector{3,T}(nodeCoords[elemConn[e, 2], :] - nodeCoords[elemConn[e, 1], :])
        push!(lengths, norm(dR))
        push!(frames, undeformed_frame(dR; sectionNormal=sectionNormal))
    end
    return BeamAssembly{T}(Matrix{T}(nodeCoords), Matrix{Int}(elemConn), Vector{BeamSection{T}}(sections), frames, lengths)
end

n_nodes(assembly::BeamAssembly) = size(assembly.nodeCoords, 1)
n_elements(assembly::BeamAssembly) = size(assembly.elemConn, 1)
has_warping(assembly::BeamAssembly) = any(has_warping, assembly.sections)

"""
Cantilever of `span` along `direction` from the origin with `nElem` equal elements of one section.
`stations` overrides the uniform nodal arc-length distribution (values from 0 to span).
"""
function make_cantilever_assembly(span, nElem::Int, section::BeamSection; direction=[0.0, 1.0, 0.0], stations=nothing, sectionNormal=nothing)
    s = stations === nothing ? collect(range(0, span, length=nElem + 1)) : collect(stations)
    d = direction / norm(direction)
    nodeCoords = s * transpose(d)
    elemConn = hcat(1:nElem, 2:nElem+1)
    return BeamAssembly(nodeCoords, elemConn, fill(section, nElem); sectionNormal=sectionNormal)
end

"""
Boundary condition and concentrated loads of one node. For every one of the six components either
the displacement/rotation is prescribed (the state slot then holds the unknown reaction) or the load
is prescribed. Dead loads are global-frame vectors; follower loads are deformed-frame vectors rotated
by Cᵀ every iteration. Rotations are prescribed as Wiener-Milenkovic parameters.
"""
struct PointCondition{T}
    prescribedDisplacement::SVector{6,Bool}
    displacement::SVector{6,T}
    deadLoad::SVector{6,T} # [F; M] global
    followerLoad::SVector{6,T} # [F; M] deformed frame
    restrainedWarping::Bool # Vlasov: twist rate κ1 = 0 at this node (its slot holds the reaction bimoment)
end

"""
`restrainedWarping` defaults to a fully clamped rotation, the only case where the twist axis of the
adjoining elements is known to be constrained whatever the beam direction.
"""
function PointCondition(; prescribedDisplacement=falses(6), displacement=zeros(6), deadLoad=zeros(6), followerLoad=zeros(6),
    restrainedWarping=all(prescribedDisplacement[4:6]))
    T = float(promote_type(eltype(displacement), eltype(deadLoad), eltype(followerLoad)))
    return PointCondition{T}(SVector{6,Bool}(prescribedDisplacement), SVector{6,T}(displacement),
        SVector{6,T}(deadLoad), SVector{6,T}(followerLoad), restrainedWarping)
end

clamped() = PointCondition(prescribedDisplacement=trues(6))
dead_load(F, M) = PointCondition(deadLoad=vcat(F, M))
follower_load(F, M) = PointCondition(followerLoad=vcat(F, M))
prescribed_displacement(u, θ) = PointCondition(prescribedDisplacement=trues(6), displacement=vcat(u, θ))

"""
Distributed load of one element pre-integrated to its two ends with the linear weights (1 - ξ) and
ξ, ξ being the element coordinate. Dead components are global, follower components deformed-frame.
"""
struct DistributedLoad{T}
    f1::SVector{3,T}
    f2::SVector{3,T}
    m1::SVector{3,T}
    m2::SVector{3,T}
    f1Follower::SVector{3,T}
    f2Follower::SVector{3,T}
    m1Follower::SVector{3,T}
    m2Follower::SVector{3,T}
end

function DistributedLoad(; f1=zeros(3), f2=zeros(3), m1=zeros(3), m2=zeros(3), f1Follower=zeros(3), f2Follower=zeros(3), m1Follower=zeros(3), m2Follower=zeros(3))
    T = float(promote_type(eltype(f1), eltype(f2), eltype(m1), eltype(m2), eltype(f1Follower), eltype(f2Follower), eltype(m1Follower), eltype(m2Follower)))
    return DistributedLoad{T}(f1, f2, m1, m2, f1Follower, f2Follower, m1Follower, m2Follower)
end

const GAUSS_4 = (nodes=SVector(-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526),
    weights=SVector(0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538))

"""
Integrate global-frame force and moment per unit length, `force(s)` and `moment(s)` with s the arc
length from the element start, onto the element ends with four-point Gauss-Legendre quadrature.
"""
function distributed_load(assembly::BeamAssembly, iElem::Int, force::Function; moment::Function=s -> zeros(3))
    L = assembly.lengths[iElem]
    samples = [(ξ=(1 + ξg) / 2, weight=wg * L / 2) for (ξg, wg) in zip(GAUSS_4.nodes, GAUSS_4.weights)]
    forces = [SVector{3}(force(sample.ξ * L)) for sample in samples]
    moments = [SVector{3}(moment(sample.ξ * L)) for sample in samples]
    f1 = sum((1 - s.ξ) * s.weight * f for (s, f) in zip(samples, forces))
    f2 = sum(s.ξ * s.weight * f for (s, f) in zip(samples, forces))
    m1 = sum((1 - s.ξ) * s.weight * m for (s, m) in zip(samples, moments))
    m2 = sum(s.ξ * s.weight * m for (s, m) in zip(samples, moments))
    return DistributedLoad(f1=f1, f2=f2, m1=m1, m2=m2)
end

"""
Everything the residual needs besides the state: assembly, conditions per node, distributed loads
per element, gravity (global), and the state layout.
"""
struct StateIndices
    pointColumns::Vector{Int}
    elementColumns::Vector{Int}
    pointRows::Vector{Int}
    elementRows::Vector{Int}
    statesPerPoint::Int
    statesPerElement::Int
    nStates::Int
    forceScaling::Float64
    warping::Bool
end

"""
Node-major state layout: 6 unknowns per point (u, θ or their reactions) then 6 per element (F, M in
the deformed frame, scaled by forceScaling). With Vlasov warping 7 and 7 (twist rate κ1 at the points,
bimoment Mω on the elements). Point rows hold equilibrium, element rows hold compatibility.
"""
function state_indices(assembly::BeamAssembly)
    warping = has_warping(assembly)
    perPoint = warping ? 7 : 6
    perElement = warping ? 7 : 6
    nNodes = n_nodes(assembly)
    nElem = n_elements(assembly)
    pointColumns = [1 + (p - 1) * perPoint for p in 1:nNodes]
    elementColumns = [nNodes * perPoint + 1 + (e - 1) * perElement for e in 1:nElem]
    nStates = nNodes * perPoint + nElem * perElement
    return StateIndices(pointColumns, elementColumns, pointColumns, elementColumns, perPoint, perElement, nStates,
        default_force_scaling(assembly), warping)
end

"""
Force scaling so the scaled internal loads are O(strain × 100), following GXBeam: a power of two near
1 / (100 × mean nonzero compliance entry).
"""
function default_force_scaling(assembly::BeamAssembly)
    total = 0.0
    count = 0
    for section in assembly.sections
        for value in section.compliance
            if value != 0
                total += abs(value)
                count += 1
            end
        end
    end
    return count == 0 ? 1.0 : nextpow(2.0, count / total / 100)
end

struct StaticModel{T}
    assembly::BeamAssembly{T}
    conditions::Dict{Int,PointCondition{T}}
    distributedLoads::Dict{Int,DistributedLoad{T}}
    gravity::SVector{3,T}
    indices::StateIndices
end

function StaticModel(assembly::BeamAssembly{T}, conditions::AbstractDict, distributedLoads::AbstractDict=Dict{Int,DistributedLoad{T}}();
    gravity=zeros(3)) where {T}
    return StaticModel{T}(assembly, Dict{Int,PointCondition{T}}(conditions), Dict{Int,DistributedLoad{T}}(distributedLoads),
        SVector{3,T}(gravity), state_indices(assembly))
end

const FREE_POINT = PointCondition()

point_condition(model::StaticModel, iPoint::Int) = get(model.conditions, iPoint, FREE_POINT)

# ==============================================================================
#                         State access
# ==============================================================================
"""
Displacement u and rotation parameters θ of a point: prescribed values (scaled by the load factor)
where prescribed, state entries otherwise.
"""
function point_displacement(x, model::StaticModel, iPoint::Int, loadFactor)
    icol = model.indices.pointColumns[iPoint]
    condition = point_condition(model, iPoint)
    pd = condition.prescribedDisplacement
    T = promote_type(eltype(x), typeof(loadFactor), eltype(condition.displacement))
    values = SVector{6,T}(ntuple(i -> pd[i] ? loadFactor * condition.displacement[i] : x[icol+i-1], 6))
    return SVector(values[1], values[2], values[3]), SVector(values[4], values[5], values[6])
end

"""
External force and moment on a point in global components: prescribed dead loads (scaled by the
load factor) plus follower loads rotated by Cᵀ, or the unknown reaction where the displacement is
prescribed.
"""
function point_loads(x, model::StaticModel, iPoint::Int, C, loadFactor)
    icol = model.indices.pointColumns[iPoint]
    condition = point_condition(model, iPoint)
    pd = condition.prescribedDisplacement
    forceScaling = model.indices.forceScaling
    follower = condition.followerLoad
    Ffollower = transpose(C) * SVector(follower[1], follower[2], follower[3])
    Mfollower = transpose(C) * SVector(follower[4], follower[5], follower[6])
    externalFollower = vcat(Ffollower, Mfollower)
    T = promote_type(eltype(x), typeof(loadFactor), eltype(externalFollower), eltype(condition.deadLoad))
    values = SVector{6,T}(ntuple(i -> pd[i] ? x[icol+i-1] * forceScaling : loadFactor * (condition.deadLoad[i] + externalFollower[i]), 6))
    return SVector(values[1], values[2], values[3]), SVector(values[4], values[5], values[6])
end

"""
Twist rate κ1 of a point (Vlasov path): zero where warping is restrained, the state entry otherwise.
The restrained slot holds the reaction bimoment.
"""
function point_twist_rate(x, model::StaticModel, iPoint::Int)
    icol = model.indices.pointColumns[iPoint] + 6
    return point_condition(model, iPoint).restrainedWarping ? zero(eltype(x)) : x[icol]
end

function point_reaction_bimoment(x, model::StaticModel, iPoint::Int)
    icol = model.indices.pointColumns[iPoint] + 6
    return point_condition(model, iPoint).restrainedWarping ? x[icol] * model.indices.forceScaling : zero(eltype(x))
end

"""
Internal force F and moment M of an element in the deformed frame (physical units). With Vlasov
warping M is the total moment whose first component is M1 - Mω′.
"""
function element_loads(x, model::StaticModel, iElem::Int)
    icol = model.indices.elementColumns[iElem]
    forceScaling = model.indices.forceScaling
    F = SVector(x[icol], x[icol+1], x[icol+2]) * forceScaling
    M = SVector(x[icol+3], x[icol+4], x[icol+5]) * forceScaling
    return F, M
end

element_bimoment(x, model::StaticModel, iElem::Int) = x[model.indices.elementColumns[iElem]+6] * model.indices.forceScaling

"""
Strains γ (engineering shear) and curvatures κ of an element from its loads through the compliance.
Returns (γ, κ, M1SaintVenant). Without warping M1SaintVenant = M[1]; with warping κ1 is the given
independent twist rate and the Saint-Venant torque follows from the partial inversion.
"""
function element_strains(section::BeamSection, F, M, twistRate)
    if has_warping(section)
        reducedLoads = SVector(F[1], F[2], F[3], M[2], M[3])
        reducedStrains = section.reducedCompliance * (reducedLoads - section.torsionCoupling * twistRate)
        γ = SVector(reducedStrains[1], reducedStrains[2], reducedStrains[3])
        κ = SVector(twistRate, reducedStrains[4], reducedStrains[5])
        M1SaintVenant = dot(section.torsionCoupling, reducedStrains) + section.torsionStiffness * twistRate
        return γ, κ, M1SaintVenant
    end
    strains = section.compliance * vcat(F, M)
    return SVector(strains[1], strains[2], strains[3]), SVector(strains[4], strains[5], strains[6]), M[1]
end

# ==============================================================================
#                         Residual (Hodges mixed formulation, GEBT element)
# ==============================================================================
"""
Residual of the static mixed formulation. Element rows: compatibility of displacements
u2 - u1 = Cᵀ Cab L (e1 + γ) - L Cab e1, of rotations θ2 - θ1 = Q⁻¹(θe) Cab L κ, and of the twist rate
κ1,2 - κ1,1 = L Mω / Ŝ55. Point rows: equilibrium of the external loads with the element end
resultants F_end = Cᵀ Cab F and M_end = Cᵀ Cab [M ± ½ L (e1 + γ) × F], with the element weight,
distributed loads, and the bimoment jump Mω′ = M1,SV - M1,total. Loads and prescribed displacements
are multiplied by `loadFactor`. All arithmetic is in eltype(x) so ForwardDiff duals pass through.
"""
function compute_residual(x::AbstractVector, model::StaticModel, loadFactor)
    indices = model.indices
    assembly = model.assembly
    T = promote_type(eltype(x), typeof(loadFactor), eltype(assembly.nodeCoords))
    residual = zeros(T, indices.nStates)
    forceScaling = indices.forceScaling
    warping = indices.warping

    # Point equilibrium: external loads first, element resultants added below
    for p in 1:n_nodes(assembly)
        _, θ = point_displacement(x, model, p, loadFactor)
        C = wiener_milenkovic_rotation(θ)
        Fext, Mext = point_loads(x, model, p, C, loadFactor)
        irow = indices.pointRows[p]
        residual[irow:irow+2] .= Fext
        residual[irow+3:irow+5] .= Mext
        if warping
            residual[irow+6] = point_reaction_bimoment(x, model, p)
        end
    end

    for e in 1:n_elements(assembly)
        n1, n2 = assembly.elemConn[e, 1], assembly.elemConn[e, 2]
        section = assembly.sections[e]
        L = assembly.lengths[e]
        Cab = assembly.elementFrames[e]
        u1, θ1 = point_displacement(x, model, n1, loadFactor)
        u2, θ2 = point_displacement(x, model, n2, loadFactor)
        θe = (θ1 + θ2) / 2
        C = wiener_milenkovic_rotation(θe)
        CtCab = transpose(C) * Cab
        F, M = element_loads(x, model, e)
        twistRate = warping ? (point_twist_rate(x, model, n1) + point_twist_rate(x, model, n2)) / 2 : zero(T)
        γ, κ, M1SaintVenant = element_strains(section, F, M, twistRate)

        # Compatibility (element rows)
        irow = indices.elementRows[e]
        Δu = CtCab * (L * (E1 + γ)) - L * (Cab * E1)
        Δθ = rotation_tangent_inverse(θe) * (Cab * (L * κ))
        residual[irow:irow+2] .= (u2 - u1 - Δu)
        residual[irow+3:irow+5] .= (θ2 - θ1 - Δθ)
        if warping
            Mω = element_bimoment(x, model, e)
            residual[irow+6] = point_twist_rate(x, model, n2) - point_twist_rate(x, model, n1) - L * Mω / section.warpingStiffness
        end

        # Element end resultants in global components
        lever = L * (E1 + γ)
        Fend = CtCab * F
        Mstart = CtCab * (M + cross(lever, F) / 2)
        Mend = CtCab * (M - cross(lever, F) / 2)

        # Element weight (dead) and its moment about the reference axis from the mass center offset
        μ = section.mass[1, 1]
        weight = loadFactor * L * μ * model.gravity
        gB = transpose(CtCab) * model.gravity # gravity in the deformed frame
        offsetMoment = loadFactor * L * (CtCab * mass_offset_moment(section, gB))

        # Distributed loads
        f1 = f2 = m1 = m2 = SVector{3,T}(zeros(T, 3))
        if haskey(model.distributedLoads, e)
            load = model.distributedLoads[e]
            f1 = loadFactor * (load.f1 + CtCab * load.f1Follower)
            f2 = loadFactor * (load.f2 + CtCab * load.f2Follower)
            m1 = loadFactor * (load.m1 + CtCab * load.m1Follower)
            m2 = loadFactor * (load.m2 + CtCab * load.m2Follower)
        end

        # The element pulls its start node with +F and its end node with -F
        irow1 = indices.pointRows[n1]
        irow2 = indices.pointRows[n2]
        residual[irow1:irow1+2] .+= Fend + weight / 2 + f1
        residual[irow1+3:irow1+5] .+= Mstart + offsetMoment / 2 + m1
        residual[irow2:irow2+2] .+= -Fend + weight / 2 + f2
        residual[irow2+3:irow2+5] .+= -Mend + offsetMoment / 2 + m2
        if warping
            Mω = element_bimoment(x, model, e)
            jump = L * (M1SaintVenant - M[1]) / 2 # ½ L Mω′
            residual[irow1+6] += Mω - jump
            residual[irow2+6] -= Mω + jump
        end
    end

    # Scale equilibrium rows so they are strain-like, as the internal loads are
    for p in 1:n_nodes(assembly)
        irow = indices.pointRows[p]
        residual[irow:irow+indices.statesPerPoint-1] ./= forceScaling
    end
    return residual
end

"""
Moment per unit length about the reference axis from the section weight acting at the mass center,
in the deformed frame: x̄ × (μ g_B) with x̄ = (0, x̄2, x̄3) read from the mass matrix (M[3,4] = μ x̄2,
M[2,4] = -μ x̄3).
"""
function mass_offset_moment(section::BeamSection, gB)
    μx̄2 = section.mass[3, 4]
    μx̄3 = -section.mass[2, 4]
    offset = SVector(zero(μx̄2), μx̄2, μx̄3) # already multiplied by μ
    return cross(offset, gB)
end

"""
Jacobian of the residual by forward-mode AD ("FAD") or central finite differences ("FiDi").
"""
function compute_jacobian(x::AbstractVector, model::StaticModel, loadFactor; mode="FAD")
    if mode == "FAD"
        return ForwardDiff.jacobian(y -> compute_residual(y, model, loadFactor), x)
    elseif mode == "FiDi"
        n = length(x)
        J = zeros(n, n)
        h = 1e-6
        for j in 1:n
            xp = copy(x)
            xm = copy(x)
            xp[j] += h
            xm[j] -= h
            J[:, j] = (compute_residual(xp, model, loadFactor) - compute_residual(xm, model, loadFactor)) / (2h)
        end
        return J
    end
    error("Unknown Jacobian mode $(mode); use \"FAD\" or \"FiDi\"")
end

# ==============================================================================
#                         Solvers
# ==============================================================================
"""
Newton iterations with a backtracking line search on the residual norm. Returns
(x, residualNorm, iterations, converged).
"""
function newton_solve(residual::Function, jacobian::Function, x0::AbstractVector; tol=1e-9, maxIters=30, verbose=false)
    x = copy(x0)
    r = residual(x)
    rNorm = norm(r)
    iterations = 0
    while rNorm > tol && iterations < maxIters
        J = jacobian(x)
        Δx = -(J \ r)
        step = 1.0
        xTrial = x + Δx
        rTrial = residual(xTrial)
        rTrialNorm = norm(rTrial)
        nBacktracks = 0
        while (!isfinite(rTrialNorm) || rTrialNorm > (1 - 1e-4 * step) * rNorm) && nBacktracks < 10
            step /= 2
            xTrial = x + step * Δx
            rTrial = residual(xTrial)
            rTrialNorm = norm(rTrial)
            nBacktracks += 1
        end
        x = xTrial
        r = rTrial
        rNorm = rTrialNorm
        iterations += 1
        if verbose
            println("  Newton iteration $(iterations): |r| = $(rNorm), step = $(step)")
        end
        if !isfinite(rNorm)
            return x, rNorm, iterations, false
        end
    end
    return x, rNorm, iterations, rNorm <= tol
end

"""
Solution container. Nodal arrays are [nNodes, 3], element arrays [nElem, 3]. Loads and strains are in
the deformed element frame and physical units; `moments` is the total moment (M1 - Mω′ with warping),
`saintVenantTorque` the Saint-Venant part. `rotations` are the rotation tensors R = Cᵀ in global
components; `twist` is the twist angle about the deformed axis.
"""
struct StaticSolution{T}
    x::Vector{T}
    displacements::Matrix{T}
    rotationParameters::Matrix{T}
    rotations::Vector{SMatrix{3,3,T,9}}
    twist::Vector{T}
    positions::Matrix{T}
    forces::Matrix{T}
    moments::Matrix{T}
    saintVenantTorque::Vector{T}
    strains::Matrix{T}
    curvatures::Matrix{T}
    bimoment::Vector{T}
    twistRate::Vector{T}
    reactions::Matrix{T} # [nNodes, 6] global reaction loads at prescribed components (zero elsewhere)
    loadFactors::Vector{T}
    stateHistory::Vector{Vector{T}} # converged state after every load step
    iterations::Vector{Int}
    residualNorms::Vector{T}
end

"""
Nodal displacements [nNodes, 3] of a state at a load factor.
"""
function nodal_displacements(x::AbstractVector, model::StaticModel, loadFactor)
    nNodes = n_nodes(model.assembly)
    displacements = zeros(eltype(x), nNodes, 3)
    for p in 1:nNodes
        u, _ = point_displacement(x, model, p, loadFactor)
        displacements[p, :] .= u
    end
    return displacements
end

"""
Post-process a converged state into a StaticSolution.
"""
function build_solution(x::AbstractVector, model::StaticModel, loadFactor, loadFactors, stateHistory, iterations, residualNorms)
    assembly = model.assembly
    T = eltype(x)
    nNodes = n_nodes(assembly)
    nElem = n_elements(assembly)
    displacements = zeros(T, nNodes, 3)
    rotationParameters = zeros(T, nNodes, 3)
    rotations = SMatrix{3,3,T,9}[]
    twist = zeros(T, nNodes)
    twistRate = zeros(T, nNodes)
    reactions = zeros(T, nNodes, 6)
    for p in 1:nNodes
        u, θ = point_displacement(x, model, p, loadFactor)
        displacements[p, :] .= u
        rotationParameters[p, :] .= θ
        C = wiener_milenkovic_rotation(θ)
        push!(rotations, transpose(C))
        Cab = assembly.elementFrames[min(p, nElem)]
        twist[p] = twist_angle(transpose(Cab) * transpose(C) * Cab)
        condition = point_condition(model, p)
        Fext, Mext = point_loads(x, model, p, C, loadFactor)
        for i in 1:6
            if condition.prescribedDisplacement[i]
                reactions[p, i] = i <= 3 ? Fext[i] : Mext[i-3]
            end
        end
        if model.indices.warping
            twistRate[p] = point_twist_rate(x, model, p)
        end
    end
    forces = zeros(T, nElem, 3)
    moments = zeros(T, nElem, 3)
    saintVenantTorque = zeros(T, nElem)
    strains = zeros(T, nElem, 3)
    curvatures = zeros(T, nElem, 3)
    bimoment = zeros(T, nElem)
    for e in 1:nElem
        F, M = element_loads(x, model, e)
        n1, n2 = assembly.elemConn[e, 1], assembly.elemConn[e, 2]
        rate = model.indices.warping ? (twistRate[n1] + twistRate[n2]) / 2 : zero(T)
        γ, κ, M1SaintVenant = element_strains(assembly.sections[e], F, M, rate)
        forces[e, :] .= F
        moments[e, :] .= M
        saintVenantTorque[e] = M1SaintVenant
        strains[e, :] .= γ
        curvatures[e, :] .= κ
        if model.indices.warping
            bimoment[e] = element_bimoment(x, model, e)
        end
    end
    positions = assembly.nodeCoords + displacements
    return StaticSolution{T}(Vector{T}(x), displacements, rotationParameters, rotations, twist, positions, forces, moments,
        saintVenantTorque, strains, curvatures, bimoment, twistRate, reactions, Vector{T}(loadFactors),
        [Vector{T}(state) for state in stateHistory], Vector{Int}(iterations), Vector{T}(residualNorms))
end

"""
Nonlinear static solution by Newton iterations with load stepping. The load factor goes from 0 to 1
in `nLoadSteps` increments (halved on failure, up to `maxHalvings` times), each step warm-started
from the previous converged state.
"""
function solve_static(model::StaticModel; nLoadSteps::Int=10, tol=1e-9, maxIters=30, mode="FAD", maxHalvings::Int=6, verbose=false,
    initialState=nothing)
    x = initialState === nothing ? zeros(model.indices.nStates) : copy(initialState)
    λ = 0.0
    Δλ = 1.0 / nLoadSteps
    loadFactors = Float64[]
    stateHistory = Vector{Float64}[]
    iterations = Int[]
    residualNorms = Float64[]
    nHalvings = 0
    while λ < 1 - 1e-12
        λTrial = min(λ + Δλ, 1.0)
        residual = y -> compute_residual(y, model, λTrial)
        jacobian = y -> compute_jacobian(y, model, λTrial; mode=mode)
        xTrial, rNorm, nIter, converged = newton_solve(residual, jacobian, x; tol=tol, maxIters=maxIters, verbose=verbose)
        if converged
            x = xTrial
            λ = λTrial
            push!(loadFactors, λ)
            push!(stateHistory, copy(x))
            push!(iterations, nIter)
            push!(residualNorms, rNorm)
            if verbose
                println("Load step λ = $(λ) converged in $(nIter) iterations")
            end
        else
            nHalvings += 1
            if nHalvings > maxHalvings
                error("Load stepping failed at λ = $(λTrial) after $(maxHalvings) halvings (|r| = $(rNorm))")
            end
            Δλ /= 2
            if verbose
                println("Load step λ = $(λTrial) failed; halving to Δλ = $(Δλ)")
            end
        end
    end
    return build_solution(x, model, 1.0, loadFactors, stateHistory, iterations, residualNorms)
end

"""
Linearized solution: one Newton step from the undeformed state at full load, which is the exact
linear (small displacement) beam solution of the same section model.
"""
function linear_solve(model::StaticModel; mode="FAD")
    x0 = zeros(model.indices.nStates)
    r0 = compute_residual(x0, model, 1.0)
    J0 = compute_jacobian(x0, model, 1.0; mode=mode)
    x = -(J0 \ r0)
    rNorm = norm(compute_residual(x, model, 1.0))
    return build_solution(x, model, 1.0, [1.0], [x], [1], [rNorm])
end

"""
External work of the dead point forces along the recorded load path (trapezoidal rule over the load
steps, so use enough steps for a nonlinear path) and the internal strain energy
½ ∫ (γᵀF + κᵀM_SV + Mω²/Ŝ55) dx. Dead moments, distributed loads and gravity are not included in the
external work, so use this balance with dead point forces only.
"""
function work_energy_balance(solution::StaticSolution, model::StaticModel)
    externalWork = 0.0
    previousFactor = 0.0
    previousDisplacements = zeros(n_nodes(model.assembly), 3)
    for (λ, state) in zip(solution.loadFactors, solution.stateHistory)
        displacements = nodal_displacements(state, model, λ)
        for (p, condition) in model.conditions
            F = SVector(condition.deadLoad[1], condition.deadLoad[2], condition.deadLoad[3])
            externalWork += (λ + previousFactor) / 2 * dot(F, displacements[p, :] - previousDisplacements[p, :])
        end
        previousFactor = λ
        previousDisplacements = displacements
    end
    internalEnergy = 0.0
    assembly = model.assembly
    for e in 1:n_elements(assembly)
        L = assembly.lengths[e]
        γ = solution.strains[e, :]
        κ = solution.curvatures[e, :]
        F = solution.forces[e, :]
        M = copy(solution.moments[e, :])
        M[1] = solution.saintVenantTorque[e]
        internalEnergy += L * (dot(γ, F) + dot(κ, M)) / 2
        section = assembly.sections[e]
        if has_warping(section)
            internalEnergy += L * solution.bimoment[e]^2 / section.warpingStiffness / 2
        end
    end
    return externalWork, internalEnergy
end

"""
State vector in the COMP2 layout of FEMMethods (9 per node: u, v, w in global x, y, z, then the
global rotation vector components, then zeros for the derivative DOFs). For a beam along +y this puts
the twist in ΘIND and the bending slopes -w′ and v′ in ΦIND and ΨIND.
"""
function to_fem_states(solution::StaticSolution)
    nNodes = size(solution.displacements, 1)
    states = zeros(eltype(solution.displacements), 9 * nNodes)
    for p in 1:nNodes
        states[9*(p-1)+1:9*(p-1)+3] .= solution.displacements[p, :]
        R = solution.rotations[p]
        cosφ = clamp((tr(R) - 1) / 2, -1.0, 1.0)
        φ = acos(cosφ)
        if φ > 1e-12
            axial = SVector(R[3, 2] - R[2, 3], R[1, 3] - R[3, 1], R[2, 1] - R[1, 2]) / (2 * sin(φ))
            states[9*(p-1)+4:9*(p-1)+6] .= φ * axial
        end
    end
    return states
end

end # module StrainBeam
