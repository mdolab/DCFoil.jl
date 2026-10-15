"""
@File    :   VAM.jl
@Time    :   2026-09-23
@Author  :   Galen Ng
@Desc    :   Variational asymptotic method (VAM) cross-sectional analysis of prismatic beams.
             Finite-element dimensional reduction of the 3D elasticity problem over the cross
             section following Hodges (2006) "Nonlinear Composite Beam Theory", Ch. 3 Sec. IV and
             Ch. 4 Sec. II, and Yu et al. (2005) for the generalized Vlasov model.

             Outputs: classical 4x4 stiffness, generalized Timoshenko 6x6 stiffness with shear
             center, generalized Vlasov 5x5 stiffness about the shear center, and the 6x6 mass matrix.

             Beam frame: x1 along the beam, (x2, x3) in the cross section.
             Strain ordering (Hodges eq. 3.95): Γ = [Γ11, 2Γ12, 2Γ13, Γ22, 2Γ23, Γ33]
             Classical strain measures: ε̄ = [γ̄11, κ̄1, κ̄2, κ̄3]
"""

module VAM

using LinearAlgebra
using FLOWMath: linear

export Material3D, isotropic_material, transversely_isotropic_material
export CrossSectionMesh, SectionLayup, CrossSectionProperties, uniform_layup
export build_material_stiffness, ply_rotation, rotate_material_stiffness
export cosine_stations, interpolate_airfoil_surfaces, generate_structured_quad_mesh, translate_mesh
export assemble_section_matrices, assemble_origin_dependent_matrices
export factorize_bordered_system, solve_constrained_warping
export compute_classical_stiffness, compute_second_order_energy, transform_to_timoshenko
export locate_shear_center, locate_tension_center, condense_axial_stiffness
export compute_vlasov_stiffness, compute_mass_properties
export analyze_cross_section, check_timoshenko_consistency

# ==============================================================================
#                         Materials
# ==============================================================================
"""
Orthotropic material with engineering constants in the material frame (1 = along fiber).
Poisson ratios follow the convention νij = -εj/εi under uniaxial stress along i.
"""
struct Material3D{T}
    E1::T
    E2::T
    E3::T
    G12::T
    G13::T
    G23::T
    ν12::T
    ν13::T
    ν23::T
    ρ::T
end

function Material3D(; E1, E2, E3, G12, G13, G23, ν12, ν13, ν23, ρ)
    return Material3D(promote(E1, E2, E3, G12, G13, G23, ν12, ν13, ν23, ρ)...)
end

function isotropic_material(E, ν, ρ)
    G = E / (2 * (1 + ν))
    return Material3D(E1=E, E2=E, E3=E, G12=G, G13=G, G23=G, ν12=ν, ν13=ν, ν23=ν, ρ=ρ)
end

"""
Transversely isotropic material from the four in-plane constants used elsewhere in DCFoil.
The 2-3 plane is isotropic: E3 = E2, G13 = G12, ν13 = ν12, G23 = E2 / (2 (1 + ν23)).
ν23 is not measured for the materials in MaterialLibrary; 0.4 is a typical value for
unidirectional carbon/epoxy.
"""
function transversely_isotropic_material(E1, E2, G12, ν12, ρ; ν23=0.4)
    return Material3D(E1=E1, E2=E2, E3=E2, G12=G12, G13=G12, G23=E2 / (2 * (1 + ν23)),
        ν12=ν12, ν13=ν12, ν23=ν23, ρ=ρ)
end

# Hodges ordering [11, 12, 13, 22, 23, 33] as a permutation of standard Voigt [11, 22, 33, 23, 13, 12]
const VOIGT_TO_HODGES = [1, 6, 5, 2, 4, 3]
# Index pairs (i, j) of each Hodges component
const HODGES_PAIRS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
# Map from (i, j) to the Hodges component index
const PAIR_TO_HODGES = [1 2 3; 2 4 5; 3 5 6]

"""
6x6 material stiffness in the material frame with the Hodges strain ordering
[Γ11, 2Γ12, 2Γ13, Γ22, 2Γ23, Γ33] (engineering shear strains), so that σ = D Γ
with σ = [σ11, σ12, σ13, σ22, σ23, σ33].
"""
function build_material_stiffness(material::Material3D)
    (; E1, E2, E3, G12, G13, G23, ν12, ν13, ν23) = material
    # Compliance in standard Voigt order [11, 22, 33, 23, 13, 12]
    compliance = [
        1/E1 -ν12/E1 -ν13/E1 0 0 0
        -ν12/E1 1/E2 -ν23/E2 0 0 0
        -ν13/E1 -ν23/E2 1/E3 0 0 0
        0 0 0 1/G23 0 0
        0 0 0 0 1/G13 0
        0 0 0 0 0 1/G12
    ]
    stiffnessVoigt = inv(compliance)
    return stiffnessVoigt[VOIGT_TO_HODGES, VOIGT_TO_HODGES]
end

"""
Direction cosine matrix whose columns are the material axes expressed in the beam frame for a
fiber angle θ measured from x1 toward x2 (rotation about x3, plies parallel to the x1-x2 plane).
"""
function ply_rotation(θ)
    c = cos(θ)
    s = sin(θ)
    return [
        c -s 0;
        s c 0;
        0 0 1
    ]
end

"""
Rotate a 6x6 stiffness (Hodges ordering, engineering strains) from the material frame to the
beam frame using the fourth-order tensor transformation 
    C'_ijkl = R_ia R_jb R_kc R_ld C_abcd.
The columns of R are the material axes in the beam frame. Because the strains are engineering
strains, D_IJ = C_ijkl with no extra factors in either direction.
"""
function rotate_material_stiffness(stiffness::AbstractMatrix, rotation::AbstractMatrix)
    T = promote_type(eltype(stiffness), eltype(rotation))
    tensor = zeros(T, 3, 3, 3, 3)
    for (I, (i, j)) in enumerate(HODGES_PAIRS), (J, (k, l)) in enumerate(HODGES_PAIRS)
        value = stiffness[I, J]
        tensor[i, j, k, l] = value
        tensor[j, i, k, l] = value
        tensor[i, j, l, k] = value
        tensor[j, i, l, k] = value
    end
    rotated = zeros(T, 3, 3, 3, 3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        acc = zero(T)
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
            acc += rotation[i, a] * rotation[j, b] * rotation[k, c] * rotation[l, d] * tensor[a, b, c, d]
        end
        rotated[i, j, k, l] = acc
    end
    result = zeros(T, 6, 6)
    for (I, (i, j)) in enumerate(HODGES_PAIRS), (J, (k, l)) in enumerate(HODGES_PAIRS)
        result[I, J] = rotated[i, j, k, l]
    end
    return result
end

# ==============================================================================
#                         Mesh
# ==============================================================================
"""
Quadrilateral mesh of the cross section in the (x2, x3) plane.
Elements are tensor-product Lagrange quadrilaterals of the given order (1 = Q4, 2 = Q9) with
local node (a, b) stored at column a + (order + 1) (b - 1), a along x2 and b along x3.
Connectivity is counterclockwise so the Jacobian determinant is positive.
"""
struct CrossSectionMesh{T}
    nodeCoords::Matrix{T} # [nNodes, 2]
    elemConn::Matrix{Int} # [nElem, nodesPerElem]
    order::Int
end

"""
Material index and fiber angle [rad] for every element. Kept separate from the geometry so a
complex-stepped fiber angle does not force complex nodal coordinates.
"""
struct SectionLayup{T}
    elemMaterial::Vector{Int}
    elemPlyAngle::Vector{T}
end

function uniform_layup(mesh::CrossSectionMesh, materialIndex::Int, plyAngle)
    nElem = size(mesh.elemConn, 1)
    return SectionLayup(fill(materialIndex, nElem), fill(plyAngle, nElem))
end

"""
Cosine-spaced stations clustering toward both ends.
"""
function cosine_stations(xStart, xEnd, nStations::Int)
    return [xStart + (xEnd - xStart) * (1 - cos(θ)) / 2 for θ in range(0, π, length=nStations)]
end

function deduplicate_abscissae(x::AbstractVector, y::AbstractVector)
    keep = [1]
    for ii in 2:length(x)
        if x[ii] != x[keep[end]]
            push!(keep, ii)
        end
    end
    return x[keep], y[keep]
end

"""
Split airfoil coordinates ordered counterclockwise from the trailing edge (upper surface TE→LE,
then lower surface LE→TE, as produced by generate_naca4dig) and linearly interpolate the upper
and lower surfaces onto the requested chordwise stations.

Returns (zLower, zUpper) at the stations.
"""
function interpolate_airfoil_surfaces(airfoilCoords::AbstractMatrix, xStations::AbstractVector)
    iLE = argmin(airfoilCoords[:, 1])
    upper = airfoilCoords[iLE:-1:1, :] # reverse to ascending x
    lower = airfoilCoords[iLE:end, :]
    xUpper, zUpper = deduplicate_abscissae(upper[:, 1], upper[:, 2])
    xLower, zLower = deduplicate_abscissae(lower[:, 1], lower[:, 2])
    xMin = max(xUpper[1], xLower[1])
    xMax = min(xUpper[end], xLower[end])
    xClamped = clamp.(xStations, xMin, xMax)
    zUpperStations = [linear(xUpper, zUpper, x) for x in xClamped]
    zLowerStations = [linear(xLower, zLower, x) for x in xClamped]
    return zLowerStations, zUpperStations
end

"""
Structured mesh between a lower and an upper surface sampled at the given stations.

x2Stations must be ascending and include the mid-stations for quadratic elements, so its length is
order * nChord + 1. Columns where the surfaces coincide (a sharp leading or trailing edge) collapse
to a single node, which turns the adjacent quadrilaterals into valid degenerate elements.
"""
function generate_structured_quad_mesh(x2Stations::AbstractVector, zLower::AbstractVector, zUpper::AbstractVector;
    nThick::Int=4, order::Int=2)

    nStations = length(x2Stations)
    if (nStations - 1) % order != 0
        error("Number of stations must be order * nChord + 1, got $(nStations) for order $(order)")
    end
    if !issorted(x2Stations)
        error("x2Stations must be ascending so that the element connectivity is counterclockwise")
    end
    nChord = (nStations - 1) ÷ order
    nThickNodes = order * nThick + 1
    chordExtent = x2Stations[end] - x2Stations[1]
    collapseTolerance = 1e-10 * chordExtent

    T = promote_type(eltype(x2Stations), eltype(zLower), eltype(zUpper))
    nodeId = zeros(Int, nStations, nThickNodes)
    coords = T[]
    nNodes = 0
    for ii in 1:nStations
        thickness = zUpper[ii] - zLower[ii]
        isCollapsed = real(thickness) <= collapseTolerance
        for jj in 1:nThickNodes
            if isCollapsed && jj > 1
                nodeId[ii, jj] = nodeId[ii, 1]
                continue
            end
            nNodes += 1
            nodeId[ii, jj] = nNodes
            fraction = (jj - 1) / (nThickNodes - 1)
            push!(coords, x2Stations[ii])
            push!(coords, zLower[ii] + thickness * fraction)
        end
    end
    nodeCoords = permutedims(reshape(coords, 2, nNodes))

    nodesPerElem = (order + 1)^2
    elemConn = zeros(Int, nChord * nThick, nodesPerElem)
    iElem = 0
    for jElem in 1:nThick, iChord in 1:nChord
        iElem += 1
        for b in 0:order, a in 0:order
            local_index = a + 1 + (order + 1) * b
            elemConn[iElem, local_index] = nodeId[order*(iChord-1)+a+1, order*(jElem-1)+b+1]
        end
    end
    return CrossSectionMesh(nodeCoords, elemConn, order)
end

function translate_mesh(mesh::CrossSectionMesh, offset::AbstractVector)
    shifted = copy(mesh.nodeCoords)
    shifted[:, 1] .+= offset[1]
    shifted[:, 2] .+= offset[2]
    return CrossSectionMesh(shifted, mesh.elemConn, mesh.order)
end

# ==============================================================================
#                         Shape functions and quadrature
# ==============================================================================
function gauss_legendre(nPoints::Int)
    if nPoints == 2
        return [-1 / sqrt(3), 1 / sqrt(3)], [1.0, 1.0]
    elseif nPoints == 3
        return [-sqrt(3 / 5), 0.0, sqrt(3 / 5)], [5 / 9, 8 / 9, 5 / 9]
    end
    error("Gauss-Legendre rule with $(nPoints) points is not tabulated")
end

"""
1D Lagrange basis values and derivatives at ξ ∈ [-1, 1] on equispaced nodes.
"""
function lagrange_basis(order::Int, ξ)
    if order == 1
        values = [(1 - ξ) / 2, (1 + ξ) / 2]
        derivatives = [-1 / 2, 1 / 2]
    elseif order == 2
        values = [ξ * (ξ - 1) / 2, 1 - ξ^2, ξ * (ξ + 1) / 2]
        derivatives = [(2ξ - 1) / 2, -2ξ, (2ξ + 1) / 2]
    else
        error("Lagrange basis of order $(order) is not implemented")
    end
    return values, derivatives
end

"""
Tensor-product quadrilateral shape functions and their parametric derivatives.
"""
function quad_shape_functions(order::Int, ξ, η)
    lξ, dlξ = lagrange_basis(order, ξ)
    lη, dlη = lagrange_basis(order, η)
    nen = (order + 1)^2
    N = zeros(nen)
    dNdξ = zeros(nen)
    dNdη = zeros(nen)
    for b in 1:(order+1), a in 1:(order+1)
        k = a + (order + 1) * (b - 1)
        N[k] = lξ[a] * lη[b]
        dNdξ[k] = dlξ[a] * lη[b]
        dNdη[k] = lξ[a] * dlη[b]
    end
    return N, dNdξ, dNdη
end

struct QuadraturePoint
    N::Vector{Float64}
    dNdξ::Vector{Float64}
    dNdη::Vector{Float64}
    weight::Float64
end

function build_quadrature(order::Int)
    points, weights = gauss_legendre(order + 1)
    rule = QuadraturePoint[]
    for (η, wη) in zip(points, weights), (ξ, wξ) in zip(points, weights)
        N, dNdξ, dNdη = quad_shape_functions(order, ξ, η)
        push!(rule, QuadraturePoint(N, dNdξ, dNdη, wξ * wη))
    end
    return rule
end

"""
Physical coordinates, shape function gradients, and Jacobian determinant at a quadrature point.
"""
function evaluate_mapping(point::QuadraturePoint, elemCoords::AbstractMatrix)
    x2 = dot(point.N, elemCoords[:, 1])
    x3 = dot(point.N, elemCoords[:, 2])
    J11 = dot(point.dNdξ, elemCoords[:, 1])
    J12 = dot(point.dNdξ, elemCoords[:, 2])
    J21 = dot(point.dNdη, elemCoords[:, 1])
    J22 = dot(point.dNdη, elemCoords[:, 2])
    detJ = J11 * J22 - J12 * J21
    if real(detJ) <= 0
        error("Non-positive Jacobian determinant $(detJ) in the cross-section mesh; check element orientation")
    end
    # Inverse Jacobian applied to parametric gradients
    dNdx2 = (J22 .* point.dNdξ .- J12 .* point.dNdη) ./ detJ
    dNdx3 = (-J21 .* point.dNdξ .+ J11 .* point.dNdη) ./ detJ
    return x2, x3, dNdx2, dNdx3, detJ
end

# ==============================================================================
#                         Strain operators (Hodges eqs. 3.96-3.99, 3.102)
# ==============================================================================
"""
Γ_a S: strain from in-plane derivatives of the warping (6 x 3 nen).
"""
function warping_gradient_operator(dNdx2, dNdx3)
    nen = length(dNdx2)
    T = promote_type(eltype(dNdx2), eltype(dNdx3))
    B = zeros(T, 6, 3 * nen)
    for k in 1:nen
        c = 3 * (k - 1)
        B[2, c+1] = dNdx2[k]
        B[3, c+1] = dNdx3[k]
        B[4, c+2] = dNdx2[k]
        B[5, c+2] = dNdx3[k]
        B[5, c+3] = dNdx2[k]
        B[6, c+3] = dNdx3[k]
    end
    return B
end

"""
Γ_ℓ S: strain from the axial derivative of the warping (6 x 3 nen), nonzero in the first three rows.
"""
function warping_axial_operator(N)
    nen = length(N)
    B = zeros(eltype(N), 6, 3 * nen)
    for k in 1:nen
        c = 3 * (k - 1)
        B[1, c+1] = N[k]
        B[2, c+2] = N[k]
        B[3, c+3] = N[k]
    end
    return B
end

"""
Γ_ε: strain from the classical 1D strain measures [γ̄11, κ̄1, κ̄2, κ̄3] (6 x 4).
"""
function classical_strain_operator(x2, x3)
    T = promote_type(typeof(x2), typeof(x3))
    Γε = zeros(T, 6, 4)
    Γε[1, 1] = 1
    Γε[1, 3] = x3
    Γε[1, 4] = -x2
    Γε[2, 2] = -x3
    Γε[3, 2] = x2
    return Γε
end

"""
Kernel of Γ_a evaluated at a point: rigid translations and rotation about x1 (3 x 4).
"""
function warping_kernel(x2, x3)
    T = promote_type(typeof(x2), typeof(x3))
    ψ = zeros(T, 3, 4)
    ψ[1, 1] = 1
    ψ[2, 2] = 1
    ψ[3, 3] = 1
    ψ[2, 4] = -x3
    ψ[3, 4] = x2
    return ψ
end

# ==============================================================================
#                         Assembly (Hodges eq. 4.41)
# ==============================================================================
function element_dofs(conn::AbstractVector{Int})
    dofs = zeros(Int, 3 * length(conn))
    for (k, node) in enumerate(conn)
        dofs[3k-2] = 3 * node - 2
        dofs[3k-1] = 3 * node - 1
        dofs[3k] = 3 * node
    end
    return dofs
end

"""
Scatter-add an element matrix with scalar loops so repeated node ids (collapsed elements)
accumulate correctly.
"""
function scatter_add!(global_matrix::AbstractMatrix, rows::AbstractVector{Int}, cols::AbstractVector{Int}, local_matrix::AbstractMatrix)
    for (q, col) in enumerate(cols), (p, row) in enumerate(rows)
        global_matrix[row, col] += local_matrix[p, q]
    end
    return nothing
end

function scatter_add_rows!(global_matrix::AbstractMatrix, rows::AbstractVector{Int}, local_matrix::AbstractMatrix)
    for q in 1:size(local_matrix, 2), (p, row) in enumerate(rows)
        global_matrix[row, q] += local_matrix[p, q]
    end
    return nothing
end

function element_material_stiffness(layup::SectionLayup, materials::AbstractVector{<:Material3D}, iElem::Int)
    D = build_material_stiffness(materials[layup.elemMaterial[iElem]])
    θ = layup.elemPlyAngle[iElem]
    if θ == 0
        return D
    end
    return rotate_material_stiffness(D, ply_rotation(θ))
end

function result_eltype(mesh::CrossSectionMesh, layup::SectionLayup, materials::AbstractVector{<:Material3D})
    T = promote_type(eltype(mesh.nodeCoords), eltype(layup.elemPlyAngle))
    for material in materials
        T = promote_type(T, typeof(material.E1))
    end
    return T
end

"""
Assemble every cross-sectional matrix of Hodges eq. (4.41) for a prismatic beam plus the
constraint block HΨ = ⟨Sᵀ ψ⟩.

Returns a named tuple (E, Daε, Dal, Dεε, Dlε, Dll, HΨ).
"""
function assemble_section_matrices(mesh::CrossSectionMesh, layup::SectionLayup, materials::AbstractVector{<:Material3D})
    T = result_eltype(mesh, layup, materials)
    nNodes = size(mesh.nodeCoords, 1)
    nDof = 3 * nNodes
    E = zeros(T, nDof, nDof)
    Dal = zeros(T, nDof, nDof)
    Dll = zeros(T, nDof, nDof)
    Daε = zeros(T, nDof, 4)
    Dlε = zeros(T, nDof, 4)
    HΨ = zeros(T, nDof, 4)
    Dεε = zeros(T, 4, 4)

    rule = build_quadrature(mesh.order)
    for iElem in 1:size(mesh.elemConn, 1)
        conn = mesh.elemConn[iElem, :]
        elemCoords = mesh.nodeCoords[conn, :]
        dofs = element_dofs(conn)
        D = element_material_stiffness(layup, materials, iElem)
        nDofElem = length(dofs)
        Ee = zeros(T, nDofElem, nDofElem)
        Dale = zeros(T, nDofElem, nDofElem)
        Dlle = zeros(T, nDofElem, nDofElem)
        Daεe = zeros(T, nDofElem, 4)
        Dlεe = zeros(T, nDofElem, 4)
        HΨe = zeros(T, nDofElem, 4)
        for point in rule
            x2, x3, dNdx2, dNdx3, detJ = evaluate_mapping(point, elemCoords)
            dA = point.weight * detJ
            Ba = warping_gradient_operator(dNdx2, dNdx3)
            Bl = warping_axial_operator(point.N)
            Γε = classical_strain_operator(x2, x3)
            DBa = D * Ba
            DBl = D * Bl
            DΓε = D * Γε
            Ee .+= transpose(Ba) * DBa .* dA
            Dale .+= transpose(Ba) * DBl .* dA
            Dlle .+= transpose(Bl) * DBl .* dA
            Daεe .+= transpose(Ba) * DΓε .* dA
            Dlεe .+= transpose(Bl) * DΓε .* dA
            Dεε .+= transpose(Γε) * DΓε .* dA
            HΨe .+= transpose(Bl[1:3, :]) * warping_kernel(x2, x3) .* dA
        end
        scatter_add!(E, dofs, dofs, Ee)
        scatter_add!(Dal, dofs, dofs, Dale)
        scatter_add!(Dll, dofs, dofs, Dlle)
        scatter_add_rows!(Daε, dofs, Daεe)
        scatter_add_rows!(Dlε, dofs, Dlεe)
        scatter_add_rows!(HΨ, dofs, HΨe)
    end
    return (E=E, Daε=Daε, Dal=Dal, Dεε=Dεε, Dlε=Dlε, Dll=Dll, HΨ=HΨ)
end

"""
Recompute only the matrices that depend on the origin of the cross-sectional coordinates
(those containing Γ_ε). E, Dal, Dll and the span of the constraint block are translation invariant.
"""
function assemble_origin_dependent_matrices(mesh::CrossSectionMesh, layup::SectionLayup, materials::AbstractVector{<:Material3D})
    T = result_eltype(mesh, layup, materials)
    nDof = 3 * size(mesh.nodeCoords, 1)
    Daε = zeros(T, nDof, 4)
    Dlε = zeros(T, nDof, 4)
    Dεε = zeros(T, 4, 4)
    rule = build_quadrature(mesh.order)
    for iElem in 1:size(mesh.elemConn, 1)
        conn = mesh.elemConn[iElem, :]
        elemCoords = mesh.nodeCoords[conn, :]
        dofs = element_dofs(conn)
        D = element_material_stiffness(layup, materials, iElem)
        Daεe = zeros(T, length(dofs), 4)
        Dlεe = zeros(T, length(dofs), 4)
        for point in rule
            x2, x3, dNdx2, dNdx3, detJ = evaluate_mapping(point, elemCoords)
            dA = point.weight * detJ
            Ba = warping_gradient_operator(dNdx2, dNdx3)
            Bl = warping_axial_operator(point.N)
            DΓε = D * classical_strain_operator(x2, x3)
            Daεe .+= transpose(Ba) * DΓε .* dA
            Dlεe .+= transpose(Bl) * DΓε .* dA
            Dεε .+= transpose(classical_strain_operator(x2, x3)) * DΓε .* dA
        end
        scatter_add_rows!(Daε, dofs, Daεe)
        scatter_add_rows!(Dlε, dofs, Dlεe)
    end
    return (Daε=Daε, Dεε=Dεε, Dlε=Dlε)
end

# ==============================================================================
#                         Constrained warping solves (Hodges eqs. 4.45-4.50)
# ==============================================================================
"""
Factorize the bordered system [E, α HΨ; α ΨᵀH, 0] once. The constraint block is scaled by
α = ‖E‖₁ / ‖HΨ‖₁ because E scales with the moduli and HΨ with the element area, so the unscaled
system spans more than sixteen orders of magnitude and partial pivoting fails.
"""
function factorize_bordered_system(E::AbstractMatrix, HΨ::AbstractMatrix)
    n = size(E, 1)
    nConstraints = size(HΨ, 2)
    α = opnorm(E, 1) / opnorm(HΨ, 1)
    T = promote_type(eltype(E), eltype(HΨ), typeof(α))
    bordered = zeros(T, n + nConstraints, n + nConstraints)
    bordered[1:n, 1:n] = E
    bordered[1:n, (n+1):end] = α .* HΨ
    bordered[(n+1):end, 1:n] = α .* transpose(HΨ)
    return (factorization=lu(bordered), n=n, α=α)
end

"""
Minimize ½ VᵀEV + Vᵀ rhs subject to ΨᵀHV = 0 for every column of rhs.
Returns V and the Lagrange multipliers Λ such that E V + rhs + HΨ Λ = 0.
"""
function solve_constrained_warping(borderedSystem, rhs::AbstractMatrix)
    (; factorization, n, α) = borderedSystem
    nConstraints = size(factorization, 1) - n
    T = promote_type(eltype(factorization), eltype(rhs))
    b = zeros(T, n + nConstraints, size(rhs, 2))
    b[1:n, :] = -rhs
    solution = factorization \ b
    V = solution[1:n, :]
    Λ = solution[(n+1):end, :] .* α
    return V, Λ
end

"""
Zeroth-order (classical) solution: warping V̂0 per unit classical strain and the 4x4 stiffness
S̄ = V̂0ᵀ D_aε + D_εε (Hodges eq. 4.51).
"""
function compute_classical_stiffness(sectionMatrices, borderedSystem)
    V0, Λ0 = solve_constrained_warping(borderedSystem, sectionMatrices.Daε)
    residual = sectionMatrices.E * V0 + sectionMatrices.Daε + sectionMatrices.HΨ * Λ0
    residualNorm = norm(residual) / norm(sectionMatrices.Daε)
    if real(residualNorm) > 1e-8
        error("Classical warping solve did not converge: relative residual $(residualNorm)")
    end
    stiffness = transpose(V0) * sectionMatrices.Daε + sectionMatrices.Dεε
    return symmetrize(stiffness), V0
end

symmetrize(A::AbstractMatrix) = (A + transpose(A)) / 2

"""
Second-order asymptotically correct energy 2U = ε̄ᵀAε̄ + 2ε̄ᵀBε̄′ + ε̄′ᵀCε̄′ + 2ε̄ᵀDε̄″
(Hodges eqs. 4.55-4.61 for a prismatic beam). The first-order warping V1S solves
E V1S + D_S = HΨΛ with D_S = D_aℓ V̂0 - D_aℓᵀ V̂0 - D_ℓε (eq. 4.58). The blocks come from
substituting V = V̂0 ε̄ + V1S ε̄′ and V′ = V̂0 ε̄′ + V1S ε̄″ into the discretized energy (eq. 4.40)
and keeping terms through second order.
"""
function compute_second_order_energy(sectionMatrices, borderedSystem, V0::AbstractMatrix)
    (; E, Daε, Dal, Dεε, Dlε, Dll) = sectionMatrices
    DS = Dal * V0 - transpose(Dal) * V0 - Dlε
    V1S, _ = solve_constrained_warping(borderedSystem, DS)

    EV0 = E * V0
    EV1 = E * V1S
    DalV0 = Dal * V0
    DalV1 = Dal * V1S
    A = transpose(V0) * EV0 + 2 * transpose(V0) * Daε + Dεε
    B = transpose(V0) * EV1 + transpose(V0) * DalV0 + transpose(Daε) * V1S + transpose(Dlε) * V0
    C = transpose(V1S) * EV1 + transpose(V1S) * DalV0 + transpose(DalV0) * V1S + transpose(V0) * (Dll * V0)
    D = transpose(V0) * DalV1 + transpose(Dlε) * V1S
    return symmetrize(A), B, symmetrize(C), D, V1S
end

# ==============================================================================
#                         Generalized Timoshenko (Hodges eqs. 4.99, 4.103, 4.104)
# ==============================================================================
const TIMOSHENKO_Q = [0 0; 0 0; 0 -1; 1 0]
const TIMOSHENKO_CLASSICAL_INDICES = [1, 4, 5, 6]
const TIMOSHENKO_SHEAR_INDICES = [2, 3]

"""
Generalized Timoshenko 6x6 stiffness for a prismatic beam in the ordering
[γ11, 2γ12, 2γ13, κ1, κ2, κ3] from the second-order energy matrices A, B, C.
"""
function transform_to_timoshenko(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    AinvQ = A \ TIMOSHENKO_Q
    G = inv(transpose(AinvQ) * C * AinvQ)
    Y = transpose(B) * AinvQ * G
    X = A + Y * (G \ transpose(Y))
    T = promote_type(eltype(X), eltype(Y), eltype(G))
    S6 = zeros(T, 6, 6)
    S6[TIMOSHENKO_CLASSICAL_INDICES, TIMOSHENKO_CLASSICAL_INDICES] = X
    S6[TIMOSHENKO_SHEAR_INDICES, TIMOSHENKO_SHEAR_INDICES] = G
    S6[TIMOSHENKO_CLASSICAL_INDICES, TIMOSHENKO_SHEAR_INDICES] = Y
    S6[TIMOSHENKO_SHEAR_INDICES, TIMOSHENKO_CLASSICAL_INDICES] = transpose(Y)
    return symmetrize(S6)
end

"""
Relative difference between the classical stiffness recovered from the Timoshenko flexibility
(delete the shear rows and columns and invert, Hodges p. 82) and S̄.
"""
function check_timoshenko_consistency(S6::AbstractMatrix, classical::AbstractMatrix)
    Φ = inv(S6)
    recovered = inv(Φ[TIMOSHENKO_CLASSICAL_INDICES, TIMOSHENKO_CLASSICAL_INDICES])
    return norm(recovered - classical) / norm(classical)
end

"""
Generalized shear center (Hodges eq. 4.174) from the 6x6 stiffness.
"""
function locate_shear_center(S6::AbstractMatrix)
    Φ = inv(S6)
    return [-Φ[3, 4] / Φ[4, 4], Φ[2, 4] / Φ[4, 4]]
end

"""
Tension center: the point where an axial force produces no bending curvature, from the classical
flexibility Φ̄ = S̄⁻¹. An axial force F1 at (t2, t3) adds moments M2 = t3 F1 and M3 = -t2 F1.
"""
function locate_tension_center(classical::AbstractMatrix)
    Φ = inv(classical)
    lhs = [-Φ[3, 4] Φ[3, 3]; -Φ[4, 4] Φ[4, 3]]
    rhs = -[Φ[3, 1], Φ[4, 1]]
    return lhs \ rhs
end

"""
Condense out the axial strain (F1 = 0) of a classical 4x4 stiffness, returning the 3x3
torsion/bending block in the ordering [κ1, κ2, κ3]. This is the stiffness seen by a beam element
without extension-bending coupling.
"""
function condense_axial_stiffness(classical::AbstractMatrix)
    return classical[2:4, 2:4] - classical[2:4, 1] * transpose(classical[1, 2:4]) / classical[1, 1]
end

# ==============================================================================
#                         Generalized Vlasov (Hodges Sec. 4.II.H, Yu et al. 2005)
# ==============================================================================
"""
Generalized Vlasov 5x5 stiffness about the shear center in the ordering [γ̄11, κ̄1, κ̄2, κ̄3, κ̄1′]
with forces [F1, M1, M2, M3, Mω]. The origin is moved to the shear center, the second-order energy
2U = ε̄ᵀAε̄ + 2ε̄ᵀBε̄′ + ε̄′ᵀCε̄′ + 2ε̄ᵀDε̄″ is recomputed there, and the derivatives of the classical
strains other than κ̄1′ are neglected together with the second derivative κ̄1″, which leaves
Ŝ = [A, B[:, 2]; B[:, 2]ᵀ, C22]. Keeping C22 alone (rather than integrating the D term by parts)
reproduces the VABS warping stiffness of Hodges (2006) Table 7.9 to 1e-4 and the analytical strip
result E h³ b³ / (144 (1 - ν²)) of eq. (7.81).
"""
function compute_vlasov_stiffness(mesh::CrossSectionMesh, layup::SectionLayup, materials::AbstractVector{<:Material3D},
    sectionMatrices, borderedSystem, shearCenter::AbstractVector)

    shiftedMesh = translate_mesh(mesh, -shearCenter)
    originDependent = assemble_origin_dependent_matrices(shiftedMesh, layup, materials)
    shiftedMatrices = (E=sectionMatrices.E, Daε=originDependent.Daε, Dal=sectionMatrices.Dal,
        Dεε=originDependent.Dεε, Dlε=originDependent.Dlε, Dll=sectionMatrices.Dll, HΨ=sectionMatrices.HΨ)
    _, V0 = compute_classical_stiffness(shiftedMatrices, borderedSystem)
    A, B, C, _, _ = compute_second_order_energy(shiftedMatrices, borderedSystem, V0)
    T = promote_type(eltype(A), eltype(B), eltype(C))
    vlasov = zeros(T, 5, 5)
    vlasov[1:4, 1:4] = A
    vlasov[1:4, 5] = B[:, 2]
    vlasov[5, 1:4] = B[:, 2]
    vlasov[5, 5] = C[2, 2]
    return vlasov
end

# ==============================================================================
#                         Mass and area properties (Hodges eqs. 5.23, 5.104)
# ==============================================================================
"""
Mass per unit length μ, mass center (x̄2, x̄3), mass moments of inertia i2 = ⟨ρ x3²⟩, i3 = ⟨ρ x2²⟩,
i23 = -⟨ρ x2 x3⟩ about the reference axis, the 6x6 mass matrix of Hodges eq. (5.104) in the
ordering [V1, V2, V3, Ω1, Ω2, Ω3], and the area and centroid.
"""
function compute_mass_properties(mesh::CrossSectionMesh, layup::SectionLayup, materials::AbstractVector{<:Material3D})
    T = result_eltype(mesh, layup, materials)
    μ = zero(T)
    μx2 = zero(T)
    μx3 = zero(T)
    i2 = zero(T)
    i3 = zero(T)
    i23 = zero(T)
    area = zero(T)
    areaX2 = zero(T)
    areaX3 = zero(T)
    rule = build_quadrature(mesh.order)
    for iElem in 1:size(mesh.elemConn, 1)
        elemCoords = mesh.nodeCoords[mesh.elemConn[iElem, :], :]
        ρ = materials[layup.elemMaterial[iElem]].ρ
        for point in rule
            x2, x3, _, _, detJ = evaluate_mapping(point, elemCoords)
            dA = point.weight * detJ
            area += dA
            areaX2 += x2 * dA
            areaX3 += x3 * dA
            μ += ρ * dA
            μx2 += ρ * x2 * dA
            μx3 += ρ * x3 * dA
            i2 += ρ * x3^2 * dA
            i3 += ρ * x2^2 * dA
            i23 -= ρ * x2 * x3 * dA
        end
    end
    massCenter = [μx2 / μ, μx3 / μ]
    x̄2, x̄3 = massCenter
    massMatrix = [
        μ 0 0 0 μ*x̄3 -μ*x̄2
        0 μ 0 -μ*x̄3 0 0
        0 0 μ μ*x̄2 0 0
        0 -μ*x̄3 μ*x̄2 i2+i3 0 0
        μ*x̄3 0 0 0 i2 i23
        -μ*x̄2 0 0 0 i23 i3
    ]
    return (massMatrix=massMatrix, massPerLength=μ, massCenter=massCenter, i2=i2, i3=i3, i23=i23,
        area=area, centroid=[areaX2 / area, areaX3 / area])
end

# ==============================================================================
#                         Driver
# ==============================================================================
struct CrossSectionProperties{T}
    stiffnessClassical::Matrix{T} # 4x4, [γ̄11, κ̄1, κ̄2, κ̄3]
    stiffnessTimoshenko::Matrix{T} # 6x6, [γ11, 2γ12, 2γ13, κ1, κ2, κ3]
    stiffnessVlasov::Matrix{T} # 5x5 about the shear center, [γ̄11, κ̄1, κ̄2, κ̄3, κ̄1′]
    massMatrix::Matrix{T} # 6x6, [V1, V2, V3, Ω1, Ω2, Ω3]
    area::T
    centroid::Vector{T}
    massPerLength::T
    massCenter::Vector{T}
    momentsOfInertia::Vector{T} # [i2, i3, i23]
    shearCenter::Vector{T}
    tensionCenter::Vector{T}
    warpingClassical::Matrix{T} # 3 nNodes x 4
end

function check_symmetric_positive_definite(A::AbstractMatrix, name::String)
    asymmetry = norm(A - transpose(A)) / norm(A)
    if real(asymmetry) > 1e-8
        error("$(name) is not symmetric: relative asymmetry $(asymmetry)")
    end
    if !isposdef(Symmetric(real.(A)))
        error("$(name) is not positive definite")
    end
    return nothing
end

"""
Run the full cross-sectional analysis: classical, generalized Timoshenko, generalized Vlasov
(about the shear center), and mass properties.
"""
function analyze_cross_section(mesh::CrossSectionMesh, layup::SectionLayup, materials::AbstractVector{<:Material3D})
    sectionMatrices = assemble_section_matrices(mesh, layup, materials)
    borderedSystem = factorize_bordered_system(sectionMatrices.E, sectionMatrices.HΨ)
    classical, V0 = compute_classical_stiffness(sectionMatrices, borderedSystem)
    A, B, C, _, _ = compute_second_order_energy(sectionMatrices, borderedSystem, V0)
    if real(norm(A - classical) / norm(classical)) > 1e-8
        error("Second-order energy block A does not reproduce the classical stiffness")
    end
    timoshenko = transform_to_timoshenko(A, B, C)
    shearCenter = locate_shear_center(timoshenko)
    tensionCenter = locate_tension_center(classical)
    vlasov = compute_vlasov_stiffness(mesh, layup, materials, sectionMatrices, borderedSystem, shearCenter)
    mass = compute_mass_properties(mesh, layup, materials)

    check_symmetric_positive_definite(classical, "Classical stiffness")
    check_symmetric_positive_definite(timoshenko, "Timoshenko stiffness")
    # The Vlasov coupling column Ŝ[1:4, 5] consists of boundary terms under the Vlasov assumption
    # (γ̄11, κ̄2, κ̄3 slowly varying), so the full 5x5 need not be positive definite; only the
    # warping stiffness Ŝ55 must be.
    if real(vlasov[5, 5]) <= 0
        error("Vlasov warping stiffness Ŝ55 = $(vlasov[5, 5]) is not positive")
    end

    T = promote_type(eltype(classical), eltype(timoshenko), eltype(vlasov), eltype(mass.massMatrix))
    return CrossSectionProperties{T}(classical, timoshenko, vlasov, mass.massMatrix, mass.area, mass.centroid,
        mass.massPerLength, mass.massCenter, [mass.i2, mass.i3, mass.i23], shearCenter, tensionCenter, V0)
end

end # module VAM
