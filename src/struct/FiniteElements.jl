# --- Julia 1.7---
"""
@File    :   FiniteElements.jl
@Time    :   2022/08/04
@Author  :   Galen Ng
@Desc    :   Finite element library
"""

# ==============================================================================
#                         METHODS FOR ELEMENT MATRICES
# ==============================================================================
module LinearBeamElem
"""

    2-noded linear beam element

        o---------o
        1  E,ρₛ,I  2

        {q} = [q₁, q₂, q₃, q₄, q₅, q₆]

    The shape function for the element with the {q} vector) is
        w(x, t) = <something-already-derived-in-textbooks>
    
    You impose the geometric BC's to get the α coeffs since the function must be 'admissible'
        w(x,t) = [N(x)]{q(t)}
    
    This has already been derived so no need to redo getting [N] in the code

"""

function compute_elem_stiff(EIᵉ, GJᵉ, BTᵉ, Sᵉ, lᵉ, abᵉ, elemType="bend-twist", constitutive="isotropic", useTimoshenko=false)
    """
    The internal strain energy of a beam is
        U = some-integral-function-derived-from-energy-principles = 0.5{q(t)}ᵀ[Kᵉ]{q(t)}

    Element stiffness matrix from the strain energies
    Output
    ------
    Kᵉ: Stiffness matrix, size depends on element type

    Inputs
    ------
    EIᵉ : Float64
        bending stiffness of the element [N m²]
    GJᵉ : Float64
        torsional stiffness of the element [N-m²]
    BTᵉ : Float64
        this is Kₛ from the paper (material bend-twist coupling) [N-m²]
    Sᵉ : Float64
        structural warping (cross-sections do not retain shape) [N-m⁴]
    lᵉ : Float64
        length of the element [m]
    abᵉ : Float64
        distance from midchord to EA (+ve if EA aft of midchord) [m]
    elemType : String
        which element stiffness matrix to use
    constitutive : String
        which constitutive model to use (isotropic or orthotropic)
    useTimoshenko : Bool
        whether to use Timoshenko beam theory (default is Euler-Bernoulli), only works for the bending element
    """

    # --- Handy identities ---
    if useTimoshenko
        # NOTE: only used for the bending only elem. ATM
        κ = 5 / 6 # for a rect section according to wikipedia
        ϕ = 12 / lᵉ^2 * (EIᵉ / κ * GA) # don't know what GA is yet
    else
        ϕ = 0
    end
    kb = EIᵉ / (lᵉ^3 * (1 + ϕ))
    kt = GJᵉ / lᵉ

    # --- Constitutive law ---
    if constitutive == "isotropic"
        BTᵉ = 0
    end

    # ************************************************
    #     Diff elem type stiffness matrices
    # ************************************************
    if elemType == "bend"
        k11 = 12
        k12 = 6 * lᵉ
        k13 = -12
        k14 = 6 * lᵉ
        k22 = (4 + ϕ) * lᵉ^2
        k23 = -6 * lᵉ
        k24 = (2 - ϕ) * lᵉ^2
        k33 = 12
        k34 = -6 * lᵉ
        k44 = (4 + ϕ) * lᵉ^2
        Kᵉ = kb * [
            k11 k12 k13 k14
            k12 k22 k23 k24
            k13 k23 k33 k34
            k14 k24 k34 k44
        ]

    elseif elemType == "bend-twist"
        # 6x6 elem stiffness matrix
        # Beam bending element terms
        k11 = kb * 12
        k12 = kb * 6 * lᵉ
        k14 = kb * -12
        k15 = kb * 6 * lᵉ
        k22 = kb * 4 * lᵉ^2
        k24 = kb * -6 * lᵉ
        k25 = kb * 2 * lᵉ^2
        k44 = kb * 12
        k45 = kb * -6 * lᵉ
        k55 = kb * 4 * lᵉ^2
        # Torsion element terms
        k33 = kt
        k36 = -kt
        k66 = kt
        if constitutive == "isotropic"
            # bend-twist coupling
            k23 = 0.0
            k26 = 0.0
            k35 = 0.0
            k56 = 0.0
        elseif constitutive == "orthotropic"
            kBT = BTᵉ / lᵉ
            k23 = -kBT
            k26 = kBT
            k35 = kBT
            k56 = -kBT
        end

        Kᵉ = [
            k11 k12 0.0 k14 k15 0.0 # w₁
            k12 k22 k23 k24 k25 k26 # ϕ₁
            0.0 k23 k33 0.0 k35 k36 # ψ₁
            k14 k24 0.0 k44 k45 0.0
            k15 k25 0.0 k45 k55 k56
            0.0 k26 k36 0.0 k56 k66
        ]

    elseif elemType == "bend-twist-axial"
        # TODO: can do this but only interesting if composite propeller looking at extension-twist coupling
        println("Axial elements not implemented")
    elseif elemType == "BT2" # Higher order beam element
        # 8x8 matrix
        coeff::Float64 = 1 / lᵉ^3
        # row 1
        k11_11::Float64 = 12 * EIᵉ
        k11_12::Float64 = 6 * EIᵉ * lᵉ
        k11_13::Float64 = -12 * abᵉ * EIᵉ
        k11_14::Float64 = -(6 * abᵉ * EIᵉ + BTᵉ * lᵉ) * lᵉ
        # row 2
        k11_22::Float64 = 4 * EIᵉ * lᵉ^2
        k11_23::Float64 = -(6 * abᵉ * EIᵉ - BTᵉ * lᵉ) * lᵉ
        k11_24::Float64 = -0.5 * BTᵉ * lᵉ^3 - 4 * abᵉ * EIᵉ * lᵉ^2
        # row 3
        k11_33::Float64 = 6 * GJᵉ * lᵉ^2 / 5 + 12 * Sᵉ
        k11_34::Float64 = GJᵉ * lᵉ^3 / 10 + 6 * Sᵉ * lᵉ
        # row 4
        k11_44 = (abᵉ * BTᵉ * lᵉ^3) + (2 * GJᵉ * lᵉ^4 / 15) + (4 * Sᵉ * lᵉ^2)
        # --- Block matrices ---
        K11 = coeff * [
            k11_11 k11_12 k11_13 k11_14
            k11_12 k11_22 k11_23 k11_24
            k11_13 k11_23 k11_33 k11_34
            k11_14 k11_24 k11_34 k11_44
        ]
        k12_14 = -(6 * abᵉ * EIᵉ - BTᵉ * lᵉ) * lᵉ
        k12_24 = 0.5 * BTᵉ * lᵉ^3 - 2 * abᵉ * EIᵉ * lᵉ^2
        k12_44 = 2 * Sᵉ * lᵉ^2 - GJᵉ * lᵉ^4 / 30
        k12_32 = -(6 * abᵉ * EIᵉ + BTᵉ * lᵉ) * lᵉ
        k12_42 = -0.5 * BTᵉ * lᵉ^3 - 2 * abᵉ * EIᵉ * lᵉ^2
        K12 = coeff * [
            -k11_11 k11_12 -k11_13 k12_14
            -k11_12 0.5*k11_22 -k11_23 k12_24
            -k11_13 k12_32 -k11_33 k11_34
            -k12_32 k12_42 -k11_34 k12_44
        ]
        k22_24 = 0.5 * BTᵉ * lᵉ^3 - 4 * abᵉ * EIᵉ * lᵉ^2
        k22_44 = -(abᵉ * BTᵉ * lᵉ^3) + (2 * GJᵉ * lᵉ^4 / 15) + (4 * Sᵉ * lᵉ^2)
        K22 = coeff * [
            k11_11 -k11_12 k11_13 -k11_23
            -k11_12 k11_22 -k11_14 k22_24
            k11_13 -k11_14 k11_33 -k11_34
            -k11_23 k22_24 -k11_34 k22_44
        ]
        Ktop = hcat(K11, K12)
        Kbot = hcat(K12', K22)
        Kᵉ = vcat(Ktop, Kbot)
    end

    return Kᵉ
end

function compute_elem_mass(mᵉ, iᵉ, lᵉ, x_αbᵉ, elemType="bend-twist")
    """
    The kinetic energy is
        T = 0.5∫₀ᴸ m (∂w/∂t)² dx = 0.5{q̇(t)}ᵀ[Mᵉ]{q̇(t)}

    Element mass matrix from the kinetic energies
    """

    # --- Handy identities ---
    mb = mᵉ * lᵉ / 420
    mt = iᵉ * lᵉ / 6

    if elemType == "bend"
        m11 = mb * 156
        m12 = mb * 22 * lᵉ
        m13 = mb * 54
        m14 = mb * -13 * lᵉ
        m22 = mb * 4 * lᵉ^2
        m23 = mb * 13 * lᵉ
        m24 = mb * -3 * lᵉ^2
        m33 = mb * 156
        m34 = mb * -22 * lᵉ
        m44 = mb * 4 * lᵉ^2
        Mᵉ = [
            m11 m12 m13 m14
            m12 m22 m23 m24
            m13 m23 m33 m34
            m14 m24 m34 m44
        ]
    elseif elemType == "bend-twist"
        m11 = mb * 156
        m12 = mb * 22 * lᵉ
        m14 = mb * 54
        m15 = mb * -13 * lᵉ
        m22 = mb * 4 * lᵉ^2
        m24 = mb * 13 * lᵉ
        m25 = mb * -3 * lᵉ^2
        m44 = mb * 156
        m45 = mb * -22 * lᵉ
        m55 = mb * 4 * lᵉ^2
        m33 = mt * 2
        m36 = mt
        m66 = mt * 2
        Mᵉ = [
            m11 m12 0.0 m14 m15 0.0
            m12 m22 0.0 m24 m25 0.0
            0.0 0.0 m33 0.0 0.0 m36
            m14 m24 0.0 m44 m45 0.0
            m15 m25 0.0 m45 m55 0.0
            0.0 0.0 m36 0.0 0.0 m66
        ]
    elseif elemType == "BT2"
        # row 1
        m11_11 = 13 * mᵉ * lᵉ / 35
        m11_12 = 11 * mᵉ * lᵉ^2 / 210
        m11_13 = 13 * mᵉ * x_αbᵉ * lᵉ / 35
        m11_14 = 11 * mᵉ * x_αbᵉ * lᵉ^2 / 210
        # row 2
        m11_22 = mᵉ * lᵉ^3 / 105
        m11_24 = mᵉ * x_αbᵉ * lᵉ^3 / 105
        # row 3
        m11_33 = 13 * lᵉ * iᵉ / 35
        m11_34 = 11 * lᵉ^2 * iᵉ / 210
        # row 4
        m11_44 = iᵉ * lᵉ^3 / 105
        # --- Block matrices ---
        M11 = [
            m11_11 m11_12 m11_13 m11_14
            m11_12 m11_22 m11_14 m11_24
            m11_13 m11_14 m11_33 m11_34
            m11_14 m11_24 m11_34 m11_44
        ]
        # row 1
        m12_11 = 9 * mᵉ * lᵉ / 70
        m12_12 = -13 * mᵉ * lᵉ^2 / 420
        m12_13 = 9 * mᵉ * x_αbᵉ * lᵉ / 70
        m12_14 = -13 * mᵉ * x_αbᵉ * lᵉ^2 / 420
        # row 2
        m12_22 = -mᵉ * lᵉ^3 / 140
        m12_23 = 13 * mᵉ * x_αbᵉ * lᵉ^2 / 420
        m12_24 = -mᵉ * x_αbᵉ * lᵉ^3 / 140
        # row 3
        m12_33 = 9 * lᵉ * iᵉ / 70
        m12_34 = -13 * lᵉ^2 * iᵉ / 420
        # row 4
        m12_44 = -iᵉ * lᵉ^3 / 140
        M12 = [
            m12_11 m12_12 m12_13 m12_14
            -m12_12 m12_22 m12_23 m12_24
            m12_13 -m12_23 m12_33 m12_34
            -m12_14 m12_24 -m12_34 m12_44
        ]
        M22 = [
            m11_11 -m11_12 m11_13 -m11_14
            -m11_12 m11_22 -m11_14 m11_24
            m11_13 -m11_14 m11_33 -m11_34
            -m11_14 m11_24 -m11_34 m11_44
        ]
        Mtop = hcat(M11, M12)
        Mbot = hcat(M12', M22)
        Mᵉ = vcat(Mtop, Mbot)
    end

    return Mᵉ
end

end # end module

# ==============================================================================
#                         GENERIC FEM METHODS
# ==============================================================================
module FEMMethods
"""
Module with generic FEM methods
"""

# --- Libraries ---
using LinearAlgebra
using ..LinearBeamElem
include("../solvers/SolverRoutines.jl")
using .SolverRoutines

function make_mesh(nElem::Int64, foil)
    """
    Makes a mesh and element connectivity from root to tip with root as origin

    Outputs
    -------
    mesh
        (nNodes, nDim) array 
    elemConn

    """

    println("Right now only 1D mesh in y dir...")

    mesh = LinRange(0, foil.s, nElem + 1)
    elemConn = Array{Int64}(undef, nElem, 2)
    for ii ∈ 1:nElem
        elemConn[ii, 1] = ii
        elemConn[ii, 2] = ii + 1
    end

    return mesh, elemConn
end

function assemble(coordMat, elemConn, foil, elemType="bend-twist", constitutive="isotropic")
    """
    Generic function to assemble the mass and stiffness matrices

    Inputs
    ------
    coordMat: 2D array of coordinates of nodes
    elemConn: 2D array of element connectivity (nElem x 2)
    """

    # --- Initialize the local DOF vector ---
    if elemType == "bend"
        nnd = 2
    elseif elemType == "bend-twist"
        nnd = 3
    elseif elemType == "BT2"
        nnd = 4
    end
    qLocal = zeros(nnd * 2)

    # --- Initialize matrices ---
    nElem::Int64 = size(elemConn)[1]
    nNodes = nElem + 1
    globalK::Matrix{Float64} = zeros(nnd * (nNodes), nnd * (nNodes))
    globalM::Matrix{Float64} = zeros(nnd * (nNodes), nnd * (nNodes))
    globalF::Vector{Float64} = zeros(nnd * (nNodes))

    # --- Debug printout for initialization ---
    # TODO: probabl a better way to pretty print this
    println("+", "-"^50, "+")
    println("|   Assembling global stiffness and mass matrices  |")
    println("+", "-"^50, "+")
    println("Default 2 nodes per elem, nothing else will work")
    println("Using ", constitutive, " constitutive relations...")
    println(nElem, " elements")
    println(nNodes, " nodes")

    # ************************************************
    #     Element loop
    # ************************************************
    for elemIdx ∈ 1:nElem
        # ---------------------------
        #   Extract element info
        # ---------------------------
        lᵉ::Float64 = norm(coordMat[elemIdx+1] - coordMat[elemIdx], 2) # length of elem
        nVec::Float64 = (coordMat[elemIdx+1] - coordMat[elemIdx]) / lᵉ # unit normal vector
        EIₛ::Float64 = foil.EIₛ[elemIdx]
        GJₛ::Float64 = foil.GJₛ[elemIdx]
        Kₛ::Float64 = foil.Kₛ[elemIdx]
        Sₛ::Float64 = foil.Sₛ[elemIdx]
        ab::Float64 = foil.ab[elemIdx]
        mₛ::Float64 = foil.mₛ[elemIdx]
        iₛ::Float64 = foil.Iₛ[elemIdx]
        x_αb::Float64 = foil.x_αb[elemIdx]

        # ---------------------------
        #   Local stiffness matrix
        # ---------------------------
        kLocal::Matrix{Float64} = LinearBeamElem.compute_elem_stiff(EIₛ, GJₛ, Kₛ, Sₛ, lᵉ, ab, elemType, constitutive)

        # ---------------------------
        #   Local mass matrix
        # ---------------------------
        mLocal::Matrix{Float64} = LinearBeamElem.compute_elem_mass(mₛ, iₛ, lᵉ, x_αb, elemType)

        # ---------------------------
        #   Local force vector
        # ---------------------------
        fLocal::Vector{Float64} = zeros(nnd * 2)

        # ---------------------------
        #   Assemble into global matrices
        # ---------------------------
        for nodeIdx ∈ 1:2 # loop over nodes in element
            for dofIdx ∈ 1:nnd # loop over DOFs in node
                idxRow = ((elemConn[elemIdx, nodeIdx] - 1) * nnd + dofIdx) # idx of global dof (row of global matrix)
                idxRowₑ = (nodeIdx - 1) * nnd + dofIdx

                # --- Assemble RHS ---
                globalF[idxRow] = fLocal[idxRowₑ]

                # --- Assemble LHS ---
                for nodeColIdx ∈ 1:2 # loop over nodes in element
                    for dofColIdx ∈ 1:nnd # loop over DOFs in node
                        idxCol = (elemConn[elemIdx, nodeColIdx] - 1) * nnd + dofColIdx # idx of global dof (col of global matrix)
                        idxColₑ = (nodeColIdx - 1) * nnd + dofColIdx

                        globalK[idxRow, idxCol] += kLocal[idxRowₑ, idxColₑ]
                        globalM[idxRow, idxCol] += mLocal[idxRowₑ, idxColₑ]
                    end
                end
            end
        end
    end

    return globalK, globalM, globalF
end

function get_fixed_nodes(elemType::String, BCCond="clamped")
    """
    Depending on the elemType, return the indices of fixed nodes
    """
    if BCCond == "clamped"
        if elemType == "bend"
            fixedNodes = [1, 2]
        elseif elemType == "bend-twist"
            fixedNodes = [1, 2, 3]
        elseif elemType == "BT2"
            fixedNodes = [1, 2, 3, 4]
        end
    else
        error("BCCond not recognized")
    end

    return fixedNodes
end

function apply_tip_load!(F, elemType, loadType="force")

    if loadType == "force"
        if elemType == "bend-twist"
            F[end-2] = 1.0
        elseif elemType == "BT2"
            F[end-3] = 3000.0 / 2
        end
    elseif loadType == "torque"
        if elemType == "bend-twist"
            F[end] = 1.0
        elseif elemType == "BT2"
            F[end-1] = 1.0
        end
    end
end

function apply_BCs(K, M, F, globalDOFBlankingList)
    """
    Applies BCs for nodal displacements and blanks them
    """

    newK = K[
        setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    ]
    newM = M[
        setdiff(1:end, (globalDOFBlankingList)), setdiff(1:end, (globalDOFBlankingList))
    ]
    newF = F[setdiff(1:end, (globalDOFBlankingList))]

    return newK, newM, newF
end

function put_BC_back(q, elemType::String, BCType="clamped")
    """
    appends the BCs back into the solution
    """

    if BCType == "clamped"
        if elemType == "BT2"
            uSol = vcat([0, 0, 0, 0], q)
        else
            println("Not working")
            exit()
        end
    else
        println("Not working")
    end

    return uSol, length(uSol)
end

function solve_structure(K, M, F)
    """
    Solve the structural system
    """

    println("FEM solve took")
    @time q = K \ F # TODO: should probably replace this with an iterative solver

    return q
end

function compute_modal(K, M, nEig::Int64)
    """
    Compute the eigenvalues (natural frequencies) and eigenvectors (mode shapes) of the in-vacuum system.
    i.e., this is structural dynamics, not hydroelastics.
    """

    # use krylov method to get first few smallest eigenvalues
    # Solve [K]{x} = λ[M]{x} where λ = ω²
    eVals, eVecs = SolverRoutines.compute_eigsolve(K, M, nEig)

    naturalFreqs = sqrt.(eVals) / (2π)

    return naturalFreqs, eVecs
end

end # end module

# module BrickElem
# # TODO: maybe never
# function compute_shapeFuncs(coordMat, ξ, η, ζ, order=1)
#     """
#     TODO
#     """
#     # --- Lagrange poly shape funcs ---
#     Nᵢ = 0.125 * [
#         (1 - ξ) * (1 - η) * (1 - ζ) # node 1 (-1, -1, -1)
#         (1 + ξ) * (1 - η) * (1 - ζ) # node 2 (1, -1, -1)
#         (1 + ξ) * (1 + η) * (1 - ζ) # node 3 (1, 1, -1)
#         (1 - ξ) * (1 + η) * (1 - ζ) # node 4 (-1, 1, -1)
#         (1 - ξ) * (1 - η) * (1 + ζ) # node 5 (-1, -1, 1)
#         (1 + ξ) * (1 - η) * (1 + ζ) # node 6 (1,-1, 1)
#         (1 + ξ) * (1 + η) * (1 + ζ) # node 7 (1, 1, 1)
#         (1 - ξ) * (1 - η) * (1 + ζ) # node 8 (-1, 1, 1)
#     ]
#     return nothing

# end

# end # end module