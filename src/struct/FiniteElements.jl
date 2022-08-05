# --- Julia 1.7---
"""
@File    :   FiniteElements.jl
@Time    :   2022/08/04
@Author  :   Galen Ng
@Desc    :   Finite element library
"""

module FEMMethods
function assemble()
    return nothing
end
end # end module


module BeamElem
"""
2-noded beam element

    o---------o
    1  E,ρₛ,I  2

    {q} = [q₁, q₂, q₃, q₄, q₅, q₆]

The shape function for the element w/ 6 total DOF (in the {q} vector) is
    w(x, t) = α₀(t) + α₁(t) x + α₂(t) x² + α₃(t) x³ + α₄(t) x⁴ + α₅(t) x⁵
You impose the geometric BC's to get the α coeffs since the function must be 'admissible'
    w(x,t) = [N(x)]{q(t)}
This has already been derived so no need to redo getting [N] in the code
"""

function compute_stiffness(EIᵉ, lᵉ, nDOF=2)
    """
    The internal strain energy of a beam is
        U = 0.5∫₀ᴸ EI (∂²w/∂x²)² dx + 0.5∫₀ᴸ GJ (∂ψ/∂x)² dx = 0.5{q(t)}ᵀ[Kᵉ]{q(t)}

    Element stiffness matrix from the strain energies
        # [Kᵉ] = ∫₀ᴸ EI [N''(x)]ᵀ [N''(x)]dx # maybe wrong now
    """
    if nDOF == 2 # bending + twisting
        kb = EIᵉ / lᵉ^3
        kt = GJᵉ / lᵉ
        Kᵉ = [
            [12 * kb, 0, 6 * kb * lᵉ, -12 * kb, 0, 6 * kb * lᵉ]
            [0, kt, 0, 0, -kt, 0]
            [6 * kb * lᵉ, 0, 4 * kb * lᵉ^2, -6 * kb * lᵉ, 0, 2 * kb * lᵉ^2]
            [-12 * kb, 0, -6 * kb * lᵉ, 12 * kb, 0, -6 * kb * lᵉ]
            [0, -kt, 0, 0, kt, 0]
            [6 * kb * lᵉ, 0, 2 * kb * lᵉ^2, -6 * kb * lᵉ, 0, 4 * kb * lᵉ^2]
        ] # 6x6 stiffness
    elseif nDOF = 3 # bending + twisting + axial
        # TODO
        Kᵉ = 1
    end

    return Kᵉ
end

# function compute_mass(mᵉ,,lᵉ, nDOF=2)
#     """
#     The kinetic energy is
#         T = 0.5∫₀ᴸ m (∂w/∂t)² dx = 0.5{q̇(t)}ᵀ[Mᵉ]{q̇(t)}

#     Element mass matrix is 
#         [Mᵉ] = ∫₀ᴸ m[N(x)]ᵀ[N(x)]dx
#     """
#     if nDOF == 2 # bending + twisting
#         Mᵉ = lᵉ / 420 *
#              [
#             156 22*lᵉ 54 -13*lᵉ
#             22*lᵉ 4*lᵉ^2 13*lᵉ -3*lᵉ
#             54 13*lᵉ 156 -22*lᵉ
#             -13*lᵉ -3*lᵉ^2 -22*lᵉ 4*lᵉ^2
#         ]
#     elseif nDOF = 3 # bending + twisting + axial
#         # TODO
#         Mᵉ = 1
#     end

#     # --- Add direction dependent fluid added mass ---


#     return Mᵉ
# end

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