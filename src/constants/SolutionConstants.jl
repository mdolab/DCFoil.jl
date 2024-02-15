
# --- Julia ---

# @File    :   SolutionConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Module to store used in the solution


module SolutionConstants

# using SparseArrays: SparseMatrixCSC
# ==============================================================================
#                         CONSTANTS
# ==============================================================================
const MEPSLARGE = 1.11e-14 # machine epsilon but 1e-14 instead of 1e-16 b/c it is a more robust value for solving
const P_IM_TOL = 1.11e-10 # previously 1.11e-11 but wasn't doing too well on static div # tolerance on real root criteria 
# NOTE: this is tested to work. 
# Bigger values catch the real roots and too small cause them to disappear
# You just don't want them too big that they pick up wrong roots

const XDIM = 1
const YDIM = 2
const ZDIM = 3

# ==============================================================================
#                         STRUCTS
# ==============================================================================
mutable struct DCFoilConstants{T}
    """
    This is a catch all mutable struct to store variables that we do not
    want in function calls like r(u) or f(u)

    """
    Kmat::Matrix{T} # structural stiffness matrix (after BC blanking)
    Mmat::Matrix{T} # structural mass matrix (after BC blanking)
    Cmat::Matrix{T} # structural damping matrix (after BC blanking)
    elemType::String
    mesh::Matrix{T}
    AICmat::Matrix{T} # Aero influence coeff matrix (no BC blanking)
    mode::String # type of derivative for drdu
    planformArea::T
end

mutable struct DCFoilDynamicConstants{T}
    """
    For the dynamic hydroelastic solve, there are more constants to store
    """
    elemType::String
    mesh::Array{T,2}
    Dmat::Matrix{ComplexF64} # dynamic matrix 
    AICmat::Matrix{ComplexF64} # just the aero part of Dmat 
    extForceVec::Vector{T} # external force vector excluding BC nodes
end

end # end module