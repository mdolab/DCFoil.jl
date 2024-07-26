
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

const GRAV = 9.80665 # gravity [m/s^2]
const XDIM = 1
const YDIM = 2
const ZDIM = 3

# ==============================================================================
#                         STRUCTS
# ==============================================================================
struct DCFoilSolverParams{TF,TC,TI,TS}
    """
    This is a catch all immutable struct to store variables that we do not
    want in function calls like r(u) or f(u)
    """
    Kmat::Matrix{TC} # structural stiffness matrix (no BC blanking)
    Mmat::Matrix{TC} # structural mass matrix (no BC blanking)
    Cmat::Matrix{TC} # structural damping matrix (no BC blanking)
    elemType::TS
    AICmat::Matrix{TF} # Aero influence coeff matrix (no BC blanking)
    mode::TS # type of derivative for drdu
    planformArea::TF
    dofBlank::Vector{TI} # DOF to blank out
    downwashAngles::TF # downwash angles [rad]
end

struct DCFoilDynamicConstants{TF,TC,TI,TS,TA<:AbstractVector{TF}}
    """
    For the dynamic hydroelastic solve, there are more constants to store
    """
    elemType::TS
    mesh::Matrix{TF}
    Dmat::Matrix{TC} # dynamic matrix 
    AICmat::Matrix{TC} # just the aero part of Dmat 
    extForceVec::TA # external force vector excluding BC nodes
end

end # end module