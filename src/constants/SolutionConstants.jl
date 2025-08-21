
# --- Julia ---

# @File    :   SolutionConstants.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   File to store used in the solution


# ==============================================================================
#                         CONSTANTS
# ==============================================================================
const MEPSLARGE = 1.11e-14 # machine epsilon but 1e-14 instead of 1e-16 b/c it is a more robust value for solving
const P_IM_TOL = 1.11e-10 # previously 1.11e-11 but wasn't doing too well on static div # tolerance on real root criteria 
# NOTE: this is tested to work. 
# Bigger values catch the real roots and too small cause them to disappear
# You just don't want them too big that they pick up wrong roots

const PVAP = 2.34e3 # vapor pressure of freshwater at 20C [Pa]
const GRAV = 9.80665 # gravity [m/s^2]
const XDIM = 1
const YDIM = 2
const ZDIM = 3

const ELEMTYPE = "COMP2"

# ==============================================================================
#                         STRUCTS
# ==============================================================================
struct DCFoilSolverParams{TF,TC}
    """
    This is a catch all immutable struct to store expensive vars that we do not
    want in function calls like r(u) or f(u)
    """
    Kmat::AbstractMatrix{TC} # structural stiffness matrix (no BC blanking)
    Mmat::AbstractMatrix{TC} # structural mass matrix (no BC blanking)
    Cmat::AbstractMatrix # structural damping matrix (no BC blanking)
    AICmat::AbstractMatrix # Aero influence coeff matrix (no BC blanking)
    areaRef::TF # reference area for coefficients [m^2]
    downwashAngles::TF # downwash angles [rad]
end

struct DCFoilDynamicConstants{TF,TC,TI,TS,TA<:AbstractVector{TF}}
    """
    For the dynamic hydroelastic solve, there are more constants to store
    """
    Dmat::Matrix{TC} # dynamic matrix 
    AICmat::Matrix{TC} # just the aero part of Dmat 
    extForceVec::TA # external force vector excluding BC nodes
end
