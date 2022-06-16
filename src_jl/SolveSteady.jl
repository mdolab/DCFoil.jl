# --- Julia ---

# @File    :   SolveSteady.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Solve the beam equations for a steady aero/hydrodynamics and compute gradients



module SolveSteady
"""
Steady hydroelastic solver module
"""

# --- Libraries ---
using FLOWMath: linear
using .InitModel, .Hydro, .StructProp # use coded libraries from this project

mutable struct foil
    c::Float64
    t::Float64
    ab::Float64
    mₛ::Float64 # sectional mass

end

function solve_system(neval::Int64)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)

    """
    # --- Initialize ---

end

function compute_residual()

end

function compute_jacobian(stateVec)
    """
    Compute the jacobian dr/du

    Inputs:
        stateVec: array, shape (neval, 8), state vector for 'neval' spatial nodes

    returns:
        array, shape (neval, neval), square jacobian matrix

    """
    # TODO:

end

end # end module