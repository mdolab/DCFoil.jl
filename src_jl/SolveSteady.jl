# --- Julia ---

# @File    :   SolveSteady.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Solve the beam equations for a steady aero/hydrodynamics and compute gradients



module SolveSteady
"""
Steady hydroelastic solver module
"""

# --- Public functions ---
export solve

# --- Libraries ---
using FLOWMath: linear
include("InitModel.jl")
include("Struct.jl")
include("Hydro.jl")
using .InitModel, .Hydro, .StructProp # use coded libraries from this project

function solve(neval, DVDict)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)

    """
    # --- Initialize ---
    foil = InitModel.init_steady(neval, DVDict)

    # --- Solve ---

end

function compute_residual(stateVec)

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