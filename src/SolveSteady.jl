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
include("GovDiffEqns.jl")
using .InitModel, .Hydro, .StructProp, .Steady, .Solver

function solve(neval::Int64, DVDict)
    """
    Essentially solve [K]{u} = {f} (see paper for actual equations and algorithm)

    """
    # ---------------------------
    #   Initialize
    # ---------------------------
    foil = InitModel.init_steady(neval, DVDict)
    η = LinRange(0, 1, neval) # parametric spanwise variable
    # initialize a BC vector at the root. We know first 4 are zero
    q⁰ = zeros(8)
    q⁰[5:end] .= 1.0 

    # ---------------------------
    #   Solve the Diff Eq
    # ---------------------------
    # --- Solves a 2PT BVP ---
    ysol, qsol = Solver.solve_bvp(Steady.compute_∂q∂y, q⁰, 0, 1, 25, Steady.compute_g, foil)

    # --- Compute hydro loads of deflected shape ---
    # TODO:maybe use mutable struct to store the solution?

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

function write_sol()

end

end # end module