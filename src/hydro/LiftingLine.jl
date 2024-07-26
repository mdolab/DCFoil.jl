# --- Julia 1.9 ---
"""
@File    :   LiftingLine.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   Modern lifting line from Phillips and Snyder 2000, Reid 2020 appendix
             The major weakness is the discontinuity in the locus of aerodynamic centers
             for a highly swept wing at the root.
             Reid 2020 overcame this using a blending function at the wing root
"""

module LiftingLine

# --- DCFoil modules ---
using ..VPM: VPM
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..DCFoil: DTYPE
using ..SolverRoutines: compute_anglesFromVector

struct LiftingSurface{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    alphaGeo::TF
    planformArea::TF
    nodePts::TA
    collocationPts::TA
end

function setup(wingSpan, sweepAng, ; airfoilCoords="input.dat")
    """
    
    """
    # TODO: PICKUP HERE
    sigma = 4 * cos(sweepAng)^2 / (blend^2 * wingSpan^2)
    area = rootChord * wingSpan * (1 + TR) * 0.5

end

function solve(LiftingSystem)

end

function compute_LLJacobian(G)
    """
    Compute the Jacobian of the nonlinear, nondimensional lifting line equation
    G - Circulation distribution normalized by freestream velocity
    """
    return J
end

end