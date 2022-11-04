"""
Generic routines every solver needs
"""

module SolverRoutines

# --- Libraries ---
include("./NewtonRhapson.jl")
using .NewtonRhapson

function converge_r(compute_residuals, compute_∂r∂u, u, maxIters=200, tol=1e-6, verbose=true, mode="FAD", is_cmplx=false)
    """
    Given input u, solve the system r(u) = 0
    Tells you how many NL iters
    """

    # ************************************************
    #     Main solver loop
    # ************************************************
    println("Beginning NL solve")

    # Somewhere here, you could do something besides Newton-Rhapson if you want
    converged_u, converged_r, iters = NewtonRhapson.do_newton_rhapson(compute_residuals, compute_∂r∂u, u, maxIters, tol, verbose, mode, is_cmplx)

    return converged_u, converged_r

end

end