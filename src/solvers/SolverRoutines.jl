"""
Generic routines every solver needs
"""

module SolverRoutines

# --- Libraries ---
include("./NewtonRhapson.jl")
using .NewtonRhapson

function converge_r(compute_residuals, compute_∂r∂u, u; maxIters=200, tol=1e-6, is_verbose=true, mode="FAD", is_cmplx=false)
    """
    Given input u, solve the system r(u) = 0
    Tells you how many NL iters
    """

    # ************************************************
    #     Main solver loop
    # ************************************************
    if is_verbose
        # TODO: probably a better way to pretty print this
        println("+","-"^50,"+")
        println("|              Beginning NL solve                  |")
        println("+","-"^50,"+")
    end

    # Somewhere here, you could do something besides Newton-Rhapson if you want
    converged_u, converged_r, iters = NewtonRhapson.do_newton_rhapson(compute_residuals, compute_∂r∂u, u, maxIters, tol, is_verbose, mode, is_cmplx)

    return converged_u, converged_r

end

function return_totalStates(foilStructuralStates, FOIL, elemType="BT2")
    """
    Returns the deflected + rigid shape of the foil
    """

    alfaRad = FOIL.α₀ * π / 180

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
        staticOffset = [0, 0, alfaRad]
    elseif elemType == "BT2"
        nDOF = 4
        staticOffset = [0, 0, alfaRad, 0] #TODO: pretwist will change this
    end

    # Add static angle of attack to deflected foil
    w = foilStructuralStates[1:nDOF:end]
    foilTotalStates = copy(foilStructuralStates) + repeat(staticOffset, outer=[length(w)])

    return foilTotalStates, nDOF
end

end