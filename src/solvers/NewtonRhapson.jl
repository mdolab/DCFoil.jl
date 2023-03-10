"""
Newton-Rhapson solver
"""
module NewtonRhapson

export do_newton_rhapson

# --- Libraries ---
using LinearAlgebra, Statistics
using Printf

function print_solver_history(iterNum, resNorm, stepNorm)
    if iterNum == 1
        println("+-------+------------------------+----------+")
        println("|  Iter |         resNorm        | stepNorm |")
        println("+-------+------------------------+----------+")
    end
    @printf("   %03d    %.16e   %.2e  ", iterNum, resNorm, stepNorm)
    println()
end

function do_newton_rhapson(compute_residuals, compute_∂r∂u, u0, maxIters=200, tol=1e-12, is_verbose=true, mode="FAD", is_cmplx=false)
    """
    Simple Newton-Rhapson solver

    Inputs
    ------
    compute_residuals : function handle
        Function that computes the residuals
    compute_∂r∂u : function handle
        Function that computes the Jacobian
    u : array
        Initial guess
    maxIters : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence on norm of residual vector
    is_verbose : bool
        Print out the residual norm at each iteration
    mode : string
        Compute Jacobian using finite differences or automatic differentiation
    is_cmplx : bool
        Solve for complex roots
    """


    if !is_cmplx
        u = u0
        for ii in 1:maxIters
            # println(u)
            res = compute_residuals(u)
            ∂r∂u = compute_∂r∂u(u, mode)
            jac = ∂r∂u[1]

            # ************************************************
            #     Compute Newton step
            # ************************************************
            # show(stdout, "text/plain", jac)
            Δu = -jac \ res

            # --- Update ---
            u = u + Δu

            resNorm = norm(res, 2)

            # --- Printout ---
            if is_verbose
                print_solver_history(ii, resNorm, norm(Δu, 2))
            end

            # ************************************************
            #     Check norm
            # ************************************************
            # Note to self, the for and while loop in Julia introduce a new scope...this is pretty stupid
            if resNorm < tol
                println("+--------------------------------------------")
                println("Converged in ", ii, " iterations")
                global converged_u = copy(u)
                global converged_r = copy(res)
                global iters = copy(ii)
                break
            elseif ii == maxIters
                println("+--------------------------------------------")
                println("Failed to converge. res norm is", resNorm)
                println("DID THE FOIL STATICALLY DIVERGE? CHECK DEFLECTIONS IN POST PROC")
                global converged_u = copy(u)
                global converged_r = copy(res)
                global iters = copy(ii)
            else
                global converged_u = copy(u)
                global converged_r = copy(res)
                global iters = copy(ii)
            end
        end

    elseif is_cmplx
        uUnfolded = [real(u0); imag(u0)]
        for ii in 1:maxIters
            # NOTE: these functions handle a complex input but return the unfolded output
            # (i.e., concatenation of real and imag)
            res = compute_residuals(uUnfolded)
            ∂r∂u = compute_∂r∂u(uUnfolded, mode)
            jac = ∂r∂u[1]

            # --- Newton step ---
            Δu = -jac \ res

            # --- Update ---
            uUnfolded = uUnfolded + Δu

            resNorm = norm(res, 2)

            # --- Printout ---
            if is_verbose
                print_solver_history(ii, resNorm, norm(Δu, 2))
            end

            # --- Check norm ---
            # Note to self, the for and while loop in Julia introduce a new scope...this is pretty stupid
            if resNorm < tol
                println("+--------------------------------------------")
                println("Converged in ", ii, " iterations")
                global converged_u = uUnfolded[1:end÷2] + 1im * uUnfolded[end÷2+1:end]
                global converged_r = res[1:end÷2] + 1im * res[end÷2+1:end]
                global iters = copy(ii)
                break
            elseif ii == maxIters
                println("+--------------------------------------------")
                println("Failed to converge. res norm is", resNorm)
                println("DID THE FOIL STATICALLY DIVERGE? CHECK DEFLECTIONS IN POST PROC")
                global converged_u = uUnfolded[1:end÷2] + 1im * uUnfolded[end÷2+1:end]
                global converged_r = res[1:end÷2] + 1im * res[end÷2+1:end]
                global iters = copy(ii)
            else
                global converged_u = uUnfolded[1:end÷2] + 1im * uUnfolded[end÷2+1:end]
                global converged_r = res[1:end÷2] + 1im * res[end÷2+1:end]
                global iters = copy(ii)
            end
        end
    end

    return converged_u, converged_r, iters
end

end
