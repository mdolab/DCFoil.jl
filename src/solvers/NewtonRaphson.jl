module NewtonRaphson
"""
Newton-Raphson solver
"""

export do_newton_raphson

# --- PACKAGES ---
using LinearAlgebra, Statistics
using FLOWMath: norm_cs_safe
using Printf

using ..Utilities

function print_solver_history(iterNum::Int64, resNorm, stepNorm)
    if iterNum == 1
        println("+-------+------------------------+----------+")
        println("|  Iter |         resNorm        | stepNorm |")
        println("+-------+------------------------+----------+")
    end
    @printf("   %03d    %.16e   %.2e  ", iterNum, resNorm, stepNorm)
    println()
end

function do_newton_raphson(
    compute_residuals, compute_∂r∂u, u0::Vector, DVDictList::Vector;
    maxIters=200, tol=1e-12, is_verbose=true, mode="RAD", is_cmplx=false,
    appendageOptions=Dict(),
    solverOptions=Dict(),
    solverParams=nothing,
    iComp=1,
    CLMain=0.0
)
    """
    Simple Newton-Raphson solver

    Inputs
    ------
    compute_residuals : function handle
        Function that computes the residuals
        must have signature f(u; solverParams)
    compute_∂r∂u : function handle
        Function that computes the Jacobian
        must have signature f(u, mode; solverParams)
    u0 : array
        Initial guess
    x0 : dict
        Dictionary of design variables (these do not change during the solve)
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

    # --- Initialize output ---
    DVDict = DVDictList[iComp]
    x0, DVLengths = Utilities.unpack_dvdict(DVDict)
    res = compute_residuals(
        u0, x0, DVLengths;
        appendageOptions=appendageOptions,
        solverOptions=solverOptions,
        iComp=iComp,
        CLMain=CLMain,
        DVDictList=DVDictList,
    )

    converged_u = copy(u0)
    converged_r = copy(res)
    iters = 1

    if !is_cmplx
        u = u0
        for ii in 1:maxIters
            x0, DVLengths = Utilities.unpack_dvdict(DVDict)

            res = compute_residuals(u, x0, DVLengths;
                appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=DVDictList, iComp=iComp, CLMain=CLMain)

            ∂r∂u = compute_∂r∂u(u, mode;
                DVDictList=DVDictList,
                solverParams=solverParams,
                appendageOptions=appendageOptions,
                solverOptions=solverOptions,
                iComp=iComp,
                CLMain=CLMain,
            )

            jac = zeros(typeof(u[1]), length(u), length(u))
            jac = ∂r∂u

            # ************************************************
            #     Compute Newton step
            # ************************************************
            # show(stdout, "text/plain", jac)
            Δu = -jac \ res

            # --- Update ---
            u = u + Δu

            resNorm = norm_cs_safe(res, 2)

            # --- Printout ---
            if is_verbose
                print_solver_history(ii, resNorm, norm_cs_safe(Δu, 2))
            end

            # ************************************************
            #     Check norm
            # ************************************************
            # Note to self, the for and while loop in Julia introduce a new scope...this is pretty stupid
            converged_u = copy(u)
            converged_r = copy(res)
            iters = copy(ii)
            if isnan(resNorm)
                println("+--------------------------------------------")
                println("Failed to converge. res norm is NaN")
                break
            end
            if resNorm < tol
                println("+--------------------------------------------")
                println("Converged in ", ii, " iterations")
                break
            elseif ii == maxIters
                println("+--------------------------------------------")
                println("Failed to converge. res norm is", resNorm)
                println("DID THE FOIL STATICALLY DIVERGE? CHECK DEFLECTIONS IN POST PROC")
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

            resNorm = norm_cs_safe(res, 2)

            # --- Printout ---
            if is_verbose
                print_solver_history(ii, resNorm, norm_cs_safe(Δu, 2))
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
