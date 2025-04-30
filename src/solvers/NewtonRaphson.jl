"""
Newton-Raphson solver
"""


function print_solver_history(iterNum::Int64, resNorm, stepNorm)

    ChainRulesCore.ignore_derivatives() do
        if iterNum == 1
            println("+-------+------------------------+----------+")
            println("|  Iter |         resNorm        | stepNorm |")
            println("+-------+------------------------+----------+")
        end
        @printf("   %03d    %.16e   %.2e  ", iterNum, real(resNorm), real(stepNorm))
        println()
    end
end

function do_newton_raphson(
    compute_residuals, compute_∂r∂u, u0::Vector, WorkingListOfParams=nothing;
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
    x0 : dict, optional
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

    # ************************************************
    #     Initialize output
    # ************************************************
    if !isnothing(WorkingListOfParams)

        # if solverOptions["use_nlll"]
        if length(WorkingListOfParams) == 4
            appendageParamsList = WorkingListOfParams[3+iComp]
        else
            appendageParamsList = WorkingListOfParams[3+iComp:end]
        end

        xLE, nodeConn, xTE = WorkingListOfParams[1:3]
        xVec, mm, nn = unpack_coords(xLE, xTE)

        res = compute_residuals(u0, xVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions)
        # else
        #     DVDict = WorkingListOfParams[iComp]
        #     x0, DVLengths =  unpack_dvdict(DVDict)
        #     res = compute_residuals(
        #         u0, x0, DVLengths;
        #         appendageOptions=appendageOptions,
        #         solverOptions=solverOptions,
        #         iComp=iComp,
        #         CLMain=CLMain,
        #         DVDictList=WorkingListOfParams,
        #     )
        # end
    else
        res = compute_residuals(u0; solverParams=solverParams)
    end


    converged_u = copy(u0)
    converged_r = copy(res)
    iters = 1

    if !is_cmplx
        u = u0
        for ii in 1:maxIters
            if !isnothing(WorkingListOfParams)
                # if solverOptions["use_nlll"]
                if length(WorkingListOfParams) == 4
                    appendageParamsList = WorkingListOfParams[3+iComp]
                else
                    appendageParamsList = WorkingListOfParams[3+iComp:end]
                end

                xLE, nodeConn, xTE = WorkingListOfParams[1:3]
                xVec, mm, nn = unpack_coords(xLE, xTE)

                res = compute_residuals(u, xVec, nodeConn, appendageParamsList; appendageOptions=appendageOptions, solverOptions=solverOptions)
                ∂r∂u = compute_∂r∂u(u, xLE, xTE, nodeConn, mode;
                    appendageParamsList=appendageParamsList,
                    solverParams=solverParams,
                    appendageOptions=appendageOptions, solverOptions=solverOptions)
                # else

                #     x0, DVLengths =  unpack_dvdict(DVDict)
                #     res = compute_residuals(u, x0, DVLengths;
                #         appendageOptions=appendageOptions, solverOptions=solverOptions, DVDictList=WorkingListOfParams, iComp=iComp, CLMain=CLMain)
                #     ∂r∂u = compute_∂r∂u(u, mode;
                #         DVDictList=WorkingListOfParams,
                #         solverParams=solverParams,
                #         appendageOptions=appendageOptions,
                #         solverOptions=solverOptions,
                #         iComp=iComp,
                #         CLMain=CLMain,
                #     )
                # end
            else
                res = compute_residuals(u; solverParams=solverParams)
                ∂r∂u = compute_∂r∂u(u; solverParams=solverParams, mode=mode)
            end

            jac = zeros(typeof(u[1]), length(u), length(u))
            jac = ∂r∂u

            # ************************************************
            #     Compute Newton step
            # ************************************************
            # show(stdout, "text/plain", jac)
            Δu = -jac \ res

            # println("u before: ", u)
            # println("Newton step Δu: ", Δu)

            # --- Update ---
            u = u + Δu
            # println("u after: ", u[end-9:end])
            # println("res:\n", res)

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
                ChainRulesCore.ignore_derivatives() do
                    println("+--------------------------------------------")
                    println("Failed to converge. res norm is NaN")
                end
                break
            end
            if real(resNorm) < tol
                ChainRulesCore.ignore_derivatives() do
                    if is_verbose
                        println("+--------------------------------------------")
                        println("Converged in ", ii, " iterations")
                    end
                end
                break
            elseif ii == maxIters
                ChainRulesCore.ignore_derivatives() do
                    println("+--------------------------------------------")
                    println("Failed to converge. res norm is $(resNorm)")
                    println("DID THE FOIL STATICALLY DIVERGE? CHECK DEFLECTIONS IN POST PROC")
                end
            end
        end

    elseif is_cmplx
        uUnfolded = [real(u0); imag(u0)]
        for ii in 1:maxIters
            # NOTE: these functions handle a complex input but return the unfolded output
            # (i.e., concatenation of real and imag)
            res = compute_residuals(uUnfolded; solverParams=solverParams)
            ∂r∂u = compute_∂r∂u(uUnfolded; solverParams=solverParams, mode=mode)
            jac = ∂r∂u

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

