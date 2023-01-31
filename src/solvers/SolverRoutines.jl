module SolverRoutines
"""
Generic solver routines every solver needs
"""

# --- Libraries ---
using LinearAlgebra
include("./NewtonRhapson.jl")
include("./EigenvalueProblem.jl")
using .NewtonRhapson, .EigenvalueProblem

function converge_r(compute_residuals, compute_∂r∂u, u; maxIters=200, tol=1e-6, is_verbose=true, mode="FAD", is_cmplx=false)
    """
    Given input u, solve the system r(u) = 0
    Tells you how many NL iters
    """

    # ************************************************
    #     Main solver loop
    # ************************************************
    if is_verbose
        println("+", "-"^50, "+")
        println("|              Beginning NL solve                  |")
        println("+", "-"^50, "+")
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

function compute_cost_func(foilStructuralStates)
    # TODO
end


# ==============================================================================
#                         Linear algebra routines
# ==============================================================================
function compute_eigsolve(K, M, nEigs; issym=true)
    """
    Compute eigenvalues and eigenvectors of a system the julia way
    """

    # ************************************************
    #     Compute eigenvalues and eigenvectors
    # ************************************************
    eVals, eVecs = EigenvalueProblem.compute_eigsolve(K, M, nEigs; issym=issym)

    return eVals, eVecs
end

function cmplxInverse(A_r, A_i, n)
    """
    Compute inverse of a complex square matrix

    Inputs
    ------
        A_r - real part of A matrix
        A_i - imag part of A matrix
        n - dimension of A matrix
    Outputs
    -------
        Ainv_r - real part of inv(A)
        Ainv_i - imag part of inv(A)
    """

    # --- Assemble auxiliary matrix ---
    firstRow = hcat(A_r, -A_i)
    secondRow = hcat(A_i, A_r)
    Acopy = vcat(firstRow, secondRow)

    # --- First compute LU factorization ---
    luA = lu(Acopy)
    # RHS identity matrix for inverse
    RHS = Matrix{Float64}(I, 2 * n, 2 * n)

    # --- Then solve linear system with multiple RHS's ---
    Ainvcopy = luA \ RHS

    # --- Extract solution ---
    Ainv_r = Ainvcopy[1:n, 1:n]

    # TODO: you could add the option for transpose here like Eirikur
    Ainv_i = Ainvcopy[n+1:end, 1:n]

    return Ainv_r, Ainv_i
end

function cmplxMatmult(A_r, A_i, B_r, B_i)
    """
    Complex multiplication of matrices using real arithmetic
    """

    # Real
    C_r = A_r * B_r - A_i * B_i
    # Imag
    C_i = A_r * B_i + A_i * B_r

    return C_r, C_i
end

function cmplxStdEigValProb(A_r, A_i, n)
    """
    Inputs
    ------
        A_r
        A_i
        n

    Outputs
    -------
        w_r - real part of eigenvalues
        w_i - imag part of eigenvalues
        VL_r - left eigenvectors
        VL_i - left eigenvectors
        VR_r - right eigenvectors
        VR_i - right eigenvectors
        We mostly care about the right eigenvectors (occuring on right of A matrix)
    """

    # --- Initialize matrices ---
    A = A_r + 1im * A_i


    # --- Solve standard eigenvalue problem (Ax = λx) ---
    # This method uses the julia built-in eigendecomposition
    w, Vr = eigen(A)

    # Eigenvalues
    w_r = real(w)
    w_i = imag(w)
    # Eigenvectors
    VR_r = real(Vr)
    VR_i = imag(Vr)
    # and some dummy values that aren't actually right
    VL_r = real(Vr)
    VL_i = imag(Vr)

    return w_r, w_i, VL_r, VL_i, VR_r, VR_i
end

function argmax2d(A)
    """
    Find the indices of maximum value for each column of 2d array A

    Outputs
    -------
        locs - array of indices
    """
    ncol = size(A)[2]

    locs = zeros(Int64, ncol)

    for col in 1:ncol
        locs[col] = argmax(A[:, col])
    end

    return locs

end

function maxLocArr2d(A)
    """
    Find the maximum location for 2d array A

    Outputs
    -------
        locs - array of indices
    """
    ncol = size(A)[2]
    maxI = 1
    maxJ = 1
    maxVal = A[maxI, maxJ]

    for jj in 1:ncol
        for ii in 1:size(A)[1]
            if A[ii, jj] > maxVal
                maxI = ii
                maxJ = jj
                maxVal = A[ii, jj]
            end
        end
    end

    return maxI, maxJ, maxVal

end
    
end # end module