module SolverRoutines
"""
Generic routines every solver needs

NOTE:
any function with '_d' at the end is the one used for forward differentiation 
because certain operations cannot be differentiated
by the AD tool (e.g., anything related to file writing)
'_b' is for backward mode
"""

# --- Libraries ---
using LinearAlgebra
include("./NewtonRhapson.jl")
include("./EigenvalueProblem.jl")
using .NewtonRhapson, .EigenvalueProblem
using Zygote

# ==============================================================================
#                         Solver routines
# ==============================================================================
function converge_r(compute_residuals, compute_∂r∂u, u; maxIters=200, tol=1e-6, is_verbose=true, mode="RAD", is_cmplx=false)
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

end # converge_r

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
end # return_totalStates

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
end # compute_eigsolve

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
end # cmplxInverse

function cmplxInverse_d(A_r, A_rd, A_i, A_id, n)
    """
    Compute inverse of a complex square matrix

    Forward analytic differentiation
        Cd = -(C*Ad*C)
        where C = A^-1
    # See: 
    #     Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation

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

    # Call regular code
    Ainv_r, Ainv_i = cmplxInverse(A_r, A_i, n)

    # Inverse matrix C
    C = zeros(Float64, 2 * n, 2 * n)
    C[1:n, 1:n] = Ainv_r
    C[1:n, n+1:end] = -Ainv_i
    C[n+1:end, 1:n] = Ainv_i
    C[n+1:end, n+1:end] = Ainv_r
    # Forward seed matrix
    Ad = zeros(Float64, 2 * n, 2 * n)
    Ad[1:n, 1:n] = A_rd
    Ad[1:n, n+1:end] = -A_id
    Ad[n+1:end, 1:n] = A_id
    Ad[n+1:end, n+1:end] = A_rd

    # Forward derivative
    Cd = -1.0 * (C * Ad) * C

    # --- Unpack solution ---
    Ainv_rd = Cd[1:n, 1:n]
    Ainv_id = Cd[n+1:end, 1:n]

    return Ainv_r, Ainv_rd, Ainv_i, Ainv_id
end # cmplxInverse_d

function cmplxMatmult(A_r, A_i, B_r, B_i)
    """
    Complex multiplication of matrices using real arithmetic
    """

    # Real
    C_r = A_r * B_r - A_i * B_i
    # Imag
    C_i = A_r * B_i + A_i * B_r

    return C_r, C_i
end # cmplxMatmult

function cmplxMatmult_d(A_r, A_rd, A_i, A_id, B_r, B_rd, B_i, B_id)
    """
    Complex multiplication of matrices using real arithmetic

    Forward analytic differentiation
        Cd = (Ad*B + A*Bd)
    # See:
    #     Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation
    """

    C_r, C_i = cmplxMatmult(A_r, A_i, B_r, B_i)

    C_rd = A_rd*B_r - A_id*B_i + A_r*B_rd - A_i*B_id

    C_id = A_rd*B_i + A_id*B_r + A_r*B_id + A_i*B_rd

    return C_r, C_rd, C_i, C_id
end # cmplxMatmult


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
    # eigen() is a spectral decomposition
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
end # cmplxStdEigValProb

function cmplxStdEigValProb_d(A_r, A_rd, A_i, A_id, n)
    # """
    # Forward mode analytic derivative for the standard eigenvalue problem Av= \lambda v
    # with eigenvalues d_k
    #     Dd = I \circ (U^-1 * Ad U)
    #     Ud = U * (F \circ U^-1 * Ad * U)
    # where F_ij = (d_j - d_i)^-1 for i != j and zero otherwise --> F_ij = E_ij^-1
    # 
    # 'd' terms are the forward seeds
    # 
    # See: 
    #     Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation
    # 
    # Inputs
    # ------
    #     A_r
    #     A_rd (forward seed)
    #     A_i
    #     A_id (forward seed)
    # Outputs
    # -------
    #     w_r - real part of eigenvalues
    #     w_rd - real part of eigenvalues (derivative)
    #     w_i - imag part of eigenvalues
    #     w_id - imag part of eigenvalues (derivative)
    #     VR_r - real right eigenvectors
    #     VR_rd - real right eigenvectors (derivative)
    #     VR_i - imag right eigenvectors
    #     VR_id - imag right eigenvectors (derivative)
    # """

    # --- Initialize matrices ---
    w_rd = zeros(size(A_r))
    w_id = zeros(size(A_r))
    E = zeros(ComplexF64, size(A_r))
    F = zeros(ComplexF64, size(A_r))
    A = A_r + 1im * A_i
    Ad = A_rd + 1im * A_id

    # --- Solve standard eigenvalue problem (Ax = λx) ---
    # This method uses the julia built-in eigendecomposition
    # eigen() is a spectral decomposition
    w, Vr = eigen(A)
    w_r = real(w)
    w_i = imag(w)
    VR_r = real(Vr)
    VR_i = imag(Vr)

    # ---------------------------
    #   Eigenvalue derivatives Dd
    # ---------------------------
    # --- Compute eigenvector inverses U^-1 ---
    Vrinv_r, Vrinv_i = cmplxInverse(VR_r, VR_i, n)
    Vrinv = Vrinv_r + 1im * Vrinv_i

    # --- Compute U^-1 * Ad * U ---
    tmp1 = (Vrinv * Ad) * Vr

    # Don't do Hadamard product with identity matrix.
    for ii = 1:n
        w_rd = real(tmp1[ii, ii])
        w_id = imag(tmp1[ii, ii])
    end
    # ---------------------------
    #   Eigenvector derivatives Ud
    # ---------------------------
    # --- E ---
    for jj in 1:n
        for ii in 1:n
            E[ii, jj] = w[jj] - w[ii]
        end
    end

    # --- F ---
    for jj in 1:n
        for ii in 1:n
            if ii != jj
                F[ii, jj] = 1.0 / E[ii, jj]
            end
        end
    end

    # --- F \circ (U^-1 * Ad * U) ---
    tmp1 = F .* tmp1

    # --- Final U * (F \circ (U^-1 * Ad * U)) ---
    Vrd = Vr * tmp1
    VR_rd = real(Vrd)
    VR_id = imag(Vrd)

    return w_r, w_rd, w_i, w_id, VR_r, VR_rd, VR_i, VR_id
end # cmplxStdEigValProb_d
# ==============================================================================
#                         Utility routines
# ==============================================================================
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

end # argmax2d

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

end # maxLocArr2d

function count1d(mask)
    """
    Count number of 'true' elements in 1d array
    """

    nTrue = 0

    for ii in eachindex(mask)
        if mask[ii]
            nTrue += 1
        end
    end

    return nTrue
end # count1d

function ipack1d(A, mask, nFlow)
    """
    Extract elements from array A which have corresponding element in mask set to 'true'
    mask array contains boolean values

    Outputs
    -------
        B - subset array containing elements of A which have corresponding element in mask set to 'true'
        nFound - number of elements in B
    """

    nTrue = count1d(mask)
    B = zeros(Int64, nFlow)
    B_z = Zygote.Buffer(B)

    nFound = 0
    for ii in eachindex(A)
        if mask[ii]
            nFound += 1
            B_z[nFound] = A[ii]
        end
    end
    B = copy(B_z)

    return B, nFound
end # ipack1d

end # SolverRoutines