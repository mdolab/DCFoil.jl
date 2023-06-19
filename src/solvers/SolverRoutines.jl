module SolverRoutines
"""
Generic routines every solver needs

NOTE:
any function with '_d' at the end is the one used for forward differentiation 
because certain operations cannot be differentiated
by the AD tool (e.g., anything related to file writing)
'_b' is for backward mode.
In julia, the chainrules rrule is '_b'
"""

# --- Libraries ---
using LinearAlgebra
using ChainRulesCore
include("./NewtonRhapson.jl")
include("./EigenvalueProblem.jl")
using .NewtonRhapson, .EigenvalueProblem
using Zygote

const RealOrComplex = Union{Real,Complex}

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

function return_totalStates(foilStructuralStates, α₀, elemType="BT2")
    """
    Returns the deflected + rigid shape of the foil
    """

    alfaRad = α₀ * π / 180
    # alfaRad = deg2rad(α₀)

    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
        staticOffset = [0, 0, alfaRad]
    elseif elemType == "BT2"
        nDOF = 4
        nGDOF = nDOF * 3 # number of DOFs on node in global coordinates
        staticOffset = [0, 0, alfaRad, 0] #TODO: pretwist will change this
        staticOffset = [0, 0, 0, 0, 0, 0, alfaRad, 0, 0, 0, 0, 0]
    end

    # Add static angle of attack to deflected foil
    w = foilStructuralStates[1:nDOF:end]
    w = foilStructuralStates[1:nGDOF:end]
    foilTotalStates = copy(foilStructuralStates) + repeat(staticOffset, outer=[length(w)])

    return foilTotalStates, nGDOF
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

function cmplxInverse_b()
    # TODO:
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

    C_rd = A_rd * B_r - A_id * B_i + A_r * B_rd - A_i * B_id

    C_id = A_rd * B_i + A_id * B_r + A_r * B_id + A_i * B_rd

    return C_r, C_rd, C_i, C_id
end # cmplxMatmult

function cmplxMatmult_b(A_r, A_i, B_r, B_i)
    # Won't do this for now: 
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
    """
    Forward mode analytic derivative for the standard eigenvalue problem [A] v= λ v
        with eigenvalues d_k
        Dd = I ∘ (U^-1 * Ad U)
        Ud = U * (F ∘ U^-1 * Ad * U)
        where F_ij = (d_j - d_i)^-1 for i != j and zero otherwise --> F_ij = E_ij^-1
        
        'd' terms are the forward seeds
        
        See: 
        Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation
        
        Inputs
        ------
        A_r - nxn real part matrix
        A_rd (forward seed)
        A_i - nxn imag part matrix
        A_id (forward seed)
    Outputs
    -------
    w_r - real part of eigenvalues
    w_rd - real part of eigenvalues (derivative)
    w_i - imag part of eigenvalues
    w_id - imag part of eigenvalues (derivative)
    VR_r - real right eigenvectors
    VR_rd - real right eigenvectors (derivative)
    VR_i - imag right eigenvectors
    VR_id - imag right eigenvectors (derivative)
    The eigenvalue derivatives compare well with the FD check
    """

    # --- Initialize matrices ---
    w_rd = zeros(n)
    w_id = zeros(n)
    E = zeros(ComplexF64, n, n)
    F = zeros(ComplexF64, n, n)
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
        w_rd[ii] = real(tmp1[ii, ii])
        w_id[ii] = imag(tmp1[ii, ii])
    end
    # ---------------------------
    #   Eigenvector derivatives U̇
    # ---------------------------
    # TODO: these don't work apparently
    # --- E ---
    for jj in 1:n
        for ii in 1:n
            E[ii, jj] = w[jj] - w[ii]
        end
    end

    # --- F ---
    for jj in 1:n
        for ii in 1:n
            if jj != ii
                F[ii, jj] = 1.0 / E[ii, jj]
            end
        end
    end

    # --- F ∘ (U^-1 * Ad * U) ---
    tmp2 = F .* tmp1

    # --- Final U * (F ∘ (U^-1 * Ad * U)) ---
    Vrd = Vr * tmp2
    VR_rd = real(Vrd)
    VR_id = imag(Vrd)

    return w_r, w_rd, w_i, w_id, VR_r, VR_rd, VR_i, VR_id
end # cmplxStdEigValProb_d

function cmplxStdEigValProb_b(A_r, A_i, n, w̄_r, w̄_i, VR̄_r, VR̄_i)
    """
    Reverse mode analytic derivative for the standard eigenvalue problem [A] v= λ v
        with eigenvalues d_k
        Ā = U^-H * (D̄ + F ∘ (U^H * Ū)) * U^H
        where F_ij = (d_j - d_i)^-1 for i != j and zero otherwise --> F_ij = E_ij^-1
        
        overbar terms are the reverse seeds
        
    See: 
        Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation

    Inputs
    ------
    A_r - nxn real part matrix
    A_i - nxn imag part matrix
    Outputs
    -------
    w_r - real part of eigenvalues
    w_i - imag part of eigenvalues
    VR_r - real right eigenvectors
    VR_i - imag right eigenvectors
    """

    # --- Initialize matrices ---
    E = zeros(ComplexF64, n, n)
    F = zeros(ComplexF64, n, n)
    A = A_r + 1im * A_i
    VR̄ = VR̄_r + 1im * VR̄_i
    w̄ = w̄_r + 1im * w̄_i

    # --- Solve standard eigenvalue problem (Ax = λx) ---
    # This method uses the julia built-in eigendecomposition
    # eigen() is a spectral decomposition
    w, Vr = eigen(A)
    w_r = real(w)
    w_i = imag(w)
    VR_r = real(Vr)
    VR_i = imag(Vr)

    # ---------------------------
    #   Hermitian transpose (U^-H) conj transpose
    # ---------------------------
    VrHerm = Vr'
    VrHerminv_r, VrHerminv_i = cmplxInverse(real(VrHerm), imag(VrHerm), n)

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

    # --- Calculate F ∘ (U^H * Ū) ---
    tmp1 = F .* (VrHerm * VR̄)

    # Add D̄
    for ii = 1:n
        tmp1[ii, ii] += w̄[ii]
    end

    Ā = ((VrHerminv_r + 1im * VrHerminv_i) * tmp1) * VrHerm
    Ā_r = real(Ā)
    Ā_i = imag(Ā)

    # Then zero seeds out
    w̄_r = zeros(n)
    w̄_i = zeros(n)
    VR̄_r = zeros(n, n)
    VR̄_i = zeros(n, n)

    return Ā_r, Ā_i, w_r, w̄_r, w_i, w̄_i, VR_r, VR̄_r, VR_i, VR̄_i

end # cmplxStdEigValProb_b

function cmplxStdEigValProb2(A_r, A_i, n)
    """
    Give back eigenvalues and vectors as a unrolled vector
    """

    # --- Solve standard eigenvalue problem (Ax = λx) ---
    # This method uses the julia built-in eigendecomposition
    # eigen() is a spectral decomposition
    A = A_r + 1im * A_i
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

    # Unroll output so derivatives work
    y_r = vec(real(w))
    y_i = vec(imag(w))
    Y_r = vec(real(Vr))
    Y_i = vec(imag(Vr))
    y = vcat(y_r, y_i, Y_r, Y_i)

    return y
end # cmplxStdEigValProb

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
    locs_z = Zygote.Buffer(locs)

    for col in 1:ncol
        locs_z[col] = argmax(A[:, col])
    end
    locs = copy(locs_z)

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

function get_transMat(nVec, elemType="BT2")
    """
    Returns the transformation matrix for a given element type
    """
    if elemType == "BT2"
        Γ = [
            # Node 1
            nVec[1] nVec[2] nVec[3] 0.0 0.0 0.0 0.00000 0.00000 0.00000 0.0 0.0 0.0 # w1
            0.00000 0.00000 0.00000 1.0 0.0 0.0 0.00000 0.00000 0.00000 0.0 0.0 0.0 # w1'
            0.00000 0.00000 0.00000 0.0 1.0 0.0 0.00000 0.00000 0.00000 0.0 0.0 0.0 # ψ1
            0.00000 0.00000 0.00000 0.0 0.0 1.0 0.00000 0.00000 0.00000 0.0 0.0 0.0 # ψ1'
            # Node 2
            0.00000 0.00000 0.00000 0.0 0.0 0.0 nVec[1] nVec[2] nVec[3] 0.0 0.0 0.0 # w2
            0.00000 0.00000 0.00000 0.0 0.0 0.0 0.00000 0.00000 0.00000 1.0 0.0 0.0 # w2'
            0.00000 0.00000 0.00000 0.0 0.0 0.0 0.00000 0.00000 0.00000 0.0 1.0 0.0 # ψ2
            0.00000 0.00000 0.00000 0.0 0.0 0.0 0.00000 0.00000 0.00000 0.0 0.0 1.0 # ψ2'
        ]
    end
    return Γ
end
# ==============================================================================
#                         INTERPOLATION ROUTINES
# ==============================================================================
# The following functions are based off of Andrew Ning's publicly available akima spline code
# Except the derivatives are generated implicitly using Zygote RAD
function abs_smooth(x, Δx)
    """
    Absolute value function with quadratic in valley for C1 continuity
    """
    y = 0.0
    if (x >= Δx)
        y = x
    elseif (x <= -Δx)
        y = -x
    else
        y = x^2 / (2.0 * Δx) + Δx / 2.0
    end

    return y
end

function setup_akima(npt, xpt, ypt, Δx)
    """
    Setup for the akima spline
    Returns spline coefficients
    """
    eps = 1e-30

    # --- Output ---
    p0 = zeros(npt - 1)
    p1 = zeros(npt - 1)
    p2 = zeros(npt - 1)
    p3 = zeros(npt - 1)

    # --- Local working vars ---
    t = zeros(npt)
    m = zeros(npt + 3) # segment slopes
    # There are two extra end points and beginning and end
    # x---x---o--....--o---x---x
    # estimate             estimate

    # Zygote buffers
    p0_z = Zygote.Buffer(p0)
    p1_z = Zygote.Buffer(p1)
    p2_z = Zygote.Buffer(p2)
    p3_z = Zygote.Buffer(p3)
    t_z = Zygote.Buffer(t)
    m_z = Zygote.Buffer(m)

    # --- Compute segment slopes ---
    for ii in 1:npt-1
        m_z[ii+2] = (ypt[ii+1] - ypt[ii]) / (xpt[ii+1] - xpt[ii])
    end
    # Estimations
    m_z[2] = 2.0 * m_z[3] - m_z[4]
    m_z[1] = 2.0 * m_z[2] - m_z[3]
    m_z[npt+2] = 2.0 * m_z[npt+1] - m_z[npt]
    m_z[npt+3] = 2.0 * m_z[npt+2] - m_z[npt+1]
    m = copy(m_z)

    # --- Slope at points ---
    for ii in 1:npt
        m1 = m[ii]
        m2 = m[ii+1]
        m3 = m[ii+2]
        m4 = m[ii+3]
        w1 = abs_smooth(m4 - m3, Δx)
        w2 = abs_smooth(m2 - m1, Δx)
        if (w1 < eps && w2 < eps)
            t_z[ii] = 0.5 * (m2 + m3)  # special case to avoid divide by zero
        else
            t_z[ii] = (w1 * m2 + w2 * m3) / (w1 + w2)
        end
    end
    t = copy(t_z)

    # --- Polynomial coefficients ---
    for ii in 1:npt-1
        dx = xpt[ii+1] - xpt[ii]
        t1 = t[ii]
        t2 = t[ii+1]
        p0_z[ii] = ypt[ii]
        p1_z[ii] = t1
        p2_z[ii] = (3.0 * m[ii+2] - 2.0 * t1 - t2) / dx
        p3_z[ii] = (t1 + t2 - 2.0 * m[ii+2]) / dx^2
    end

    return copy(p0_z), copy(p1_z), copy(p2_z), copy(p3_z)
end

function interp_akima(npt, n, x, xpt, p0, p1, p2, p3,
    dp0dxpt, dp1dxpt, dp2dxpt, dp3dxpt, dp0dypt, dp1dypt, dp2dypt, dp3dypt,
)
    """
    Evaluate Akima spline and its derivatives

    Returns
    y - interpolated value
    dydx - derivative of y wrt x
    dydxpt, dydypt - derivative of y wrt xpt and ypt
    """
    # --- Outputs ---
    y = zeros(n)
    dydx = zeros(n)
    dydxpt = zeros(n, npt)
    dydypt = zeros(n, npt)
    # Zygote buffers
    y_z = Zygote.Buffer(y)
    dydx_z = Zygote.Buffer(dydx)
    dydxpt_z = Zygote.Buffer(dydxpt)
    dydypt_z = Zygote.Buffer(dydypt)


    # --- Interpolate at each point ---
    for ii in 1:n

        # --- Find location of spline in array (uses end segments if out of bounds) ---
        jj = 1 # give jj an initial value
        if x[ii] < xpt[1]
            jj = 1
        else
            # Linear search
            for jj in npt-1:-1:1
                if x[ii] >= xpt[jj]
                    break
                end
            end
        end

        # --- Evaluate poly and derivative ---
        dx = (x[ii] - xpt[jj])
        y_z[ii] = p0[jj] + p1[jj] * dx + p2[jj] * dx^2 + p3[jj] * dx^3
        dydx_z[ii] = p1[jj] + 2.0 * p2[jj] * dx + 3.0 * p3[jj] * dx^2


        for kk in 1:npt
            dydxpt_z[ii, kk] = dp0dxpt[jj, kk] + dp1dxpt[jj, kk] * dx + dp2dxpt[jj, kk] * dx^2 + dp3dxpt[jj, kk] * dx^3
            if (kk == jj)
                dydxpt_z[ii, kk] = dydxpt[ii, kk] - dydx_z[ii]
            end
            dydypt_z[ii, kk] = dp0dypt[jj, kk] + dp1dypt[jj, kk] * dx + dp2dypt[jj, kk] * dx^2 + dp3dypt[jj, kk] * dx^3
        end
    end

    return copy(y_z), copy(dydx_z), copy(dydxpt_z), copy(dydypt_z)
end

function do_akima_interp(xpt, ypt, xq, Δx=1e-7)
    npt = length(xpt)
    n = length(xq)
    p0, p1, p2, p3 = setup_akima(npt, xpt, ypt, Δx)
    zeros_in = zeros(npt - 1, npt)
    y, _, _, _ = interp_akima(npt, n, xq, xpt, p0, p1, p2, p3, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in, zeros_in)

    if n == 1 # need it returned as a float
        return y[1]
    else
        return y
    end
end

function do_linear_interp(xpt, ypt, xqvec)
    npt = length(xpt)
    n = length(xqvec)
    y = zeros(n)
    y_z = Zygote.Buffer(y)

    if length(xpt) != length(ypt)
        throw(ArgumentError("xpt and ypt must be the same length"))
    end


    for jj in 1:n

        xq = xqvec[jj]

        # Catch cases in case we're just outside the domain
        # This extends the slope of the first/last segment
        if xq <= xpt[1]
            x0 = xpt[1]
            x1 = xpt[2]
            y0 = ypt[1]
            y1 = ypt[2]
        elseif xq >= xpt[npt]
            x0 = xpt[npt-1]
            x1 = xpt[npt]
            y0 = ypt[npt-1]
            y1 = ypt[npt]
        else
            # Perform search
            ii = 1
            while xq > xpt[ii+1]
                ii += 1
            end

            x0 = xpt[ii]
            x1 = xpt[ii+1]
            y0 = ypt[ii]
            y1 = ypt[ii+1]

        end

        m = (y1 - y0) / (x1 - x0) # slope
        y_z[jj] = y0 + m * (xq - x0)
    end
    y = copy(y_z)

    if n == 1 # need it returned as a float
        return y[1]
    else
        return y
    end
end
# ==============================================================================
#                         Custom derivative routines
# ==============================================================================
function ChainRulesCore.rrule(::typeof(cmplxStdEigValProb2), A_r, A_i, n)

    # TODO: redo using Sicheng and Eirikur's eigenvalue derivative method 
    # that does not require the full eigenvalue solve

    y = cmplxStdEigValProb2(A_r, A_i, n)

    function cmplxStdEigen_pullback(y)
        """
        See cmplxStdEigValProb_b()
        """
        # We unpack the y vector into the real and imaginary parts of the eigenvalues and eigenvectors
        w̄ = y[1:n] + 1im * y[n+1:2*n]
        vr_r = reshape(y[2*n+1:2*n+n^2], n, n)
        vr_i = reshape(y[2*n+n^2+1:end], n, n)
        VR̄ = vr_r + 1im * vr_i
        # No need to transpose to get the right shape even though julia is column major
        # --- Initialize matrices ---
        E = zeros(ComplexF64, n, n)
        F = zeros(ComplexF64, n, n)
        A = A_r + 1im * A_i

        # --- Solve standard eigenvalue problem (Ax = λx) ---
        # This method uses the julia built-in eigendecomposition
        # eigen() is a spectral decomposition
        w, Vr = eigen(A)

        # ---------------------------
        #   Hermitian transpose (U^-H) conj transpose
        # ---------------------------
        VrHerm = Vr'
        VrHerminv_r, VrHerminv_i = cmplxInverse(real(VrHerm), imag(VrHerm), n)

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

        # --- Calculate F ∘ (U^H * Ū) ---
        tmp1 = F .* (VrHerm * VR̄)

        # Add D̄
        for ii = 1:n
            tmp1[ii, ii] += w̄[ii]
        end

        Ā = ((VrHerminv_r + 1im * VrHerminv_i) * tmp1) * VrHerm
        Ā_r = real(Ā)
        Ā_i = imag(Ā)

        # Return NoTangent() because??
        return (NoTangent(), Ā_r, Ā_i, NoTangent())

    end # cmplxStdEigValProb_b


    return y, cmplxStdEigen_pullback
end

function ChainRulesCore.rrule(::typeof(*), A::Matrix{<:RealOrComplex}, B::Matrix{<:RealOrComplex})
    """ 
    MATRIX MULTIPLY
    """
    function times_pullback(ΔΩ)
        ∂A = @thunk(ΔΩ * B')
        ∂B = @thunk(A' * ΔΩ)
        return (NoTangent(), ∂A, ∂B)
    end
    return A * B, times_pullback
end

function ChainRulesCore.rrule(::typeof(inv), A::Matrix{<:RealOrComplex})
    """
    MATRIX INVERSE
    """
    Ω = inv(A)

    function inv_pullback(ΔΩ)

        ∂A = -Ω' * ΔΩ * Ω'

        return (NoTangent(), ∂A)
    end

    return Ω, inv_pullback
end

end # SolverRoutines