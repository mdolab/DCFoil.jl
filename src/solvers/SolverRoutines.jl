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
using Zygote, ChainRulesCore

include("./NewtonRaphson.jl")
using .NewtonRaphson

include("./EigenvalueProblem.jl")
using .EigenvalueProblem

include("../adrules/CustomRules.jl")
using .CustomRules

# --- Globals ---
include("../constants/SolutionConstants.jl")
using .SolutionConstants: XDIM, YDIM, ZDIM
include("../struct/EBBeam.jl")
using .EBBeam: EBBeam as BeamElement


# ==============================================================================
#                         Solver routines
# ==============================================================================
function converge_r(compute_residuals, compute_∂r∂u, u; maxIters=200, tol=1e-6, is_verbose=true, 
    mode="analytic", 
    # mode="RAD",
    is_cmplx=false
    )
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

    # Somewhere here, you could do something besides Newton-Raphson if you want
    converged_u, converged_r, iters = NewtonRaphson.do_newton_raphson(compute_residuals, compute_∂r∂u, u, maxIters, tol, is_verbose, mode, is_cmplx)

    return converged_u, converged_r

end # converge_r

function return_totalStates(foilStructuralStates, α₀, elemType="BT2"; STRUT=nothing, solverOptions=Dict())
    """
    Returns the deflected + rigid shape of the foil
    Inputs
    ------
        foilStructuralStates - structural states of the foil in global ref frame!
        α₀ - angle of attack
        elemType - element type
        beta - yaw angle
    Outputs
    -------
        foilTotalStates - total states of the foil in global reference frame
        nDOF - number of DOF per node
    """

    # alfaRad = α₀ * π / 180
    alfaRad = deg2rad(α₀)
    if STRUT != nothing
        beta = STRUT.α₀
    else
        beta = 0.0
    end
    betaRad = deg2rad(beta)
    nDOF = BeamElement.NDOF
    # Get flow angles of attack in local beam coords first
    #TODO: pretwist will change this
    if elemType == "bend"
        error("Only bend-twist element type is supported for load computation")
    elseif elemType == "bend-twist"
        nDOF = 3
        staticOffset = [0, 0, alfaRad]
    elseif elemType == "BT2"
        nDOF = 4
        nGDOF = nDOF * 3 # number of DOFs on node in global coordinates
        staticOffset = [0, 0, alfaRad, 0] 
    elseif elemType == "COMP2"
        staticOffset_wing = [0, 0, 0, alfaRad, 0, 0, 0, 0, 0]
        staticOffset_strut = [0, 0, 0, betaRad, 0, 0, 0, 0, 0]
    end

    # ---------------------------
    #   Transformation into global ref frame
    # ---------------------------
    if elemType == "COMP2"
        angleDefault = deg2rad(90) # default angle of rotation of the axes from global wing to match local beam
        axisDefault = "z"
        T1 = get_rotate3dMat(angleDefault, axis=axisDefault)
        T = T1
        Z = zeros(3, 3)
        transMatL2G = [
            T Z Z
            Z T Z
            Z Z T
        ]
        staticOffset_wing = transMatL2G * staticOffset_wing
        staticOffset_junctionNode = staticOffset_wing
        if solverOptions["config"] == "t-foil"
            # TODO: MAKE IT SO ALL WING NODES ARE YAWED ALSO
            angleDefault = deg2rad(-90)
            axisDefault = "x"
            T2 = get_rotate3dMat(angleDefault, axis=axisDefault)
            T = T2*T1
            transMatL2G = [
                T Z Z
                Z T Z
                Z Z T
            ]
            staticOffset_strut = transMatL2G * staticOffset_strut

            staticOffset_junctionNode = staticOffset_strut + staticOffset_wing
        elseif solverOptions["config"] == "full-wing" || solverOptions["config"] == "wing"
            staticOffset_junctionNode = staticOffset_wing
        end
    else
        angleDefault = 0.0
    end

    # In the following formulation, we assume junction node is always first!
    nStrutDOFs = 0
    staticOffsetGlobalRef_strut = []
    if solverOptions["config"] == "t-foil"
        nStrutDOFs = (solverOptions["nNodeStrut"]-1)*nDOF # subtract 1 because of the junction node
        w_strut = foilStructuralStates[end-nStrutDOFs+1:nDOF:end]
        staticOffsetGlobalRef_strut = repeat(staticOffset_strut, outer=[length(w_strut)])
    end
    w_wing = foilStructuralStates[1:nDOF:end-nStrutDOFs] # These wing DOFS include the junction node
    
    # # Correct the root "junction" node
    staticOffsetGlobalRef_wing = vcat(staticOffset_junctionNode, repeat(staticOffset_wing, outer=[length(w_wing)-1]))
    # staticOffsetGlobalRef_wing[1:nDOF] = staticOffset_junctionNode

    staticOffsetGlobalRef = vcat(staticOffsetGlobalRef_wing, staticOffsetGlobalRef_strut)

    # Add static angle of attack to deflected foil
    foilTotalStates = copy(foilStructuralStates) + staticOffsetGlobalRef


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

function cmplxInverse(A_r::Matrix{Float64}, A_i::Matrix{Float64}, n::Int64)
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
    RHS = Matrix{Int64}(I, 2 * n, 2 * n)

    # --- Then solve linear system with multiple RHS's ---
    Ainvcopy = luA \ RHS

    # --- Extract solution ---
    @inbounds begin
        Ainv_r = Ainvcopy[1:n, 1:n]

        # TODO: you could add the option for transpose here like Eirikur
        Ainv_i = Ainvcopy[n+1:end, 1:n]
    end
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

function cmplxStdEigValProb2(A_r, A_i, n; nEigs=10)
    """
    Give back eigenvalues and vectors as an unrolled vector
    where the real and imaginary parts are concatenated (eigenvalues first, then eigenvectors)
    """

    # --- Solve standard eigenvalue problem (Ax = λx) ---
    # This method uses the julia built-in eigendecomposition
    # eigen() is a spectral decomposition
    @fastmath begin
        A = A_r + 1im * A_i
        
        w, Vr = eigen(A)
        # TODO: swap this out with a more efficient eigen method
        # w, Vr = lanczos(A, nEigs)
    end
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

function ChainRulesCore.rrule(::typeof(cmplxStdEigValProb2), A_r, A_i, n)
    """
    Reverse rule for the eigenvalue problem
    """
    
    # Primal function
    y = cmplxStdEigValProb2(A_r, A_i, n)

    function cmplxStdEigen_pullback(yb)
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
        # We unpack the y vector into the real and imaginary parts of the eigenvalues and eigenvectors
        w̄ = yb[1:n] + 1im * yb[n+1:2*n]
        vr_r = reshape(yb[2*n+1:2*n+n^2], n, n)
        vr_i = reshape(yb[2*n+n^2+1:end], n, n)
        VR̄ = vr_r + 1im * vr_i
        # No need to transpose to get the right shape even though julia is column major
        # --- Initialize matrices ---
        E = zeros(ComplexF64, n, n)
        F = zeros(ComplexF64, n, n)
        A = A_r + 1im * A_i

        # --- Solve standard eigenvalue problem (Ax = λx) ---
        # This method uses the julia built-in eigendecomposition
        # eigen() is a spectral decomposition and is allegedly the fastest
        w, Vr = eigen(A)
        # TODO: swap this out with a more efficient eigen method using Sicheng and Eirikur's eigenvalue derivative method
        # Paper: 
        # that does not require all eigenvalues and vectors

        # ---------------------------
        #   Hermitian transpose (U^-H) conj transpose
        # ---------------------------
        VrHerm = Vr'
        VrHerminv_r, VrHerminv_i = cmplxInverse(real(VrHerm), imag(VrHerm), n)

        # --- E ---
        # @simd for jj in 1:n SIMD MADE IT SLOWER THUS DEFAULT VECTORIZATIONS ARE GOOD
        for jj in 1:n
            for ii in 1:n
                @inbounds @fastmath E[ii, jj] = w[jj] - w[ii]
            end
        end

        # --- F ---
        # @simd for jj in 1:n
        for jj in 1:n
            for ii in 1:n
                if ii != jj
                    @inbounds @fastmath F[ii, jj] = 1.0 / E[ii, jj]
                end
            end
        end

        # --- Calculate F ∘ (U^H * Ū) ---
        @fastmath tmp1 = F .* (VrHerm * VR̄)

        # Add D̄
        # @simd for ii = 1:n
        for ii = 1:n
            @inbounds @fastmath tmp1[ii, ii] += w̄[ii]
        end

        Ā = ((VrHerminv_r + 1im * VrHerminv_i) * tmp1) * VrHerm
        Ā_r = real(Ā)
        Ā_i = imag(Ā)

        # Return NoTangent() because of order of args in this parent function
        return (NoTangent(), Ā_r, Ā_i, NoTangent())

    end # cmplxStdEigValProb_b


    return y, cmplxStdEigen_pullback
end

# ==============================================================================
#                         Utility routines
# ==============================================================================
function lagrangeArrInterp(x0, y0, m::Int64, n::Int64, d::Int64, x)
    """
    Interpolate/extrapolate polynomials of order 'd-1'
    Providing 'd' points of array of size m x n, we obtain inter/extrapolant order 'd-1'
    Comes from Eirikur's DLM4PY code

    Inputs
    ------
        x0 - input array size(d) (domain)
        y0 - input array y0(x0) size(m,n,d) (values)
        m, n  - size of array
        d - number of points to use for interpolation
        x  - the location we want to inter/extrapolate at - scalar
    Outputs
    -------
        y  - the inter/extrapolated array at x, or y(x)
    """

    # 2 dimensional array interpolation
    y::Matrix{Float64} = zeros(m, n)

    # @simd for ii in 1:d
    @inbounds @fastmath begin
        for ii in 1:d
            L = 1.0
            for jj in 1:d
                if jj != ii
                    L *= (x - x0[jj]) / (x0[ii] - x0[jj])
                end
            end
            y += y0[:, :, ii] .* L
        end
    end

    return y
end

function lagrangeInterp(x0, y0, n, x)
    """
    Interpolate/extrapolate polynomial of order 'm'
    Providing 'n' points gives us inter/extrapolant of order m = n-1

    Inputs
    ------
        x0 - input vector
        y0 - input vector y0(x0)
        n  - size of array
        x  - the location we want to inter/extrapolate at
    Outputs
    -------
        y  - the inter/extrapolated value at x, or y(x)
    """
    y = 0.0

    for ii in 1:n # loop over points
        L = 1.0 # Lagrange weight
        for kk in 1:n
            if kk != ii
                # This is the lagrange polynomial
                L *= (x - x0[kk]) / (x0[ii] - x0[kk])
            end
        end
        y += y0[ii] * L
    end

    return y
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
    locs_z = Zygote.Buffer(locs)

    for col in 1:ncol
        locs_z[col] = argmax(A[:, col])
    end
    locs = copy(locs_z)

    # # List comprehension to avoid Zygote.Buffer
    # locs = argmax(A[:, col] for col in 1:ncol)

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

function find_signChange(x::Vector{Float64})
    """
    Find the location where a sign changes in an array
    Inputs
    ------
        x - array which signchange is to be found. Size(n)
    Outputs
    -------
        locs - array of size 2 containing the location of the sign change
    """

    # Get signs of each element in x
    sgn = sign.(x)
    n = length(sgn)

    for ii in 1:n-1
        @fastmath @inbounds begin
            if sgn[ii+1] != sgn[ii]
                return ii, ii+1
            else
                continue
            end
        end
    end

end

function get_rotate3dMat(rot; axis="x")
    """
    Give rotation matrix about axis by rot radians (RH rule!)
    """
    rotMat = zeros(Float64, 3, 3)
    # rotMat = @SMatrix zeros(Float64, 3, 3)
    c::Float64 = cos(rot)
    s::Float64 = sin(rot)
    if axis == "x"
        rotMat = [
            1 0 0
            0 c -s
            0 s c
        ]
    elseif axis == "y"
        rotMat = [
            c 0 s
            0 1 0
            -s 0 c
        ]
    elseif axis == "z"
        rotMat = [
            c -s 0
            s c 0
            0 0 1
        ]
    else
        println("Only axis rotation implemented")
    end
    return rotMat
end

function transform_euler_ang(phi, theta, psi; rotType=1)
    """
    Parameters
    ----------
    phi : float
        Rotation about x axis, radians
    theta : float
        Rotation about y axis
    rotType : int, by default 1
        1 - This is 3-2-1 rotation (yaw, pitch, roll)
        2 - This is 1-2-3 rotation (roll, pitch, yaw)
    """
    taux = get_rotate3dMat(phi, axis="x")
    tauy = get_rotate3dMat(theta, axis="y")
    tauz = get_rotate3dMat(psi, axis="z")

    if rotType == 1
        RMat = taux * tauy * tauz
    elseif rotType == 2
        RMat = tauz * tauy * taux
    else
        error("Only 3-2-1 rotation implemented")
    end

    return RMat
end


function get_transMat(dR1::Float64, dR2::Float64, dR3::Float64, l::Float64, elemType="BT2")
    """
    Returns the transformation matrix for a given element type into 3D space

    Inputs
    -------
        dR: vector along beam length
        l: length of element
        elemType: element type
    """
    # This line breaks when the vector is straight up and down
    # TODO: there is probably a better angle parametrization like Rodrigues or quaternions!
    if dR1 == 0.0 && dR2 == 0.0
        rxyz_div = 1.0 / sqrt(dR1^2 + dR2^2 + dR3^2)
        ca = dR1 * rxyz_div
        cb = dR2 * rxyz_div
        cc = dR3 * rxyz_div
        T1 = get_rotate3dMat(-deg2rad(90); axis="x")
        T2 = get_rotate3dMat(-deg2rad(90); axis="z")
        T = T2*T1
    else    
        # beta is the angle above the xy plane
        rxy_div::Float64 = 1 / sqrt(dR1^2 + dR2^2) # length of projection onto xy plane
        calpha::Float64 = dR1 * rxy_div
        salpha::Float64 = dR2 * rxy_div
        cbeta::Float64 = 1 / rxy_div / l
        sbeta::Float64 = dR3 / l
        # Direction cosine matrix to get from 3d space to having x-axis be the long dimension
        T =  [
            calpha*cbeta salpha calpha*sbeta
            -salpha*cbeta calpha -salpha*sbeta
            -sbeta 0.0 cbeta
        ]
    end

    Z = zeros(3, 3)

    if elemType == "BT2"
        # Because BT2 had reduced DOFs, we need to transform the reduced DOFs into 3D space which results in storing more numbers
        Γ = Matrix(I, 8, 8)
    elseif elemType == "bend-twist"
        Γ = Matrix(I, 6, 6)
    elseif elemType == "BT3"
        Γ = Matrix(I, 10, 10)
    elseif elemType == "bend"
        # 4x12
        Γ = [
            T Z Z Z
            Z T Z Z
            Z Z T Z
            Z Z Z T
        ]
        # Γ = Matrix(I, 4, 4)
    elseif elemType == "BEAM3D"
        # 12x12
        Γ = [
            T Z Z Z
            Z T Z Z
            Z Z T Z
            Z Z Z T
        ]
    elseif elemType == "COMP2"
        Γ = [
            T Z Z Z Z Z
            Z T Z Z Z Z
            Z Z T Z Z Z
            Z Z Z T Z Z
            Z Z Z Z T Z
            Z Z Z Z Z T
        ]
    else
        error("Unsupported element type")
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
    """
    KNOWN BUG, DOES NOT LIKE NEGATIVE DOMAINS
    """
    npt = length(xpt)
    n = length(xqvec)
    y = zeros(n)
    y_z = Zygote.Buffer(y)

    if length(xpt) != length(ypt)
        throw(ArgumentError("xpt and ypt must be the same length"))
    end


    for jj in 1:n
        @inbounds @fastmath begin
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
    end
    y = copy(y_z)

    if n == 1 # need it returned as a float
        return y[1]
    else
        return y
    end
end

end # SolverRoutines