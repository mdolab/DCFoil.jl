"""
Generic routines every solver needs

NOTE:
any function with '_d' at the end is the one used for forward differentiation
because certain operations cannot be differentiated
by the AD tool (e.g., anything related to file writing)
'_b' is for backward mode.
In julia, the chainrules rrule is '_b'
"""

# --- PACKAGES ---
# using LinearAlgebra
# using Zygote
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent
# using FLOWMath: abs_cs_safe, atan_cs_safe
# using Printf
# using Debugger

# --- DCFoil modules ---
# using ..NewtonRaphson
# using ..EigenvalueProblem

# --- Globals ---
# using ..EBBeam: EBBeam as BeamElement
# using ..DesignConstants: SORTEDDVS
# using ..DCFoil: RealOrComplex, DTYPE
# const RealOrComplex = Union{Real, Complex}
# const DTYPE = AbstractFloat
# using ..Rotations: get_rotate3dMat

# ==============================================================================
#                         Solver routines
# ==============================================================================
function converge_resNonlinear(compute_residuals, compute_∂r∂u, u0::Vector, x0List=nothing;
    maxIters=50, tol=1e-6, is_verbose=false,
    solverParams=nothing,
    appendageOptions=Dict(),
    solverOptions=Dict(),
    mode="Analytic",
    is_cmplx=false,
    iComp=1, CLMain=0.0
)
    """
    Given input u, solve the system r(u) = 0
    Tells you how many NL iters

    Parameters:
    -----------
    compute_residuals : function handle
    compute_∂r∂u : function handle
        Residual Jacobian
    mode : String
        Analytic, CS, RAD, or FiDi

    """
    if !isnothing(x0List)
        x0 = x0List[iComp]
    end
    # ************************************************
    #     Main solver loop
    # ************************************************
    @ignore_derivatives() do
        if is_verbose
            println("+", "-"^50, "+")
            println("|              Beginning NL solve                  |")
            println("+", "-"^50, "+")
            println(@sprintf("Residual Jacobian computed via the %s mode", uppercase(mode)))
        end
    end
    # Somewhere here, you could do something besides Newton-Raphson if you want
    converged_u, converged_r, iters = NewtonRaphson.do_newton_raphson(
        compute_residuals, compute_∂r∂u, u0, x0List;
        maxIters, tol, is_verbose, mode, is_cmplx,
        solverParams=solverParams, appendageOptions=appendageOptions, solverOptions=solverOptions, iComp=iComp, CLMain=CLMain)

    return converged_u, converged_r

end

function return_totalStates(foilStructuralStates, DVDict, elemType="BT2"; appendageOptions=Dict(), alphaCorrection=0.0)
    """
    Returns the deflected + rigid shape of the foil
    So like pre-twist
    Inputs
    ------
        foilStructuralStates - structural states of the foil in global ref frame!
        alphaCorrection - correction to alpha in deg
        elemType - element type
    Outputs
    -------
        foilTotalStates - total states of the foil in global reference frame
        nDOF - number of DOF per node
    """

    alfaRad = deg2rad(DVDict["alfa0"]) + alphaCorrection
    rakeRad = deg2rad(DVDict["rake"])
    betaRad = deg2rad(DVDict["beta"])
    nDOF = BeamElement.NDOF

    # Get flow angles of attack in "local" beam coords first
    #TODO: pretwist will change this
    if elemType == "BT2"
        nGDOF = nDOF * 3 # number of DOFs on node in global coordinates
        staticOffset = [0, 0, alfaRad, 0]
    elseif elemType == "COMP2"
        staticOffset_wing = [0, 0, 0, alfaRad + rakeRad, 0, betaRad, 0, 0, 0]
        staticOffset_strut = [0, 0, 0, betaRad, 0, rakeRad, 0, 0, 0]
    end

    # ---------------------------
    #   Transformation into global ref frame
    # ---------------------------
    if elemType == "COMP2"
        angleDefault = deg2rad(90) # default angle of rotation of the axes from global wing to match local beam
        axisDefault = "z"
        T1 = get_rotate3dMat(angleDefault, axisDefault)
        T = T1
        Z = zeros(3, 3)
        transMatL2G = [
            T Z Z
            Z T Z
            Z Z T
        ]
        staticOffset_wing = transMatL2G * staticOffset_wing
        staticOffset_junctionNode = staticOffset_wing
        if appendageOptions["config"] == "t-foil"
            angleDefault = deg2rad(-90)
            axisDefault = "x"
            T2 = get_rotate3dMat(angleDefault, axisDefault)
            T = T2 * T1
            transMatL2G = [
                T Z Z
                Z T Z
                Z Z T
            ]
            staticOffset_strut = transMatL2G * staticOffset_strut

            staticOffset_junctionNode = staticOffset_wing

        elseif appendageOptions["config"] == "full-wing" || appendageOptions["config"] == "wing"

            staticOffset_junctionNode = staticOffset_wing

        end
    else
        angleDefault = 0.0
    end

    # In the following formulation, we assume junction node is always first!
    nStrutDOFs = 0
    staticOffsetGlobalRef_strut = []
    if appendageOptions["config"] == "t-foil"
        nStrutDOFs = (appendageOptions["nNodeStrut"] - 1) * nDOF # subtract 1 because of the junction node
        w_strut = foilStructuralStates[end-nStrutDOFs+1:nDOF:end]
        staticOffsetGlobalRef_strut = repeat(staticOffset_strut, outer=[length(w_strut)])
    end
    w_wing = foilStructuralStates[1:nDOF:end-nStrutDOFs] # These wing DOFS include the junction node

    # # Correct the root "junction" node
    staticOffsetGlobalRef_wing = vcat(staticOffset_junctionNode, repeat(staticOffset_wing, outer=[length(w_wing) - 1]))
    # staticOffsetGlobalRef_wing[1:nDOF] = staticOffset_junctionNode

    staticOffsetGlobalRef = vcat(staticOffsetGlobalRef_wing, staticOffsetGlobalRef_strut)

    # Add static angle of attack to deflected foil
    # AKA jig shape
    foilTotalStates = copy(foilStructuralStates) + staticOffsetGlobalRef


    return foilTotalStates
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

function ChainRulesCore.rrule(::typeof(cmplxStdEigValProb2), A_r::Matrix, A_i::Matrix, n::Int)
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

function cross3D(arr1, arr2)
    """
    Cross product of two 3D arrays
    where the first dimension is length 3
    """
    @assert size(arr1, 1) == 3
    @assert size(arr2, 1) == 3
    M, N = size(arr1, 2), size(arr1, 3)

    arr1crossarr2 = zeros(RealOrComplex, 3, M, N)
    # arr1crossarr2 = zeros(DTYPE, 3, M, N) # doesn't actually affect the result
    arr1crossarr2_z = Zygote.Buffer(arr1crossarr2)

    for jj in 1:M
        for kk in 1:N
            arr1crossarr2_z[:, jj, kk] = cross(arr1[:, jj, kk], arr2[:, jj, kk])
        end
    end
    arr1crossarr2 = copy(arr1crossarr2_z)

    return arr1crossarr2

end
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
    taux = get_rotate3dMat(phi, "x")
    tauy = get_rotate3dMat(theta, "y")
    tauz = get_rotate3dMat(psi, "z")

    if rotType == 1
        RMat = taux * tauy * tauz
    elseif rotType == 2
        RMat = tauz * tauy * taux
    else
        error("Only 3-2-1 rotation implemented")
    end

    return RMat
end




# function do_linear_interp(xpt, ypt, xqvec)
#     """
#     KNOWN BUG, DOES NOT LIKE NEGATIVE DOMAINS
#     """
#     npt = length(xpt)
#     n = length(xqvec)
#     y = zeros(RealOrComplex, n)
#     y_z = Zygote.Buffer(y)
#     if length(xpt) != length(ypt)
#         throw(ArgumentError("xpt and ypt must be the same length"))
#     end
#     loop_interp!(y_z, xpt, ypt, xqvec, n, npt)
#     y = copy(y_z)
#     if n == 1 # need it returned as a float
#         return y[1]
#     else
#         return y
#     end
# end

# function do_linear_interp(xpt::Vector, ypt::Vector, xqvec)
#     npt = length(xpt)
#     n = length(xqvec)
#     y = zeros(RealOrComplex, n)
#     if length(xpt) != length(ypt)
#         throw(ArgumentError("xpt and ypt must be the same length"))
#     end
#     loop_interp!(y, xpt, ypt, xqvec, n, npt)
#     if n == 1 # need it returned as a float
#         return y[1]
#     else
#         return y
#     end
# end

# function loop_interp!(y, xpt, ypt, xqvec, n, npt)
#     for jj in 1:n
#         @inbounds @fastmath begin
#             xq = xqvec[jj]

#             # Catch cases in case we're just outside the domain
#             # This extends the slope of the first/last segment
#             if real(xq) <= real(xpt)[1]
#                 x0 = xpt[1]
#                 x1 = xpt[2]
#                 y0 = ypt[1]
#                 y1 = ypt[2]
#             elseif real(xq) >= real(xpt)[npt]
#                 x0 = xpt[npt-1]
#                 x1 = xpt[npt]
#                 y0 = ypt[npt-1]
#                 y1 = ypt[npt]
#             else
#                 # Perform search
#                 ii = 1
#                 while real(xq) > real(xpt)[ii+1]
#                     ii += 1
#                 end

#                 x0 = xpt[ii]
#                 x1 = xpt[ii+1]
#                 y0 = ypt[ii]
#                 y1 = ypt[ii+1]

#             end

#             m = (y1 - y0) / (x1 - x0) # slope
#             y[jj] = y0 + m * (xq - x0)

#             # # actually just use end value if we're at the end
#             # if real(xq) >= real(xpt)[npt]
#             #     y[jj] = ypt[npt]
#             # elseif real(xq) <= real(xpt)[1]
#             #     y[jj] = ypt[1]
#             # end
#         end
#     end
# end

function normalize_3Dvector(r)
    rhat = r ./ √(r[XDIM]^2 + r[YDIM]^2 + r[ZDIM]^2)
    return rhat
end
