module EigenvalueProblem

# --- Libraries ---
using LinearAlgebra
using ChainRulesCore
using ForwardDiff, DiffRules

function compute_eigsolve(K, M, nEig; issym=true)

    # TODO: ok this is most definitely NOT the way it should be in the production version because this
    # is super expensive but whatever
    # eigsorter = :SR # smallest real part
    # eigsorter = KrylovKit.EigSorter(abs; rev=false)
    # I THINK THE CODE HERE IS BROKEN TRY OTEHRS
    # eVals, eVecs, info = KrylovKit.geneigsolve((K, M), nEig, which; krylovdim=3 * nEig, maxiter=1000, issymmetric=true, verbosity=1)
    # eVals, eVecs, info = KrylovKit.eigsolve(inv(M) * K, nEig, eigsorter; krylovdim=2 * nEig, maxiter=1000, verbosity=1, issymmetric=true) # not good sorting
    A = inv(M) * K # Applied mathematics are probably rolling over in their graves right now
    eValsAll = my_eigvals(A)
    eVecsAll = my_eigvecs(A)

    if issym
        if typeof(eValsAll) <: ForwardDiff.Dual
            eVals = eValsAll[1][1:nEig]
            eVecs = eVecsAll[:, 1:nEig]
        else
            eVals = eValsAll[1:nEig]
            eVecs = eVecsAll[:, 1:nEig]
        end
    else
        # Only take positive imag value eigenvalues
        indices = findall(x -> (imag(x) > 0.0), eValsAll)
        eVals = eValsAll[indices]
        eVecs = eVecsAll[:, indices]
    end

    return eVals[1:nEig], eVecs[:, 1:nEig]
end

# ==============================================================================
#                         Take advantage of multiple dispatch
# ==============================================================================
# function my_eigvals_dual(::Type{T}, dual_args...) where {T<:ForwardDiff.Dual}
#     ȧrgs = (NO_FIELDS, partials.(dual_args)...)
#     args = (my_eigvals, value.(dual_args)...)
#     y, ẏ = ChainRulesCore.frule(ȧrgs, args...)
#     T(y, ẏ)
# end

function my_eigvals(A::Matrix{T1}) where {T1}

    T = promote_type(T1)
    if T <: ForwardDiff.Dual
        # --- Loop matrix and get values and partials from dual number ---
        Avals = zeros(size(A))
        Apartials = zeros(size(A))
        for i in 1:size(A)[1]
            for j in 1:size(A)[2]
                Avals[i, j] = (A[i, j].value)
                Apartials[i, j] = 0.0
            end
        end
        wd, _ = eigenderiv(Avals, Apartials)
        return eigvals(Avals)
    else
        eigvals(A)
    end
end

function my_eigvecs(A::Matrix{T1}) where {T1}

    T = promote_type(T1)
    if T <: ForwardDiff.Dual
        # --- Loop matrix and get values and partials from dual number ---
        Avals = zeros(size(A))
        Apartials = zeros(size(A))
        for i in 1:size(A)[1]
            for j in 1:size(A)[2]
                Avals[i, j] = (A[i, j].value)
                Apartials[i, j] = 0.0
            end
        end
        _, Vrd = eigenderiv(Avals, Apartials)
        return eigvecs(Avals)
    else
        eigvecs(A)
    end
end

# function ChainRulesCore.frule((_, ΔA), ::typeof(my_eigvals), A)
#     """
#     This gets called instead of 'eigvals()' in ForwardDiff mode.
#     The Δself arg is blank

#     Forward mode analytic derivative for the standard eigenvalue problem [A] v= λ v
#     with eigenvalues d_k
#         Dd = I ∘ (U^-1 * Ad U)
#         Ud = U * (F ∘ U^-1 * Ad * U)
#     where F_ij = (d_j - d_i)^-1 for i != j and zero otherwise --> F_ij = E_ij^-1

#     'd' terms are the forward seeds

#     See: 
#         Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation
#     """

#     # --- Initialize matrices ---
#     n = size(A)[1]
#     w_d = zeros(n)
#     E = zeros(ComplexF64, n, n)
#     F = zeros(ComplexF64, n, n)
#     A = A_r + 1im * A_i
#     Ad = A_rd + 1im * A_id

#     # --- Solve standard eigenvalue problem (Ax = λx) ---
#     # This method uses the julia built-in eigendecomposition
#     # eigen() is a spectral decomposition
#     w, Vr = eigen(A)

#     # ---------------------------
#     #   Eigenvalue derivatives Dd
#     # ---------------------------
#     # --- Compute eigenvector inverses U^-1 ---
#     Vrinv = inv(Vr)

#     # --- Compute U^-1 * Ad * U ---
#     tmp1 = (Vrinv * Ad) * Vr

#     # Don't do Hadamard product with identity matrix.
#     for ii = 1:n
#         w_d[ii] = (tmp1[ii, ii])
#     end

#     return (w, w_d)
# end

# function ChainRulesCore.frule((_, ΔA), ::typeof(my_eigvecs), A)
#     """
#     Forward mode analytic derivative for the standard eigenvalue problem [A] v= λ v
#     with eigenvalues d_k
#         Dd = I ∘ (U^-1 * Ad U)
#         Ud = U * (F ∘ U^-1 * Ad * U)
#     where F_ij = (d_j - d_i)^-1 for i != j and zero otherwise --> F_ij = E_ij^-1

#     'd' terms are the forward seeds

#     See: 
#         Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation
#     """

#     # --- Initialize matrices ---
#     n = size(A)[1]
#     w_d = zeros(n)
#     E = zeros(ComplexF64, n, n)
#     F = zeros(ComplexF64, n, n)
#     A = A_r + 1im * A_i
#     Ad = A_rd + 1im * A_id

#     # --- Solve standard eigenvalue problem (Ax = λx) ---
#     # This method uses the julia built-in eigendecomposition
#     # eigen() is a spectral decomposition
#     w, Vr = eigen(A)

#     # --- Compute eigenvector inverses U^-1 ---
#     Vrinv = inv(Vr)

#     # --- Compute U^-1 * Ad * U ---
#     tmp1 = (Vrinv * Ad) * Vr

#     # ---------------------------
#     #   Eigenvector derivatives U̇
#     # ---------------------------
#     # TODO: these don't work apparently
#     # --- E ---
#     for jj in 1:n
#         for ii in 1:n
#             E[ii, jj] = w[jj] - w[ii]
#         end
#     end

#     # --- F ---
#     for jj in 1:n
#         for ii in 1:n
#             if jj != ii
#                 F[ii, jj] = 1.0 / E[ii, jj]
#             end
#         end
#     end

#     # --- F ∘ (U^-1 * Ad * U) ---
#     tmp2 = F .* tmp1

#     # --- Final U * (F ∘ (U^-1 * Ad * U)) ---
#     Vrd = Vr * tmp2

#     return Vr, Vrd
# end


DiffRules.@define_diffrule LinearAlgebra.eigvals(A) = :(eigenderiv($A, ones($A)))

function eigenderiv(A, Ad)
    """
    This gets called instead of 'eigvals()' in ForwardDiff mode.
    The Δself arg is blank

    Forward mode analytic derivative for the standard eigenvalue problem [A] v= λ v
    with eigenvalues d_k
        Dd = I ∘ (U^-1 * Ad U)
        Ud = U * (F ∘ U^-1 * Ad * U)
    where F_ij = (d_j - d_i)^-1 for i != j and zero otherwise --> F_ij = E_ij^-1

    'd' terms are the forward seeds

    See: 
        Giles, M. (2008). An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation
    """

    # --- Initialize matrices ---
    n = size(A)[1]
    w_d = zeros(n)
    E = zeros(ComplexF64, n, n)
    F = zeros(ComplexF64, n, n)

    # --- Solve standard eigenvalue problem (Ax = λx) ---
    # This method uses the julia built-in eigendecomposition
    # eigen() is a spectral decomposition
    w, Vr = eigen(A)

    # ---------------------------
    #   Eigenvalue derivatives Dd
    # ---------------------------
    # --- Compute eigenvector inverses U^-1 ---
    Vrinv = inv(Vr)

    # --- Compute U^-1 * Ad * U ---
    tmp1 = (Vrinv * Ad) * Vr

    # Don't do Hadamard product with identity matrix.
    for ii = 1:n
        w_d[ii] = (tmp1[ii, ii])
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


    return w_d, Vrd
end

end # end module