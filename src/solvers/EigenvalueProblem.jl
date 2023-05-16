module EigenvalueProblem

# --- Libraries ---
using LinearAlgebra
using ChainRulesCore
using ForwardDiff

function compute_eigsolve(K, M, nEig; issym=true)

    # TODO: ok this is most definitely NOT the way it should be in the production version because this
    # is super expensive but whatever
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


end # end module