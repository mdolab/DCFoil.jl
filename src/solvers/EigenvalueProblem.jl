
using ChainRulesCore: ChainRulesCore, NoTangent
using ChainRules: ChainRules # rrule for eigen used through rd_rrule

function compute_eigsolve(K::AbstractMatrix{<:RealOrComplex}, M::AbstractMatrix{<:RealOrComplex}, nEig::Int; issym=true)

    # TODO: ok this is most definitely NOT the way it should be in the production version because this
    # is super expensive but whatever
    # A = similar(K) # uninitialized matrix
    # println("A: ", typeof(A))
    A::typeof(K) = inv(M) * K # Applied mathematicians are probably rolling over in their graves right now

    if is_tracked(A) # ReverseDiff cannot go through LAPACK, so use the ChainRules eigen rule
        eValsAll, eVecsAll = compute_eigenTracked(A)
    else
        eValsAll = eigvals(A)
        eVecsAll = eigvecs(A)
    end

    if issym
        eVals = eValsAll[1:nEig]
        eVecs = eVecsAll[:, 1:nEig]
    else
        # Only take positive imag value eigenvalues
        indices = findall(x -> (imag(x) > 0.0), eValsAll)
        eVals = eValsAll[indices]
        eVecs = eVecsAll[:, indices]
    end

    λ = eVals[1:nEig]
    RR = eVecs[:, 1:nEig]
    # LL = eigvecs(K', M')[:, 1:nEig]

    return λ, RR
end

function compute_eigenTracked(A)
    """
    eigen(A) for a ReverseDiff-tracked real matrix A
    Returns real eigenpairs when all eigenvalues are real, like eigvals/eigvecs do
    """
    n = size(A, 1)
    y = rd_rrule(compute_eigenParts, A)

    λ_r = y[1:n]
    λ_i = y[n+1:2*n]
    V_r = reshape(y[2*n+1:2*n+n^2], n, n)
    V_i = reshape(y[2*n+n^2+1:end], n, n)

    if all(iszero, nograd(λ_i))
        return λ_r, V_r
    else
        return λ_r + 1im * λ_i, V_r + 1im * V_i
    end
end

function compute_eigenParts(A::AbstractMatrix{<:Real})
    """
    eigen(A) packed into one real vector [real(λ); imag(λ); vec(real(V)); vec(imag(V))]
    """
    F = eigen(A)
    return vcat(real(F.values), imag(F.values), vec(real(F.vectors)), vec(imag(F.vectors)))
end

function ChainRulesCore.rrule(::typeof(compute_eigenParts), A::AbstractMatrix{<:Real})
    F, eigen_pullback = ChainRulesCore.rrule(eigen, A)
    y = vcat(real(F.values), imag(F.values), vec(real(F.vectors)), vec(imag(F.vectors)))
    n = size(A, 1)

    function eigenParts_pullback(ȳ)
        λb = ȳ[1:n] + 1im * ȳ[n+1:2*n]
        Vb = reshape(ȳ[2*n+1:2*n+n^2], n, n) + 1im * reshape(ȳ[2*n+n^2+1:end], n, n)
        if eltype(F.values) <: Real # all eigenpairs are real so the cotangents are too
            λb = real(λb)
            Vb = real(Vb)
        end
        ΔF = ChainRulesCore.Tangent{typeof(F)}(values=λb, vectors=Vb)
        _, Ab = eigen_pullback(ΔF)
        return NoTangent(), real(ChainRulesCore.unthunk(Ab))
    end

    return y, eigenParts_pullback
end
