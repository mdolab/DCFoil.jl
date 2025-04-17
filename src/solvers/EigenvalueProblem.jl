
function compute_eigsolve(K::Matrix{<:RealOrComplex}, M::Matrix{<:RealOrComplex}, nEig::Int; issym=true)

    # TODO: ok this is most definitely NOT the way it should be in the production version because this
    # is super expensive but whatever
    # A = similar(K) # uninitialized matrix
    # println("A: ", typeof(A))
    A::typeof(K) = inv(M) * K # Applied mathematicians are probably rolling over in their graves right now

    eValsAll = eigvals(A)
    eVecsAll = eigvecs(A)

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
