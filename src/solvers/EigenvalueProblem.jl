module EigenvalueProblem

# --- PACKAGES ---
using LinearAlgebra

function compute_eigsolve(K, M, nEig; issym=true)

    # TODO: ok this is most definitely NOT the way it should be in the production version because this
    # is super expensive but whatever
    A::Matrix{Float64} = inv(M) * K # Applied mathematicians are probably rolling over in their graves right now
    # println("A: ", typeof(A))
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

    return eVals[1:nEig], eVecs[:, 1:nEig]
end

# ==============================================================================
#                         Take advantage of multiple dispatch
# ==============================================================================


end # end module