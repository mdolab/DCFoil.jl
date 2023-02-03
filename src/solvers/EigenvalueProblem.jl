module EigenvalueProblem

# --- Libraries ---
using LinearAlgebra

function compute_eigsolve(K, M, nEig; issym=true)

    # TODO: ok this is most definitely NOT the way it should be in the production version because this is super expensive but whatever
    # eigsorter = :SR # smallest real part
    # eigsorter = KrylovKit.EigSorter(abs; rev=false)
    # I THINK THE CODE HERE IS BROKEN TRY OTEHRS
    # eVals, eVecs, info = KrylovKit.geneigsolve((K, M), nEig, which; krylovdim=3 * nEig, maxiter=1000, issymmetric=true, verbosity=1)
    # eVals, eVecs, info = KrylovKit.eigsolve(inv(M) * K, nEig, eigsorter; krylovdim=2 * nEig, maxiter=1000, verbosity=1, issymmetric=true) # not good sorting
    A = inv(M) * K # Applied mathematics are probably rolling over in their graves right now
    eValsAll = eigvals(A)
    eVecsAll = eigvecs(A)

    if issym
        eVals = eValsAll[1:nEig]
        eVecs = eVecsAll[:, 1:nEig]
    else
        # Only take positive imag value eigenvalues
        # TODO:
        indices = findall(x -> (imag(x) > 0.0), eValsAll)
        eVals = eValsAll[indices]
        eVecs = eVecsAll[:, indices]
    end

    return eVals[1:nEig], eVecs[:, 1:nEig]
end

end # end module