# --- Julia ---
"""
@File    :   SolveFlutter.jl
@Time    :   2022/10/07
@Author  :   Galen Ng
@Desc    :   p-k method for flutter analysis
"""

module SolveFlutter
"""
Eigenvalue and eigenvector solution
"""

# --- Public functions ---
export solve

# --- Libraries ---
using LinearAlgebra, Statistics
using JSON
using Zygote
using Profile


# --- DCFoil modules ---
include("../InitModel.jl")
include("../struct/BeamProperties.jl")
include("../struct/FiniteElements.jl")
include("../hydro/Hydro.jl")
include("SolveStatic.jl")
include("../constants/SolutionConstants.jl")
include("./SolverRoutines.jl")
# then use them
using .InitModel, .Hydro, .StructProp
using .FEMMethods
using .SolveStatic
using .SolutionConstants
using .SolverRoutines

function solve(DVDict::Dict, outputDir::String, uSweep::StepRangeLen{Float64,Base.TwicePrecision{Float64}}, fSearch::StepRangeLen{Float64,Base.TwicePrecision{Float64}}; use_freeSurface=false, cavitation=nothing)
    """
    Use p-k method to find roots (p) to the equation
        (-p²[M]-p[C]+[K]){ũ} = {0}
    """

    # ************************************************
    #     Initialize
    # ************************************************
    global FOIL = InitModel.init_dynamic(fSearch, DVDict, uSweep=uSweep)
    nElem = FOIL.neval - 1

    println("====================================================================================")
    println("        BEGINNING FLUTTER SOLUTION")
    println("====================================================================================")
    # ---------------------------
    #   Assemble structure
    # ---------------------------
    elemType = "BT2"
    loadType = "force"

    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL)
    globalKs, globalMs, globalF = FEMMethods.assemble(structMesh, elemConn, FOIL, elemType, FOIL.constitutive)
    FEMMethods.apply_tip_load!(globalF, elemType, loadType)

    # ---------------------------
    #   Apply BC blanking
    # ---------------------------
    globalDOFBlankingList = FEMMethods.get_fixed_nodes(elemType, "clamped")
    Ks, Ms, F = FEMMethods.apply_BCs(globalKs, globalMs, globalF, globalDOFBlankingList)

    # --- Initialize stuff ---
    u = copy(globalF)
    # globalMf = copy(globalMs) * 0
    # globalCf_r = copy(globalKs) * 0
    # globalKf_r = copy(globalKs) * 0
    # globalKf_i = copy(globalKs) * 0
    # globalCf_i = copy(globalKs) * 0
    # extForceVec = copy(F) * 0 # this is a vector excluded the BC nodes
    # extForceVec[end-1] = tipForceMag # this is applying a tip twist
    # LiftDyn = zeros(length(fSweep)) # * 0im
    # MomDyn = zeros(length(fSweep)) # * 0im
    # TipBendDyn = zeros(length(fSweep)) # * 0im
    # TipTwistDyn = zeros(length(fSweep)) # * 0im

    # --- Test eigensolve ---
    omegaSquared, structModeShapes = SolverRoutines.compute_eigsolve(Ks, Ms, 3)
    structNatFreqs = sqrt.(omegaSquared) / (2π)
    println(structNatFreqs)

    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(Ks, Ms, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ 1:length(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q

    derivMode = "FAD"
    global CONSTANTS = SolutionConstants.DCFoilConstants(Ks, Ms, elemType, structMesh, zeros(2, 2), derivMode, 0.0)

    b_ref = Statistics.mean(FOIL.c) # mean semichord
    dim = size(Ks)[1] + length(globalDOFBlankingList)

    # --- Apply the flutter solution method ---
    compute_pkFlutterAnalysis(uSweep, structMesh, FOIL, b_ref, dim, elemType, globalDOFBlankingList)

end

function compute_correlationMatrix(old_r, old_i, new_r, new_i)
    """
    DESCRIPTION

    This routine computes the eigenvector correlation matrix needed for associating converged eigenvectors
    This implementation is based on the van Zyl mode tracking method 1992. https://arc.aiaa.org/doi/abs/10.2514/3.46380

    ARGUMENTS
        old_r, old_i - the real/imaginary part of the old eigenvectors from previous iteration. The size is (Mold,Mold)
        new_r, new_i - the real/imaginary part of the new eigenvector from current iteration. The size is (Mnew,Mnew)
    Outputs
    -------
        C - The correlation matrix. Values range from 0-1 where line represent old eigenvectors and columns new eigenvectors. The size is (M_old,M_new)
    """

    M_old = size(old_r)[1]
    N_old = size(old_r)[2]
    M_new = size(new_r)[1]
    N_new = size(new_r)[2]
    normOld = zeros(Float64, N_old)
    normNew = zeros(Float64, N_new)
    C = zeros(Float64, M_new, N_new)

    # Norm of each eigenvector for old array
    normOld = sqrt.(sum(old_r .^ 2 + old_i .^ 2, dims=1))

    # Norm of each eigenvector for new array
    normNew = sqrt.(sum(new_r .^ 2 + new_i .^ 2, dims=1))

    # Dot product the arrays
    old = old_r + 1im * old_i
    new = new_r + 1im * new_i
    Ctmp = abs.(transpose(conj(old)) * new)

    # Now normalize correlation matrix
    for j in 1:N_new # loop cols
        for i in 1:M_new # loop rows
            if normOld[i] == 0.0 || normNew[j] == 0.0
                C[i, j] = 0.0
            else
                C[i, j] = Ctmp[i, j] / (normOld[i] * normNew[j])
            end
        end
    end

    return C

end

function compute_pkFlutterAnalysis(vel, structMesh, FOIL, b_ref, dim, elemType, globalDOFBlankingList, nModes=1)
    """
    Non-iterative flutter solution following van Zyl https://arc.aiaa.org/doi/abs/10.2514/2.2806
    Everything from here on is based on the FORTRAN code written by Eirikur Jonsson

    Inputs
    ------
    vel: array, size(# of flight conditions)
        free-stream velocities for eigenvalue solve (flight conditions)
    structMesh: StructMesh
        mesh object
    FOIL: FOIL  
        foil object
    dim: int
        dimension of hydro matrices
        """

    # --- Initialize stuff ---
    globalMf::Matrix{Float64} = zeros(Float64, dim, dim)
    globalCf_r::Matrix{Float64} = zeros(Float64, dim, dim)
    globalKf_r::Matrix{Float64} = zeros(Float64, dim, dim)
    globalKf_i::Matrix{Float64} = zeros(Float64, dim, dim)
    globalCf_i::Matrix{Float64} = zeros(Float64, dim, dim)
    initialSigns = zeros(Int64, nModes)
    ξVec = zeros(Float64, nModes)
    kVec = zeros(Float64, nModes)
    dynP = 0.5 * FOIL.ρ_f * vel .^ 2 # vector of dynamic pressures
    ωSweep = 2π * FOIL.fSweep



    # ---------------------------
    #   Loop over velocity range
    # ---------------------------
    nFlow = 1
    for U∞ in vel

        println("Running nFlow = ", nFlow, " dynP ", round(U∞^2 * 0.5 * FOIL.ρ_f, digits=3), "[Pa]", " rho_f ", FOIL.ρ_f, " vel ", U∞, "[m/s])")

        # --- Sweep k and find crossings ---
        kSweep = ωSweep * b_ref / ((U∞ * cos(FOIL.Λ)))
        p_cross_r, p_cross_i, R_cross_r, R_cross_i = compute_kCrossings(dim, kSweep, b_ref, FOIL, U∞, CONSTANTS.Mmat, CONSTANTS.Kmat, structMesh, globalDOFBlankingList)

        if nFlow == 1
            # Sort eigenvalues based on the frequency (imaginary part)
            idxTmp = sortperm(p_cross_i)

        else
            # Mode tracking

        end

        ξVec[ii] = p_cross_r
        kVec[ii] = p_cross_i

        nFlow += 1
    end

    return ξVec, kVec

end

function compute_kCrossings(dim, kSweep, b, FOIL, U∞, MM, KK, structMesh, globalDOFBlankingList)
    """
    # This routine solves an eigenvalue problem over a range of reduced frequencies k searches for the
    # intersection of each mode with the diagonal line Im(p) = k and then does a linear interpolation
    # for the eigenvalue and eigenvector. This is method of van Zyl https://arc.aiaa.org/doi/abs/10.2514/2.2806
    MM - structural mass matrix (dim, dim)
    KK - structural stiffness matrix (dim, dim)
    Outputs
    -------
        p_cross_r - unsorted
        p_cross_i - unsorted
        R_cross_r - unsorted
        R_cross_i - unsorted
    """

    N_MAX_K_ITER = 500 # max iterations
    k_ctr = 0 # reduced freq (k) counter

    # --- Loop over reduced frequency search range ---
    p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history = sweep_kCrossings(dim, kSweep, b, U∞, MM, KK, structMesh, FOIL, globalDOFBlankingList, N_MAX_K_ITER)

    # --- Extract valid solutions through interpolation ---
    p_cross_r, p_cross_i, R_cross_r, R_cross_i = extract_kCrossings(dim, p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history)


    return p_cross_r, p_cross_i, R_cross_r, R_cross_i
end

function sweep_kCrossings(dim, kSweep, b, U∞, MM, KK, structMesh, FOIL, globalDOFBlankingList, N_MAX_K_ITER)
    """
    Solve the eigenvalue problem over a range of reduced frequencies (k)
    Inputs
    ------
        dim - size of problem (nDOF w/o BC)
        kSweep - sweep of reduced frequencies
        b - semichord
        MM
        KK
    Outputs
    -------

    """

    # Fluid matrices
    globalMf::Matrix{Float64} = zeros(Float64, dim, dim)
    globalCf_r::Matrix{Float64} = zeros(Float64, dim, dim)
    globalKf_r::Matrix{Float64} = zeros(Float64, dim, dim)
    globalKf_i::Matrix{Float64} = zeros(Float64, dim, dim)
    globalCf_i::Matrix{Float64} = zeros(Float64, dim, dim)

    dimwithBC = dim - length(globalDOFBlankingList)
    # Eigenvalue and vector matrices
    p_eigs_r::Matrix{Float64} = zeros(Float64, 2 * dimwithBC, N_MAX_K_ITER)
    p_eigs_i::Matrix{Float64} = zeros(Float64, 2 * dimwithBC, N_MAX_K_ITER)
    R_eigs_r::Array{Float64} = zeros(Float64, 2 * dimwithBC, 2 * dimwithBC, N_MAX_K_ITER)
    R_eigs_i::Array{Float64} = zeros(Float64, 2 * dimwithBC, 2 * dimwithBC, N_MAX_K_ITER)
    k_history = zeros(Float64, N_MAX_K_ITER)

    p_diff_max = 0.2 # maximum allowed change in poles btwn two steps

    # ************************************************
    #     Perform k sweep
    # ************************************************
    failed = false
    ik = 1 # k counter
    for k in kSweep # TODO: CHANGE THIS TO A WHILE LOOP THAT ALLOWS NEW K CHOICES

        ω = k * U∞ / b
        # ---------------------------
        #   Set the hydrodynamics
        # ---------------------------
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = Hydro.compute_AICs!(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, FOIL, U∞, ω, CONSTANTS.elemType)
        Kf_r, Cf_r, Mf = Hydro.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
        Kf_i, Cf_i, _ = Hydro.apply_BCs(globalKf_i, globalCf_i, globalMf, globalDOFBlankingList)

        # # --- Test wet eigensolve ---
        # omegaSquared, _ = SolverRoutines.compute_eigsolve(KK, MM .+ Mf, 3)
        # wetNatFreqs = sqrt.(omegaSquared) / (2π)
        # println(wetNatFreqs)
        # # TODO: The wet natural frequencies are off so there might be a bug with the added mass since air is good

        # --- Solve eigenvalue problem ---
        pkEqnType = "Hassig"
        p_r_tmp, p_i_tmp, R_aa_r_tmp, R_aa_i_tmp = solve_eigenvalueProblem(pkEqnType, dimwithBC, b, U∞, FOIL, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK)

        # --- Sort eigenvalues from small to large ---
        p_r = sort(p_r_tmp)
        idxs = sortperm(p_r_tmp)
        p_i = p_i_tmp[idxs]
        R_aa_r = R_aa_r_tmp[:, idxs]
        R_aa_i = R_aa_i_tmp[:, idxs]

        if (ik > 1)
            # van Zyl tracking method: Find correlation matrix btwn current and previous eigenvectors (mode shape)
            # Rows are old eigenvector number and columns are new eigenvector number
            corr = compute_correlationMatrix(R_eigs_r[:, :, ik-1], R_eigs_i[:, :, ik-1], R_aa_r, R_aa_i)

            # Determine location of new eigenvectors
            idxs = SolverRoutines.argmax2d(transpose(corr))

            # Check if entries are missing/duplicated 
            # TODO: this might not be a problem

            # --- Order eigenvalues and eigenvectors ---
            # p_r_tmp = p_r
            p_r = p_r_tmp[idxs]
            # p_i_tmp = p_i
            p_i = p_i_tmp[idxs]

            R_aa_r = R_aa_r_tmp[:, idxs]
            R_aa_i = R_aa_i_tmp[:, idxs]

            # --- If too big of jump, back up ---
            tmp_p_diff = sqrt.((p_eigs_r[:, ik-1] - p_r) .^ 2 + (p_eigs_i[:, ik-1] - p_i) .^ 2)
            p_diff = maximum(tmp_p_diff)
            if (p_diff > p_diff_max)
                println("p_diff", p_diff)
                failed = true
            end
        end

        # ---------------------------
        #   Check solution
        # ---------------------------
        if failed
            println("failed, need to adjust k")
        else
            # Success so store solution
            p_eigs_r[:, ik] = p_r
            p_eigs_i[:, ik] = p_i
            R_eigs_r[:, :, ik] = R_aa_r
            R_eigs_i[:, :, ik] = R_aa_i
            k_history[ik] = k

            ik += 1 # increment freq search counter

            if (maxImP < k_history[ik-1]) || (k > maxK)
                
                keep_looping = false
            else

        end

        ik = ik -1 # Reduce counter b/c it was incremented in the last iteration
    end

    return p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history
end

function solve_eigenvalueProblem(pkEqnType, dim, b, U∞, FOIL, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK)
    # """
    # This routine solves the following eigenvalue problem.
    #   [ (U/b)^2 * p^2 * M + (U/b) * C + K - F_aero ] * u = 0

    # Form a first order system by introducing I*\dot{u} = I*\dot{u} where I is identity
    # Then the problem is a generalized eigenvalue problem, p*A*u = B*u where p,u are eigen- values/vectors
    # Then we cast ot into standard form p * u = A^{-1} * B * u which then is solved
    # Note: The A and B in the are just general matrices and serve as placeholders, A is NOT the aero loads
    #       as written in the equaions below.

    # Hassig
    #     The Aerodynamic loads are written
    #     F_aero = qinf * A
    #         p * |I       0     | * | u |  =  |    0        I   | * | u |
    #             |0  (U/b)^2 * M|   |p*u|     |-(K-q*A)  -U/b*C |   |p*u|

    # Rodden
    #     The Aerodynamic loads are written in terms of real (R) and imaginary part (I)
    #     F_aero = qinf * (A^R + i * A^I)
    #         p * |I       0     | * | u |  =  |    0                   I            | * | u |
    #             |0  (U/b)^2 * M|   |p*u|     |-(K-q*A^R)  -(U/b*C - qinf/k * A^I)  |   |p*u|
    # ARGUMENTS
    #       dim - the size of reduced problem
    #       b - the half chord. Scalar
    #       vel - the velocity of the fluid. Scalar
    #       Mf, Cf_r, Cf_i, Kf_r, Kf_i - The AIC (real/imag parts). Array(dim,dim)
    #       MM - structural mass matrix. Array(dim,dim)
    #       CC - structural damping matrix. Array(dim,dim)
    #       KK - structural stiffness matrix. Array(dim,dim)
    # Outputs
    # -------
    #     p_r - real part of flutter eigenvalue (2*dim)
    #     p_i - imag part of flutter eigenvalue (2*dim)
    #     R_aa_r - real part of flutter mode shape (2*dim, 2*dim)
    #     R_aa_i - imag part of flutter mode shape (2*dim, 2*dim)
    # """

    # --- Initialize ---
    iMat = Matrix{Float64}(I, dim, dim)
    zeroMat = zeros(dim, dim)
    A_r = zeros(2 * dim, 2 * dim)
    A_i = zeros(2 * dim, 2 * dim) # this will always be zero
    B_r = zeros(2 * dim, 2 * dim)
    B_i = zeros(2 * dim, 2 * dim)

    # Form the matrices that form the generalized eigenvalue problem
    # All entries in A and B are real except the ones coming from the AIC
    if pkEqnType == "Hassig"

        # B - Real part
        firstRow = hcat(zeroMat, iMat)
        secondRow = hcat(-1 * (Kf_r + KK), -1 * (U∞ * cos(FOIL.Λ) / b) .* (Cf_r))

        B_r = vcat(firstRow, secondRow)

        # B - Imag part
        secondRow = hcat(-1 * (Kf_i), -1 * (U∞ * cos(FOIL.Λ) / b) .* (Cf_i))

        # A - real part
        firstRow = hcat(iMat, zeroMat)
        secondRow = hcat(zeroMat, (U∞ / b)^2 .* (Mf + MM))

        A_r = vcat(firstRow, secondRow)

    elseif pkEqnType == "rodden"

        println("Not implemented yet")

    end


    # --- Invert the complex matrix and solve ---
    Ainv_r, Ainv_i = SolverRoutines.cmplxInverse(A_r, A_i, 2 * dim)
    FlutterMat_r, FlutterMat_i = SolverRoutines.cmplxMatmult(Ainv_r, Ainv_i, B_r, B_i)

    # --- Compute the eigenvalues ---
    p_r, p_i, _, _, R_aa_r, R_aa_i = SolverRoutines.cmplxStdEigValProb(FlutterMat_r, FlutterMat_i, 2 * dim)

    return p_r, p_i, R_aa_r, R_aa_i
end

end # end module