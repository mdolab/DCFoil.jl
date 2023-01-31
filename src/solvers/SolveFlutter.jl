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
# using Profile # for profiling
using Plots # for debugging
using Printf
using JLD # julia data format

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

    # ---------------------------
    #   Initialize stuff
    # ---------------------------
    u = copy(globalF)
    globalMf = copy(globalMs) * 0
    globalCf_r = copy(globalKs) * 0
    globalKf_r = copy(globalKs) * 0
    globalKf_i = copy(globalKs) * 0
    globalCf_i = copy(globalKs) * 0
    # extForceVec = copy(F) * 0 # this is a vector excluded the BC nodes
    # extForceVec[end-1] = tipForceMag # this is applying a tip twist
    # LiftDyn = zeros(length(fSweep)) # * 0im
    # MomDyn = zeros(length(fSweep)) # * 0im
    # TipBendDyn = zeros(length(fSweep)) # * 0im
    # TipTwistDyn = zeros(length(fSweep)) # * 0im
    derivMode = "FAD"
    global CONSTANTS = SolutionConstants.DCFoilConstants(Ks, Ms, elemType, structMesh, zeros(2, 2), derivMode, 0.0)

    # ---------------------------
    #   Test eigensolver
    # ---------------------------
    # --- Dry solve ---
    omegaSquared, structModeShapes = SolverRoutines.compute_eigsolve(Ks, Ms, 3)
    structNatFreqs = sqrt.(omegaSquared) / (2π)
    println("+------------------------------------+")
    println("Structural natural frequencies [Hz]:")
    println("+------------------------------------+")
    ctr = 1
    for natFreq in structNatFreqs
        println(@sprintf("mode %i: %.3f", ctr, natFreq))
        ctr += 1
    end
    println("+------------------------------------+")
    # --- Wetted solve ---
    # Provide dummy inputs for the hydrodynamic matrices; we really just need the mass!
    globalMf, globalCf_r, _, globalKf_r, _ = Hydro.compute_AICs!(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, FOIL, 0.1, 0.1, CONSTANTS.elemType)
    _, _, Mf = Hydro.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
    omegaSquared, _ = SolverRoutines.compute_eigsolve(Ks, Ms .+ Mf, 3)
    wetNatFreqs = sqrt.(omegaSquared) / (2π)
    println("Wetted natural frequencies [Hz]:")
    println("+------------------------------------+")
    ctr = 1
    for natFreq in wetNatFreqs
        println(@sprintf("mode %i: %.3f", ctr, natFreq))
        ctr += 1
    end
    println("+------------------------------------+")
    return # my manual breakpoint

    # ---------------------------
    #   Pre-solve system
    # ---------------------------
    q = FEMMethods.solve_structure(Ks, Ms, F)

    # --- Populate displacement vector ---
    u[globalDOFBlankingList] .= 0.0
    idxNotBlanked = [x for x ∈ 1:length(u) if x ∉ globalDOFBlankingList] # list comprehension
    u[idxNotBlanked] .= q


    b_ref = Statistics.mean(FOIL.c) # mean semichord
    dim = size(Ks)[1] + length(globalDOFBlankingList)

    # --- Apply the flutter solution method ---
    N_MAX_Q_ITER = 1000 # TEST VALUE
    true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory = compute_pkFlutterAnalysis(uSweep, structMesh, FOIL, b_ref, dim, elemType, globalDOFBlankingList, N_MAX_Q_ITER)


    # ************************************************
    #     Write solution out to files
    # ************************************************
    write_sol(true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory, outputDir)

end # end function

function write_sol(true_eigs_r, true_eigs_i, R_eigs_r, R_eigs_i, iblank, flowHistory, outputDir="./OUTPUT/")
    """
    Write out the p-k flutter results
    """

    # Store solutions here
    workingOutput = outputDir * "pkFlutter/"
    mkpath(workingOutput)

    # --- Store eigenvalues ---
    fname = workingOutput * "eigs_r.jld"
    save(fname, "data", true_eigs_r)
    fname = workingOutput * "eigs_i.jld"
    save(fname, "data", true_eigs_i)

    # --- Store eigenvectors ---
    fname = workingOutput * "eigenvectors_r.jld"
    save(fname, "data", R_eigs_r)
    fname = workingOutput * "eigenvectors_i.jld"
    save(fname, "data", R_eigs_i)

    # --- Store blanking ---
    fname = workingOutput * "iblank.jld"
    save(fname, "data", iblank)

    # --- Store flow history ---
    fname = workingOutput * "flowHistory.jld"
    save(fname, "data", flowHistory)
end # end function

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
        C - The correlation matrix. Values range from 0-1 where rows represent old eigenvectors and columns new eigenvectors. The size is (M_old,M_new)
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

    # Now normalize correlation matrix Ctmp
    for jj in 1:N_new # loop cols
        for ii in 1:N_old # loop rows
            if normOld[ii] == 0.0 || normNew[jj] == 0.0
                C[ii, jj] = 0.0
            else
                C[ii, jj] = Ctmp[ii, jj] / (normOld[ii] * normNew[jj])
            end
        end
    end

    return C

end # end function

function compute_correlationMetrics(old_r, old_i, new_r, new_i, p_old_i, p_new_i)
    """
    This routine computes the correlation metrics based on previous and current eigenvectors
    between dynamic pressure increments qᵢ and qᵢ₊₁
    Inputs
    ------
        old_r, old_i - the real/imaginary part of the old eigenvectors from previous iteration. The size is (Mold,Nold)
        new_r, new_i - the real/imaginary part of the new eigenvector from current iteration. The size is (Mnew,Nnew)
        p_old_i - imaginary part of eigenvalues from previous iteration. Size is (Nold)
        p_new_i - imaginary part of eigenvalues from previous iteration. Size is (Nnew)

    Outputs
    -------
        corr - correlation metric used to determine how well modes correlate, should be between 0-1 where 1 is perfect correlation
        m - array where for each line the first column has the index location of the old eigenvector and the second column has the new eigenvector location
        newModesIdx - holds the indices of any new modes

    Note that Mold == Mnew and Nold and Nnew should be at maximum Mold or MNew
    """

    # Rows and columns of old and new eigenvectors
    M_old::Int64 = size(old_r)[1]
    N_old::Int64 = size(old_r)[2]
    M_new::Int64 = size(new_r)[1]
    N_new::Int64 = size(new_r)[2]

    # --- Initialize ---
    corr = zeros(Float64, N_old)
    m = zeros(Int64, N_old, 2)
    newModesIdx = zeros(Int64, N_old)

    nCorrelatedModes = 0
    nNewModes = 0

    # Working matrices
    corrTmp = zeros(Float64, N_new)
    mTmp = zeros(Int64, N_new, 2)
    newModesIdxTmp = zeros(Int64, N_new)

    # --- Compute correlation matrix ---
    C = compute_correlationMatrix(old_r, old_i, new_r, new_i)

    isMaxValZero = true
    while isMaxValZero

        # --- Find max value in correlation matrix and location ---
        maxI, maxJ, maxVal = SolverRoutines.maxLocArr2d(C)

        # --- Check if max value is zero ---
        if maxVal == 0.0
            isMaxValZero = false
        else
            # Store the correlation value in its proper location
            corrTmp[maxJ] = maxVal

            # Store where previous eigenvector was in proper spot for current eigenvector
            mTmp[maxJ, 1] = maxI

            # Now zero out corresponding row and column in correlation matrix
            C[maxI, :] .= 0.0
            C[:, maxJ] .= 0.0
        end
    end # end while

    # --- Add indices for location of new eigenvector ---
    for ii in 1:N_new
        mTmp[ii, 2] = ii
    end

    # --- Find zero and nonzero elements in correlation array ---
    # Then find how many each has
    nz::Int64 = 0
    for ii in 1:N_new
        if corrTmp[ii] == 0.0 # new mode?
            # If correlation is zero, then it could be a new mode
            nz += 1
            newModesIdxTmp[nz] = mTmp[ii, 2]
        else # store index and correlation
            nCorrelatedModes += 1
            m[nCorrelatedModes, :] = mTmp[ii, :]
            corr[nCorrelatedModes] = corrTmp[ii]
        end
    end

    # --- Lower frequency modes missed? ---
    # Loop over all possible new modes and see if they are lower than some max
    nNewModes = 0
    maxVal = maximum(p_old_i)
    for ii in 1:nz
        if (p_new_i[newModesIdxTmp[ii]] < maxVal / 2)
            nNewModes += 1
            newModesIdx[nNewModes] = newModesIdxTmp[ii]
        end
    end

    return corr, m, newModesIdx, nCorrelatedModes, nNewModes
end # end function

function compute_pkFlutterAnalysis(vel, structMesh, FOIL, b_ref, dim, elemType, globalDOFBlankingList, N_MAX_Q_ITER, nModes=10, ΔdynP=10.0, debug=false)
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
    elemType: string
        element type
    globalDOFBlankingList: array, size(# of DOFs blanked)
        list of DOFs to be blanked
    N_MAX_Q_ITER: int
        maximum number of dynamic pressure iterations 
    nModes: int
        number of modes to solve for
    ΔdynP: float
        dynamic pressure increment [Pa]
    debug: bool
        flag to print debug statements
    Outputs
    -------
    true_eigs_r, true_eigs_i: array, size(3*nModes, N_MAX_Q_ITER)
        real and imaginary parts of eigenvalues of flutter modes
    R_eigs_r_tmp, R_eigs_i_tmp: array, size(2*dimwithBC, 3*nModes, N_MAX_Q_ITER)
        real and imaginary parts of eigenvectors of flutter modes
    iblank: array, size(3*nModes, N_MAX_Q_ITER)
        array of indices of blanked modes (which indicate failed solution)
    flowHistory: array, size(N_MAX_Q_ITER, 3)
        history of flow conditions [velocity, density, dynamic pressure]
    """

    dimwithBC = dim - length(globalDOFBlankingList) # get dimension of matrix with BC applied
    # ---------------------------
    #   Initializations
    # ---------------------------
    # --- Correlation arrays ---
    m::Matrix{Int64} = zeros(Int64, nModes * 3, 2)
    # The m matrix stores which mode is correlated with what.
    # The first column stores indices of old m[:,1] and
    # the second column stores indices of the newly found modes m[:,2]
    # --- Outputs ---
    # ξVec::Matrix{Float64} = zeros(Float64, length(vel), nModes)
    # kVec::Matrix{Float64} = zeros(Float64, length(vel), nModes)
    p_r::Matrix{Float64} = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    p_i::Matrix{Float64} = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    true_eigs_r::Matrix{Float64} = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    true_eigs_i::Matrix{Float64} = zeros(Float64, 3 * nModes, N_MAX_Q_ITER)
    R_eigs_r_tmp = zeros(Float64, 2 * dimwithBC, 3 * nModes, N_MAX_Q_ITER)
    R_eigs_i_tmp = zeros(Float64, 2 * dimwithBC, 3 * nModes, N_MAX_Q_ITER)
    iblank::Matrix{Int64} = zeros(Int64, 3 * nModes, N_MAX_Q_ITER) # stores which modes are blanked and therefore have a failed solution


    # --- Working vars ---
    flowHistory = zeros(Float64, N_MAX_Q_ITER, 3) # stores [velocity, density, dynamic pressure]
    tmp = zeros(Float64, 3 * dim)
    dynP = 0.5 * FOIL.ρ_f * vel .^ 2 # vector of dynamic pressures
    ωSweep = 2π * FOIL.fSweep
    p_diff_max = 0.1 # max allowed change in roots between steps


    # ************************************************
    #     Loop over velocity range
    # ************************************************
    # ---------------------------
    #   Initialize loop vars
    # ---------------------------
    global nFlow = 1 # first flow iter
    # Set working fluid values for the loop
    dynPTmp = dynP[1] # set temporary dynamic pressure used in loop to first dynamic pressure
    U∞ = vel[1] # first velocity
    dynPMax = dynP[end] # set max dynamic pressure to last value
    # ---------------------------
    #   Begin loop
    # ---------------------------
    while (nFlow <= N_MAX_Q_ITER)

        # Flow condition printout
        println("Running nFlow = ", nFlow, " dynP ", round(U∞^2 * 0.5 * FOIL.ρ_f, digits=3), "[Pa]", " rho_f ", FOIL.ρ_f, "[kg/m^3] vel ", U∞, " [m/s])")

        # Set the proper fail flag
        failed = false

        # Sweep k and find crossings
        kSweep = ωSweep * b_ref / ((U∞ * cos(FOIL.Λ)))
        p_cross_r, p_cross_i, R_cross_r, R_cross_i, kCtr = compute_kCrossings(dim, kSweep, b_ref, FOIL, U∞, CONSTANTS.Mmat, CONSTANTS.Kmat, structMesh, globalDOFBlankingList)

        # --- Check flight condition ---
        if (nFlow == 1) # first flight condition
            # Sort eigenvalues based on the frequency (imaginary part)
            idxTmp = sortperm(p_cross_i[1:kCtr])

            # Set the number of modes to correlate equal to the number of modes we're solving for
            global nCorr = nModes
            global NTotalModesFound = nModes

            for ii in 1:nModes
                m[ii, 1] = ii
                m[ii, 2] = idxTmp[ii]
            end


        else # not first condition; apply mode tracking between flow speeds

            # Compute correlation matrix
            corr, m, newModesIdx, nCorr, nCorrNewModes = compute_correlationMetrics(R_eigs_r_tmp[:, :, nFlow-1], R_eigs_i_tmp[:, :, nFlow-1], R_cross_r[:, 1:kCtr], R_cross_i[:, 1:kCtr], p_i[:, nFlow-1], p_cross_i[1:kCtr])

            # --- Check if eigenvalue jump is too big ---
            # If the jump is too big, we back up
            # We do this by scaling the 'p' to the true eigenvalue
            eigScale = sqrt(dynPTmp / flowHistory[nFlow-1, 3]) # This is a velocity scale


            inner = (p_r[m[1:nCorr, 1], nFlow-1] - p_cross_r[m[1:nCorr, 2]] .* eigScale) .^ 2 +
                    (p_i[m[1:nCorr, 1], nFlow-1] - p_cross_i[m[1:nCorr, 2]] .* eigScale) .^ 2
            tmp[1:nCorr] = sqrt.(inner)

            maxVal = maximum(tmp[1:nCorr])

            if (maxVal > p_diff_max)
                # println("Eigenvalue jump too big, backing up. p_diff: ", maxVal, " > ", p_diff_max)
                failed = true
            end

            # Were there too many iterations w/o progress? Mode probably disappeared
            if (dynPTmp - flowHistory[nFlow-1, 3] < ΔdynP / 50)
                # Let's check the fail flag, if yes then accept
                if (failed)
                    # We should keep all modes that have high correlations, drop others
                    nKeep::Int64 = 0
                    for ii in 1:nCorr
                        if corr[ii] > 0.5
                            nKeep += 1
                            m[nKeep, :] = m[ii, :]
                        end
                    end

                    # We only want the first nKeep modes so we need to overwrite nCorr, which is used later in the code for indexing
                    nCorr = nKeep

                    failed = false
                end
            end

            # Now we add in any new lower frequency modes that weren't correlated above
            if (nCorrNewModes > 0) && !failed

                # Append new modes' indices to the new 'm' array
                for ii in 1:nCorrNewModes
                    m[nCorr+ii, 1] = NTotalModesFound + ii
                    m[nCorr+ii, 2] = newModesIdx[ii]
                end

                # Increment the nCorr variable counter to include the recently added modes
                nCorr += nCorrNewModes

                # Increment the shift variable for the next new mode
                # NOTE: this shift variable has the total number of modes found over the ENTIRE simulation, not the current number of modes found
                NTotalModesFound += nCorrNewModes

            end
        end


        # --- Store solution if good ---
        if failed # backup dynamic pressure
            dynPTmp = (dynPTmp - flowHistory[nFlow-1, 3]) * 0.5 + flowHistory[nFlow-1, 3]
        else # store solution
            # Eigenvalues
            if m[nCorr, 1] > 3 * nModes # is it column 1 or 2 (it's column 1 so there's a bug)
                println("NTotalModesFound ", NTotalModesFound)
                println("nCorrNewModes", nCorrNewModes)
            end
            p_r[m[1:nCorr, 1], nFlow] = p_cross_r[m[1:nCorr, 2]]
            p_i[m[1:nCorr, 1], nFlow] = p_cross_i[m[1:nCorr, 2]]
            # Eigenvectors
            R_eigs_r_tmp[:, m[1:nCorr, 1], nFlow] = R_cross_r[:, m[1:nCorr, 2]]
            R_eigs_i_tmp[:, m[1:nCorr, 1], nFlow] = R_cross_i[:, m[1:nCorr, 2]]

            # Non-dimensionalization factor
            tmpFactor = U∞ * cos(FOIL.Λ) / b_ref
            # Dimensional eigenvalues [rad/s]
            true_eigs_r[m[1:nCorr, 1], nFlow] = p_cross_r[m[1:nCorr, 2]] * tmpFactor
            true_eigs_i[m[1:nCorr, 1], nFlow] = p_cross_i[m[1:nCorr, 2]] * tmpFactor

            flowHistory[nFlow, 1] = U∞
            flowHistory[nFlow, 2] = FOIL.ρ_f
            flowHistory[nFlow, 3] = dynPTmp

            iblank[m[1:nCorr, 1], nFlow] .= 1

            # # --- Correlated eigenvalues found ---
            # filename = "dynP" * string(nFlow, pad=2) * ".txt"
            # exampleFileIOStream = open(filename, "w")
            # for idx in m[1:nCorr, 1]
            #     write(exampleFileIOStream, string(p_i[idx, nFlow]))
            #     write(exampleFileIOStream, "\n")
            # end
            # close(exampleFileIOStream)

            # # --- Raw eigenvalues found ---
            # filename = "raw_dynP" * string(nFlow, pad=2) * ".txt"
            # exampleFileIOStream = open(filename, "w")
            # for idx in 1:nModes
            #     write(exampleFileIOStream, string(p_cross_i[idx]))
            #     write(exampleFileIOStream, "\n")
            # end
            # write(exampleFileIOStream, "Number of modes found: " * string(nCorr) * "\n")
            # write(exampleFileIOStream, "First and second column of correlation\n")
            # for idx in m[1:nCorr, 1]
            #     write(exampleFileIOStream, string(m[idx, 1]))
            #     write(exampleFileIOStream, "\t")
            #     write(exampleFileIOStream, string(m[idx, 2]))
            #     write(exampleFileIOStream, "\n")
            # end
            # close(exampleFileIOStream)


            # --- Increment for next iteration ---
            nFlow += 1
            dynPTmp += ΔdynP
            # Determine flow speed
            U∞ = sqrt(2 * dynPTmp / FOIL.ρ_f)

            # --- Check if we're done ---
            if (dynPTmp > dynPMax)
                # We should stop at (or near) the max velocity specified so we should check if we're within some tolerance

                # First subtract previously added increment, then subtract max velocity
                if (abs((dynPTmp - ΔdynP) - dynPMax) < SolutionConstants.mepsLarge)
                    # Exit the loop
                    break
                else # Try max value
                    # If this fails, the step will be halved anyway and this process should repeat until the exit condition is met
                    dynPTmp = dynPMax
                end

            end

        end


    end # end while

    # Decrement flow index since it was incremented before exit
    nFlow -= 1

    return true_eigs_r, true_eigs_i, R_eigs_r_tmp, R_eigs_i_tmp, iblank, flowHistory

end # end function

function compute_kCrossings(dim, kSweep, b, FOIL, U∞, MM, KK, structMesh, globalDOFBlankingList)
    """
    # This routine solves an eigenvalue problem over a range of reduced frequencies k searches for the
    # intersection of each mode with the diagonal line Im(p) = k and then does a linear interpolation
    # for the eigenvalue and eigenvector. This is method of van Zyl https://arc.aiaa.org/doi/abs/10.2514/2.2806
    MM - structural mass matrix (dim, dim)
    KK - structural stiffness matrix (dim, dim)
    Outputs
    -------
        p_cross_r - unsorted eigenvalues [rad/s]
        p_cross_i - unsorted eigenvalues [rad/s]
        R_cross_r - unsorted eigenvectors [-]
        R_cross_i - unsorted eigenvectors [-]
        ctr - reduced frequency counter
    """

    N_MAX_K_ITER = 500 # max iterations

    # --- Loop over reduced frequency search range to construct lines ---
    p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik = sweep_kCrossings(dim, kSweep, b, U∞, MM, KK, structMesh, FOIL, globalDOFBlankingList, N_MAX_K_ITER)

    # # DEBUG CODE FOR VISUALIZING THE OUTPUT LINES WHERE MODES CROSS Im(p) = k 
    # plot(k_history[1:ik], p_eigs_i[1, 1:ik], label="mode 1")
    # plot!(k_history[1:ik], p_eigs_i[2, 1:ik], label="mode 2")
    # plot!(k_history[1:ik], p_eigs_i[3, 1:ik], label="mode 3")
    # plot!(k_history[1:ik], p_eigs_i[4, 1:ik], label="mode 4")
    # plot!(k_history[1:ik], p_eigs_i[5, 1:ik], label="mode 5")
    # plot!(k_history[1:ik], p_eigs_i[6, 1:ik], label="mode 6")
    # plot!(k_history[1:ik], p_eigs_i[7, 1:ik], label="mode 7")
    # plot!(k_history[1:ik], p_eigs_i[8, 1:ik], label="mode 8")
    # plot!(k_history[1:ik], k_history[1:ik], lc=:black ,label="Im(p)=k")
    # xlabel!("k")
    # ylabel!("Im(p)")
    # savefig("debug.pdf")

    # --- Extract valid solutions through interpolation ---
    dimwithBC = dim - length(globalDOFBlankingList)
    p_cross_r, p_cross_i, R_cross_r, R_cross_i, ctr = extract_kCrossings(dimwithBC, p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik, N_MAX_K_ITER)


    return p_cross_r, p_cross_i, R_cross_r, R_cross_i, ctr
end # end function

function sweep_kCrossings(dim, kSweep, b, U∞, MM, KK, structMesh, FOIL, globalDOFBlankingList, N_MAX_K_ITER)
    """
    Solve the eigenvalue problem over a range of reduced frequencies (k)

    Inputs
    ------
        dim - size of problem (nDOF w/o BC) (half the number of flutter modes you're solving for)
        kSweep - sweep of reduced frequencies
        b - semichord
        MM - structural mass matrix
        KK - structural stiffness matrix
    Outputs
    -------
        p_eigs_r - unsorted eigenvalues of length ik [rad/s]
        p_eigs_i - unsorted eigenvalues of length ik [rad/s]
        R_eigs_r - unsorted eigenvectors of length ik [-] 
        R_eigs_i - unsorted eigenvectors of length ik [-] 
        k_history - history of reduced frequencies. Actual k values that we analyzed and accepted
        ik - number of k's that were actually analyzed and accepted
        The result is essentially 'nMode' sets of lines that the eigenvalue problem was solved for
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

    # --- Determine Δk to step ---
    # based on maximum in-vacuum natural frequency
    # Δk = maximum(sqrt.(inv(MM) * KK)) * b / (U∞ * 100.0)
    omegaSquared, _ = SolverRoutines.compute_eigsolve(KK, MM, size(KK)[1])
    Δk = maximum(sqrt.(omegaSquared) * b / (U∞ * 100.0))

    # ************************************************
    #     Perform iterations on k values
    # ************************************************
    keepLooping = true
    k = 1e-15 # first guess of machine zero
    maxK = kSweep[end]
    ik = 1 # k counter
    pkEqnType = "Hassig"
    while keepLooping
        failed = false # fail flag on k jump must be reset to false on every k iteration

        # ---------------------------
        #   Compute hydrodynamics
        # ---------------------------
        # In Eirikur's code, he interpolates the AIC matrix but since it is cheap, we just compute it exactly
        ω = k * U∞ / b
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = Hydro.compute_AICs!(globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i, structMesh, FOIL, U∞, ω, CONSTANTS.elemType)
        Kf_r, Cf_r, Mf = Hydro.apply_BCs(globalKf_r, globalCf_r, globalMf, globalDOFBlankingList)
        Kf_i, Cf_i, _ = Hydro.apply_BCs(globalKf_i, globalCf_i, globalMf, globalDOFBlankingList)

        # # --- Test wet eigensolve ---
        # omegaSquared, _ = SolverRoutines.compute_eigsolve(KK, MM .+ Mf, 3)
        # wetNatFreqs = sqrt.(omegaSquared) / (2π)
        # println(wetNatFreqs)
        # # TODO: The wet natural frequencies are off so there might be a bug with the added mass since air is good

        # --- Solve eigenvalue problem ---
        p_r_tmp, p_i_tmp, R_aa_r_tmp, R_aa_i_tmp = solve_eigenvalueProblem(pkEqnType, dimwithBC, b, U∞, FOIL, Mf, Cf_r, Cf_i, Kf_r, Kf_i, MM, KK)

        # --- Sort eigenvalues from small to large ---
        p_r = sort(p_r_tmp)
        idxs = sortperm(p_r_tmp)
        p_i = p_i_tmp[idxs]
        R_aa_r = R_aa_r_tmp[:, idxs]
        R_aa_i = R_aa_i_tmp[:, idxs]

        # --- Mode tracking (prevent mode hopping between k's) ---
        # Don't need mode tracking for the very first step
        if (ik > 1)
            # van Zyl tracking method: Find correlation matrix btwn current and previous eigenvectors (mode shape)
            # Rows are old eigenvector number and columns are new eigenvector number
            corr = compute_correlationMatrix(R_eigs_r[:, :, ik-1], R_eigs_i[:, :, ik-1], R_aa_r, R_aa_i)

            # Determine location of new eigenvectors
            idxs = SolverRoutines.argmax2d(transpose(corr))

            # Check if entries are missing/duplicated 
            # TODO: this might not be a problem

            # --- Order eigenvalues and eigenvectors based on correlation matrix ---
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
                # println("p_diff: ", p_diff)
                failed = true
            end
        end

        # ---------------------------
        #   Check solution
        # ---------------------------
        if failed
            # We need to try some new k guesses. Halve the step
            # println("failed, need to adjust k")
            k = 0.5 * (k - k_history[ik-1]) + k_history[ik-1]
        else # Success
            # Store solution
            p_eigs_r[:, ik] = p_r
            p_eigs_i[:, ik] = p_i
            R_eigs_r[:, :, ik] = R_aa_r
            R_eigs_i[:, :, ik] = R_aa_i
            k_history[ik] = k

            # increment for next k
            ik += 1
            k += Δk

            # Check if we solved the matched Im(p) = k problem
            maxImP = maximum(p_i)
            if (maxImP < k_history[ik-1]) || (k > maxK)
                # Assumes highest mode does NOT cross Im(p) = k line from below later
                # i.e. continue looping until the highest mode crosses the line or we reach maxK
                keepLooping = false
            end # if

        end # if

    end # end while

    ik = ik - 1 # Reduce counter b/c it was incremented in the last iteration

    return p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik
end # end function

function extract_kCrossings(dim, p_eigs_r, p_eigs_i, R_eigs_r, R_eigs_i, k_history, ik, N_MAX_K_ITER)
    """
    Find where solutions intersect Im(p) = k line and interpolate value

    Inputs
    ------
        dim - number of degrees of freedom
        p_eigs_r - unsorted real part of eigenvalues [rad/s]
        p_eigs_i - unsorted imaginary part of eigenvalues [rad/s]
        R_eigs_r - unsorted real part of eigenvectors [-]
        R_eigs_i - unsorted imaginary part of eigenvectors [-]
        k_history - history of k values [rad/s]
        ik - number of k values
        N_MAX_K_ITER - maximum number of k iterations
    Outputs
    -------
        p_cross_r - unsorted eigenvalues that cross Im(p) = k line [rad/s]
        p_cross_i - unsorted eigenvalues that cross Im(p) = k line [rad/s]
        R_cross_r - unsorted eigenvectors that cross Im(p) = k line [-]
        R_cross_i - unsorted eigenvectors that cross Im(p) = k line [-]
        ctr - Number of found matched points
    """
    # --- Initialize outputs ---
    p_cross_r = zeros(Float64, 2 * dim * 5)
    p_cross_i = zeros(Float64, 2 * dim * 5)
    R_cross_r = zeros(Float64, 2 * dim, 2 * dim * 5)
    R_cross_i = zeros(Float64, 2 * dim, 2 * dim * 5)

    # --- Look for crossing of diagonal line Im(p) = k ---
    ctr::Int64 = 1 # counter for number of found matched points and index for arrays
    for ii in 1:2*dim # loop over flutter modes (lines)
        for jj in 1:ik # loop over all reduced frequencies (tracing the mode line)
            if k_history[jj] == 0.0 && abs(p_eigs_i[ii, jj]) < SolutionConstants.mepsLarge # Real root
                # There should be another real root coming up or we already processed 
                # one matching the zero freq
                p_cross_r[ctr] = p_eigs_r[ii, jj]
                p_cross_i[ctr] = p_eigs_i[ii, jj]
                R_cross_r[:, ctr] = R_eigs_r[:, ii, jj]
                R_cross_i[:, ctr] = R_eigs_i[:, ii, jj]
                ctr += 1

            elseif jj < ik # Always true except for last mode
                # Get left and right points
                tmpCrossLeft = p_eigs_i[ii, jj] - k_history[jj]
                tmpCrossRight = p_eigs_i[ii, jj+1] - k_history[jj+1]
                # Get signs (+/-1 or 0)
                tmpCrossLeftSign = sign(tmpCrossLeft)
                tmpCrossRightSign = sign(tmpCrossRight)

                # --- Find sign change ---
                if tmpCrossLeftSign != tmpCrossRightSign
                    # Linear interpolation time since we intersected the Im(p)=k line

                    # You want the point at which Im(p) - k = 0.0
                    factor = (0.0 - (p_eigs_i[ii, jj+1] - k_history[jj+1])) / ((p_eigs_i[ii, jj] - k_history[jj]) - (p_eigs_i[ii, jj+1] - k_history[jj+1]))

                    p_cross_r[ctr] = factor * (p_eigs_r[ii, jj] - p_eigs_r[ii, jj+1]) + p_eigs_r[ii, jj+1]
                    p_cross_i[ctr] = factor * (p_eigs_i[ii, jj] - p_eigs_i[ii, jj+1]) + p_eigs_i[ii, jj+1]

                    # --- Eigenvectors ---
                    # Look at real part of inner product of two eigenvectors m and m+1
                    tmpSum_r = 0.0
                    for ll in 1:2*dim
                        # TODO: is this minus sign actually right?
                        tmpSum_r += R_eigs_r[ll, ii, jj+1] * R_eigs_r[ll, ii, jj] - (-1) * R_eigs_i[ll, ii, jj+1] * R_eigs_i[ll, ii, jj]
                    end
                    if tmpSum_r > 0.0
                        R_cross_r[:, ctr] = factor * (R_eigs_r[:, ii, jj] - R_eigs_r[:, ii, jj+1]) + R_eigs_r[:, ii, jj+1]
                        R_cross_i[:, ctr] = factor * (R_eigs_i[:, ii, jj] - R_eigs_i[:, ii, jj+1]) + R_eigs_i[:, ii, jj+1]
                    else
                        R_cross_r[:, ctr] = factor * (-R_eigs_r[:, ii, jj] - R_eigs_r[:, ii, jj+1]) + R_eigs_r[:, ii, jj+1]
                        R_cross_i[:, ctr] = factor * (-R_eigs_i[:, ii, jj] - R_eigs_i[:, ii, jj+1]) + R_eigs_i[:, ii, jj+1]
                    end

                    ctr += 1

                end
            end
        end
    end

    ctr -= 1 # Decrease counter since last successful iteration incremented it

    return p_cross_r, p_cross_i, R_cross_r, R_cross_i, ctr

end # end function

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
end # end function

end # end module