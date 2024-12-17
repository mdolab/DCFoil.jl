# --- Julia 1.11---
"""
@File          :   Preprocessing.jl
@Date created  :   2024/10/17
@Last modified :   2024/10/17
@Author        :   Galen Ng
@Desc          :   Routines to derive 1D quantities from a mesh
"""

module Preprocessing

# # --- PACKAGES ---
# using Zygote
# using ChainRulesCore
# using DelimitedFiles
# using LinearAlgebra
# using StaticArrays

# --- DCFoil modules ---
using ..DCFoil: RealOrComplex, DTYPE
# using ..EBBeam: EBBeam as BeamElement, NDOF
using ..SolverRoutines: SolverRoutines
# using ..BeamProperties
# using ..DesignConstants: DynamicFoil
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE
using FLOWMath: atan_cs_safe
using ..Utilities: Utilities
using Zygote

function compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions, appendageParams)
    """
    Compute 1D quantities from the grid
    """

    # # --- Unpack the grid ---
    # LECoords = GridStruct.LECoords
    # TECoords = GridStruct.TECoords
    # nodeConn = GridStruct.nodeConn

    # Double check coords shape 
    # println("LECoords shape: ", size(LECoords))
    size(LECoords)[1] == 3 || error("LECoords shape is wrong")


    # ************************************************
    #     Compute 1D quantities
    # ************************************************
    # Midchords
    midchords::Matrix{RealOrComplex} = 0.5 .* (LECoords .+ TECoords)

    # Chord lengths
    chordVectors = LECoords .- TECoords
    chordLengths::Vector{RealOrComplex} = vec(sqrt.(sum(chordVectors .^ 2, dims=1)))


    # About midchord
    # spanwiseVectors = zeros(RealOrComplex, 3, size(nodeConn)[2])
    # for (ii, inds) in enumerate(eachcol(nodeConn))
    #     dx = midchords[XDIM, inds[2]] - midchords[XDIM, inds[1]]
    #     dy = midchords[YDIM, inds[2]] - midchords[YDIM, inds[1]]
    #     dz = midchords[ZDIM, inds[2]] - midchords[ZDIM, inds[1]]
    #     spanwiseVectors[:, ii] = [dx, dy, dz]
    # end
    n1vec = nodeConn[1, :]
    n2vec = nodeConn[2, :]
    dxVec = midchords[XDIM, n2vec] - midchords[XDIM, n1vec]
    dyVec = midchords[YDIM, n2vec] - midchords[YDIM, n1vec]
    dzVec = midchords[ZDIM, n2vec] - midchords[ZDIM, n1vec]
    spanwiseVectors = [dxVec; dyVec; dzVec] # 3 x nNodes
    # Compute the angle
    sweepAngle = -atan_cs_safe.(dxVec, dyVec)


    # ---------------------------
    #   Twist distribution
    # ---------------------------
    # Compute the twist distribution based off of angle about midchord
    dxVec = chordVectors[XDIM, :]
    dzVec = chordVectors[ZDIM, :]
    twistVec = atan_cs_safe.(dzVec, dxVec)
    if abs(π - abs(twistVec[1])) < MEPSLARGE
        twistVec = twistVec .- π
    end
    # println("Twist vec: ", twistVec)


    # ---------------------------
    #   Sweep distribution
    # ---------------------------
    sweepAngles, qtrChords = compute_ACSweep(LECoords, TECoords, nodeConn, 0.25)
    Λ = sum(sweepAngles) / length(sweepAngles)
    # if abs(π - abs(sweepAngles[1])) > π / 2
    #     sweepAngles = sweepAngles .+ π
    # end
    # println("Sweep angle: ", Λ)


    # println("Midchords: ")
    # for xyz in eachcol(midchords)
    #     println("$(xyz)")
    # end
    # println("spanwiseVectors: ")
    # for xyz in eachcol(spanwiseVectors)
    #     println("$(xyz)")
    # end
    # println("Chord lengths: ", chordLengths)

    # ************************************************
    #     Finally, spline them to the discretization
    # ************************************************
    semispan = appendageParams["s"]
    nNodes = appendageOptions["nNodes"]
    s_loc_q = LinRange(0.0, semispan, nNodes)
    s_loc = vec(sqrt.(sum(midchords .^ 2, dims=1)))
    chordLengthsWork::Vector{RealOrComplex} = SolverRoutines.do_linear_interp(s_loc, chordLengths, s_loc_q)
    # qtrChordWork = SolverRoutines.do_linear_interp(s_loc, qtrChords, s_loc_q)


    return midchords, chordLengthsWork, spanwiseVectors, Λ
end

function compute_ACSweep(LECoords, TECoords, nodeConn, e=0.25)
    """
    Compute the approximate sweep angle of the aerodynamic center (1/4 chord assumption)
    """

    # --- Compute the approximate sweep angle ---
    # Get the quarter chord line
    qtrChord = ((1 - e) * LECoords .+ e * TECoords) / 2
    n1vec = nodeConn[1, :]
    n2vec = nodeConn[2, :]
    dxVec = qtrChord[XDIM, n2vec] - qtrChord[XDIM, n1vec]
    dyVec = qtrChord[YDIM, n2vec] - qtrChord[YDIM, n1vec]
    dzVec = qtrChord[ZDIM, n2vec] - qtrChord[ZDIM, n1vec]

    # Compute the angle
    sweepAngles = zeros(RealOrComplex, size(dxVec))
    sweepAngles_z = Zygote.Buffer(sweepAngles)
    for (ii, dy) in enumerate(dyVec)
        if real(dy) < 0.0
            sweepAngles_z[ii] = atan_cs_safe(dxVec[ii], -dy)
        else
            sweepAngles_z[ii] = atan_cs_safe(dxVec[ii], dy)
        end
    end

    return copy(sweepAngles_z), qtrChord
end

function compute_aeroSpan(midchords)

    ymax = Utilities.compute_KS(midchords[YDIM, :], 100.0)
    ymin = -Utilities.compute_KS(-midchords[YDIM, :], 100.0)
    aeroSpan = ymax - ymin

    return aeroSpan
end

function compute_structSpan(midchords)

    sVecs = .√(midchords[XDIM, :] .^ 2 + midchords[YDIM, :] .^ 2 + midchords[ZDIM, :] .^ 2)
    smax = Utilities.compute_KS(sVecs, 100.0)
    
    return smax
end

function get_1DBeamPropertiesFromFile(fname)
    """
    Get beam structural properties from file
    The file needs to be in the same format as ASWING:
    <varname>
    <values>

    Returns:
        EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ (like compute_beam() function)
    """

    return EIₛ, EIIPₛ, Kₛ, GJₛ, Sₛ, EAₛ, Iₛ, mₛ
end

function get_1DGeoPropertiesFromFile(fname)
    """
    Get beam geometric properties from file
    The file needs to be in the format:
    <varname>
    <values>

    Returns
    """
end

end