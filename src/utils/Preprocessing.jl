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
using FLOWMath: atan_cs_safe, abs_cs_safe
using ..Utilities: Utilities
using Zygote

function compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions, appendageParams)
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
    size(nodeConn)[1] == 2 || error("nodeConn shape is wrong")

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
    # sweepAngle = -atan_cs_safe.(dxVec, dyVec)


    # ---------------------------
    #   Twist distribution
    # ---------------------------
    twistVec = compute_twist(chordVectors)


    # ---------------------------
    #   Sweep distribution
    # ---------------------------
    sweepAngles, qtrChords = compute_ACSweep(LECoords, TECoords, nodeConn, idxTip, 0.25)
    # Λ = sweepAngles
    Λ = sum(sweepAngles) / length(sweepAngles)
    # println("AC Sweep angle: $(rad2deg(Λ)) deg")


    # ---------------------------
    #   Span
    # ---------------------------
    # aeroSpan = compute_aeroSpan(midchords)
    structSemispan = compute_structSpan(abs.(midchords), idxTip)
    # println("Structural semispan: $(structSemispan) m")

    # ************************************************
    #     Finally, spline them to the discretization
    # ************************************************
    semispan = structSemispan
    nNodes = appendageOptions["nNodes"]
    s_loc_q = LinRange(0.0, semispan, nNodes)

    ds = semispan / (nNodes - 1)
    s_loc_q = LinRange(ds, semispan - ds, nNodes - 2)
    # println("s_loc_q: ", s_loc_q)
    s_loc = vec(sqrt.(sum(midchords[:, 1:idxTip] .^ 2, dims=1)))
    # println("s_loc: ", s_loc)
    chordLengthsWork_interp::Vector{RealOrComplex} = SolverRoutines.do_linear_interp(s_loc, chordLengths[1:idxTip], s_loc_q)
    # This is for half of the wing
    chordLengthsWork = vcat(chordLengths[1], chordLengthsWork_interp, chordLengths[idxTip])
    # qtrChordWork = SolverRoutines.do_linear_interp(s_loc, qtrChords, s_loc_q)
    twistDistribution = SolverRoutines.do_linear_interp(s_loc, twistVec[1:idxTip], s_loc_q)
    # println("chords: ", chordLengthsWork)

    return midchords, chordLengthsWork, spanwiseVectors, Λ, twistDistribution
end

function compute_ACSweep(LECoords, TECoords, nodeConn, idxTip, e=0.25)
    """
    Compute the approximate sweep angle of the aerodynamic center (1/4 chord assumption)
    """

    # --- Compute the approximate sweep angle ---
    # Get the quarter chord line
    qtrChord = ((1 - e) * LECoords .+ e * TECoords)
    n1vec = nodeConn[1, :]
    n2vec = nodeConn[2, :]
    dxVec = qtrChord[XDIM, n2vec] - qtrChord[XDIM, n1vec]
    dyVec = qtrChord[YDIM, n2vec] - qtrChord[YDIM, n1vec]
    dzVec = qtrChord[ZDIM, n2vec] - qtrChord[ZDIM, n1vec]

    # Compute the angle
    # --- Vectorized ---
    sweepAngles = zeros(RealOrComplex, size(dxVec))
    sweepAngles_z = Zygote.Buffer(sweepAngles)
    for (ii, dy) in enumerate(dyVec)
        if real(dy) < 0.0
            sweepAngles_z[ii] = atan_cs_safe(dxVec[ii], -dy)
        else
            sweepAngles_z[ii] = atan_cs_safe(dxVec[ii], dy)
        end
    end
    sweepAngle = copy(sweepAngles_z)
    # dx = qtrChord[XDIM, idxTip] - qtrChord[XDIM, 1]
    # dy = qtrChord[YDIM, idxTip] - qtrChord[XDIM, 1]
    # # println("dx: $(dx), dy: $(dy)")
    # # println(idxTip)
    # sweepAngle = atan_cs_safe(-dx, dy)

    # sweepAngles = [atan_cs_safe(dx, dy) for (dx, dy) in zip(dxVec, dyVec) if dy > 0.0]
    # println("Sweep angles: ", sweepAngle)

    return sweepAngle, qtrChord
end

function compute_aeroSpan(midchords, idxTip)

    # ymax = Utilities.compute_KS(midchords[YDIM, :], 100.0)
    # ymin = -Utilities.compute_KS(-midchords[YDIM, :], 100.0)
    # aeroSpan = ymax - ymin
    aeroSpan = 2 * (midchords[YDIM, idxTip])

    # print("Aerodynamic span: ", aeroSpan)
    return aeroSpan
end

function compute_structSpan(midchords, idxTip)

    # sVecs = .√(midchords[XDIM, idxTip] .^ 2 + midchords[YDIM, idxTip] .^ 2 + midchords[ZDIM, idxTip] .^ 2)
    # smax = Utilities.compute_KS(sVecs, 100.0)
    smax = .√(midchords[XDIM, idxTip] .^ 2 + midchords[YDIM, idxTip] .^ 2 + midchords[ZDIM, idxTip] .^ 2)

    # print("Structural semispan: ", smax)
    return smax
end

function compute_twist(chordVectors)

    # Compute the twist distribution based off of angle about midchord
    dxVec = chordVectors[XDIM, :]
    dzVec = chordVectors[ZDIM, :]
    twistVec = atan_cs_safe.(dzVec, dxVec)

    if abs(π - abs(twistVec[1])) < MEPSLARGE
        twistVec = twistVec .- π
    end
    # println("Twist vec: ", twistVec)

    return twistVec
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

function get_tipnode(LECoords)
    """
    Based on unaltered coordinates, get the indices of the tip and root mesh points
    Need unaltered ones for differentiation purposes
    """

    idxTip = argmax(LECoords[YDIM, :])
    # if idxTip != 10
    #     println("Tip node index (hopefully not changing): ", idxTip)
    # end
    return idxTip
end

function compute_areas(LECoords, TECoords, nodeConn)

    areaRef = 0.0

    for (ii, inds) in enumerate(eachcol(nodeConn))
        # --- Compute chord ---
        chord1 = TECoords[XDIM, inds[1]] - LECoords[XDIM, inds[1]]
        chord2 = TECoords[XDIM, inds[2]] - LECoords[XDIM, inds[2]]

        midchords = 0.5 .* (LECoords .+ TECoords)
        Δy = abs_cs_safe(midchords[YDIM, inds[2]] - midchords[YDIM, inds[1]])

        # --- Compute area ---
        areaRef += 0.5 * (chord1 + chord2) * Δy
    end

    # println("Planform area: $(areaRef) m^2")

    return areaRef
end

end