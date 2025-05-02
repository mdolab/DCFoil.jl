# --- Julia 1.11---
"""
@File          :   Preprocessing.jl
@Date created  :   2024/10/17
@Last modified :   2024/10/17
@Author        :   Galen Ng
@Desc          :   Routines to derive 1D quantities from a mesh
"""

using FLOWMath: atan_cs_safe

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
    ChainRulesCore.ignore_derivatives() do # these lines caused NaNs
        size(LECoords)[1] == 3 || error("LECoords shape is wrong")
        size(nodeConn)[1] == 2 || error("nodeConn shape is wrong")
    end

    # ************************************************
    #     Compute 1D quantities
    # ************************************************
    # Midchords
    midchords = compute_midchords(LECoords, TECoords)

    # About midchord
    # spanwiseVectors = zeros(Real, 3, size(nodeConn)[2])
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
    spanwiseVectors = [dxVec, dyVec, dzVec] # 3 x nNodes
    # Compute the angle
    # sweepAngle = -atan_cs_safe.(dxVec, dyVec)


    # ---------------------------
    #   Twist distribution
    # ---------------------------
    twistVec = compute_twist(LECoords, TECoords)

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
    # semispan = structSemispan
    # semispan = LECoords[YDIM, idxTip]
    semispan = midchords[YDIM, idxTip]
    nNodes = appendageOptions["nNodes"]
    # s_loc_q = LinRange(0.0, semispan, nNodes) # old way

    ds = semispan / (nNodes - 1)
    s_loc_q = LinRange(ds, semispan - ds, nNodes - 2)
    # println("s_loc_q: ", s_loc_q)
    # s_loc = vec(sqrt.(sum(midchords[:, 1:idxTip] .^ 2, dims=1))) # this line gave NaNs in the derivatives!
    # s_loc = compute_spanwiseLocations(midchords[XDIM, :], midchords[YDIM, :], midchords[ZDIM, :])
    # println("s_loc: ", s_loc)
    # s_loc = LECoords[YDIM, 1:idxTip]
    s_loc = midchords[YDIM, 1:idxTip]

    chordLengths = compute_chordLengths(LECoords[XDIM, :], TECoords[XDIM, :], LECoords[YDIM, :], TECoords[YDIM, :], LECoords[ZDIM, :], TECoords[ZDIM, :])
    # How is this line giving NaNs?
    chordLengthsWork_interp = do_linear_interp(s_loc, chordLengths[1:idxTip], s_loc_q)

    # This is for half of the wing
    chordLengthsWork = vcat(chordLengths[1], chordLengthsWork_interp, chordLengths[idxTip])

    # qtrChordWork = Interpolation.do_linear_interp(s_loc, qtrChords, s_loc_q)
    twistDistribution = do_linear_interp(s_loc, twistVec[1:idxTip], s_loc_q)
    # println("chords: ", chordLengthsWork)

    return midchords, chordLengthsWork, spanwiseVectors, Λ, twistDistribution
end

function compute_midchords(LECoords, TECoords)
    """
    AD safe
    """
    midchordsX = 0.5 * (LECoords[XDIM, :] .+ TECoords[XDIM, :])
    midchordsY = 0.5 * (LECoords[YDIM, :] .+ TECoords[YDIM, :])
    midchordsZ = 0.5 * (LECoords[ZDIM, :] .+ TECoords[ZDIM, :])
    midchords = transpose(hcat(midchordsX, midchordsY, midchordsZ))

    return midchords
end

function ChainRulesCore.rrule(::typeof(compute_midchords), LECoords, TECoords)

    y = compute_midchords(LECoords, TECoords)

    function pullback(ȳ)
        """
        Pullback for compute_midchords

        ȳ - the seed for the pullback, same size as output of function, in this case (3,NPT)
        """

        dydLECoords = 0.5 * ones(size(LECoords, 1), size(LECoords, 2))
        dydTECoords = 0.5 * ones(size(TECoords, 1), size(TECoords, 2))

        # Compute the pullback
        LECoordsb = ȳ .* dydLECoords # should be size (3, NPT)
        TECoordsb = ȳ .* dydTECoords # should be size (3, NPT)
        # println("size(LECoordsb): ", size(LECoordsb))

        # Return the pullback
        return NoTangent(), LECoordsb, TECoordsb
    end

    return y, pullback
end

function compute_chordLengths(xLE, xTE, yLE, yTE, zLE, zTE)
    """
    Compute chord lengths
    """

    LECoords = transpose(hcat(xLE, yLE, zLE))
    TECoords = transpose(hcat(xTE, yTE, zTE))
    chordVectors = LECoords .- TECoords
    # This is a 3 x NPT matrix

    # Chord lengths
    chordLengths = zeros(Real, size(chordVectors)[2])
    for ii in eachindex(eachcol(chordVectors))
        chordLengths[ii] = √(chordVectors[XDIM, ii] .^ 2 + chordVectors[YDIM, ii] .^ 2 + chordVectors[ZDIM, ii] .^ 2)
    end

    return chordLengths
end

function ChainRulesCore.rrule(::typeof(compute_chordLengths), xLE, xTE, yLE, yTE, zLE, zTE)

    y = compute_chordLengths(xLE, xTE, yLE, yTE, zLE, zTE)

    function chordLengths_pullback(ȳ)
        """
        Pullback for compute_chordLengths

        ȳ - the seed for the pullback, same size as output of function, in this case (NPT)
        """

        # ∂Chordlength∂LECoords -->
        NPT = length(ȳ)
        dydxLE = zeros(NPT, NPT)
        dydxTE = zeros(NPT, NPT)
        dydyLE = zeros(NPT, NPT)
        dydyTE = zeros(NPT, NPT)
        dydzLE = zeros(NPT, NPT)
        dydzTE = zeros(NPT, NPT)
        for ii in 1:NPT # populate main diagonal entries
            dxii = xLE[ii] - xTE[ii]
            dyii = yLE[ii] - yTE[ii]
            dzii = zLE[ii] - zTE[ii]
            denom = sqrt(dxii^2 + dyii^2 + dzii^2)
            dydxLE[ii, ii] = (dxii) / denom
            dydyLE[ii, ii] = (dyii) / denom
            dydzLE[ii, ii] = (dzii) / denom
            dydxTE[ii, ii] = -(dxii) / denom
            dydyTE[ii, ii] = -(dyii) / denom
            dydzTE[ii, ii] = -(dzii) / denom
        end

        # Compute the pullback (vector - matrix products)
        xLEb = vec(transpose(ȳ) * dydxLE)
        xTEb = vec(transpose(ȳ) * dydxTE)
        yLEb = vec(transpose(ȳ) * dydyLE)
        yTEb = vec(transpose(ȳ) * dydyTE)
        zLEb = vec(transpose(ȳ) * dydzLE)
        zTEb = vec(transpose(ȳ) * dydzTE)
        # LECoordsb = ȳ # dummy
        # TECoordsb = ȳ # dummy

        # Return the pullback
        return NoTangent(), xLEb, xTEb, yLEb, yTEb, zLEb, zTEb
    end

    return y, chordLengths_pullback

end

# TODO: GN - make this AD safe, for now, using LECoords YDIM is OK
function compute_spanwiseLocations(xMC, yMC, zMC)
    """
    Compute the spanwise locations of the mesh points
    """

    # Compute the spanwise locations
    s = zeros(size(xMC))
    for ii in eachindex(xMC)
        s[ii] = √(xMC[ii] .^ 2 + yMC[ii] .^ 2 + zMC[ii] .^ 2)
    end

    return s
end

function ChainRulesCore.rrule(::typeof(compute_spanwiseLocations), xMC, yMC, zMC)

    y = compute_spanwiseLocations(xMC, yMC, zMC)

    function spanwiseLocations_pullback(ȳ)
        """
        Pullback for compute_spanwiseLocations

        ȳ - the seed for the pullback, same size as output of function, in this case (NPT)
        """

        # ∂spanLoc∂MCCoords -->
        NPT = length(ȳ)
        dydxMC = zeros(NPT, NPT)
        dydyMC = zeros(NPT, NPT)
        dydzMC = zeros(NPT, NPT)
        for ii in 1:NPT # populate main diagonal entries
            dxii = xMC[ii]
            dyii = yMC[ii]
            dzii = zMC[ii]
            denom = sqrt(dxii^2 + dyii^2 + dzii^2)
            dydxMC[ii, ii] = (dxii) / denom
            dydyMC[ii, ii] = (dyii) / denom
            dydzMC[ii, ii] = (dzii) / denom
        end

        # Compute the pullback (vector - matrix products)
        xMCb = vec(transpose(ȳ) * dydxMC)
        yMCb = vec(transpose(ȳ) * dydyMC)
        zMCb = vec(transpose(ȳ) * dydzMC)

        # Return the pullback
        return NoTangent(), xMCb, yMCb, zMCb
    end

    return y, spanwiseLocations_pullback

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
    sweepAngles = zeros(Real, size(dxVec))
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
    """
    Compute the full wing span
    """

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

function compute_twist(LECoords, TECoords)
    chordVectors = LECoords .- TECoords

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

    idxTip = argmax(real(LECoords)[YDIM, :])
    # println("idx of the tip node: ", idxTip)
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
