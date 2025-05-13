# --- Julia 1.11---
"""
@File          :   ComputeHydroFunctions.jl
@Date created  :   2025/02/10
@Last modified :   2025/02/10
@Author        :   Galen Ng
@Desc          :   Compute hydrodynamic cost functions
"""

function compute_profiledrag(aeroSpan, chordLengths, meanChord, qdyn, areaRef, appendageParams, appendageOptions, solverOptions)
    """
    This is the profile drag for the appendage; sum of skin friction and pressure
    Vectorized for chord lengths along the half-span
    """
    if appendageOptions["config"] == "wing" || appendageOptions["config"] == "full-wing"
        WSA = 2 * areaRef # both sides
        ds = 0.5 * aeroSpan / (length(chordLengths))
        WSA_v = 2 * chordLengths
    elseif appendageOptions["config"] == "t-foil"
        WSA = 2 * areaRef + 2 * appendageParams["s_strut"] * mean(appendageParams["c_strut"])
    end

    # Re = solverOptions["Uinf"] * meanChord / solverOptions["nu"]
    # cfittc = 0.075 / (log10(Re) - 2)^2 # flat plate friction coefficient ITTC 1957
    # # Ma = solverOptions["Uinf"] / 1500

    Re_v = solverOptions["Uinf"] / solverOptions["nu"] .* chordLengths
    cfittc_v = 0.075 ./ (log10.(Re_v) .- 2) .^ 2 # flat plate friction coefficient ITTC 1957

    # # --- Raymer equation 12.30 ---
    # xcmax = 0.3 # chordwise position of the maximum thickness
    # FF = (1 .+ 0.6 ./ (xcmax) .* DVDict["toc"] + 100 .* DVDict["toc"].^4) * (1.34*Ma^0.18 * cos(DVDict["sweep"])^0.28)

    # --- Torenbeek 1990 ---
    # First term is increase in skin friction due to thickness and quartic is separation drag
    FF = 1 .+ 2.7 .* appendageParams["toc"] .+ 100 .* appendageParams["toc"] .^ 4

    # FFmean = sum(FF) / length(FF) # average assuming the TOC nodes are spaced evenly
    # Df = qdyn * WSA * cfittc
    # Dpr = Df * FFmean
    # CDpr = Dpr / (qdyn * areaRef)

    Dpr_v = 0.0
    for ii in eachindex(chordLengths)

        if ii == 1 || ii == length(chordLengths)
            Δs = ds * 0.5
        else
            Δs = ds
        end

        Dpr_v += qdyn * WSA_v[ii] * Δs * cfittc_v[ii] * FF[ii]
    end
    if appendageOptions["config"] == "t-foil" || appendageOptions["config"] == "full-wing"
        Dpr_v *= 2
    end
    CDpr_v = Dpr_v / (qdyn * areaRef)


    return CDpr_v, Dpr_v
end

function compute_wavedrag(CL, meanChord, qdyn, areaRef, aeroSpan, appendageParams, solverOptions)

    Fnh = solverOptions["Uinf"] / sqrt(9.81 * appendageParams["depth0"])
    AR = aeroSpan / meanChord
    λ = appendageParams["depth0"] * 2 / aeroSpan

    # # Breslin 1957 wave drag for an ELLIPTIC hydrofoil
    # # ************************************************
    # #     High Fnc approximation
    # # ************************************************
    # Fnc = solverOptions["Uinf"] / sqrt(9.81 * meanChord)
    # if Fnc < 2.0
    #     println("Warning: Fnc < 2.0. Not valid!")
    # end
    # σλ, γλ = compute_biplanefreesurface(λ)

    # CDw = (
    #     σλ / (π * AR) + γλ / Fnc^2
    # ) * CL^2

    # ************************************************
    #     Arbitrary Fnc approximation
    # ************************************************
    σλ, _ = compute_biplanefreesurface(λ)
    besselInt = compute_besselInt(solverOptions["Uinf"], aeroSpan, Fnh)
    CDw = (-σλ / (π * AR) + 8 / (π * AR) * besselInt) * CL^2

    Dw = CDw * qdyn * areaRef

    return CDw, Dw
end

function compute_spraydrag(appendageParams, qdyn)

    t = appendageParams["toc_strut"][end] * appendageParams["c_strut"][end]

    # --- Hörner CHapter 10 ---
    # CDts = 0.24
    # ds = CDts * (qdyn * (t)^2)
    # CDs = ds / (qdyn * ADIM)
    # Chapman 1971 assuming x/c = 0.35
    CDs = 0.009 + 0.013 * appendageParams["toc_strut"][end]
    Ds = CDs * qdyn * t * appendageParams["c_strut"][end]

    return CDs, Ds
end

function compute_junctiondrag(appendageParams, qdyn, rootChord, areaRef)

    # From Hörner Chapter 8
    tocbar = 0.5 * (appendageParams["toc"][1] + appendageParams["toc_strut"][1])
    CDt = 17 * (tocbar)^2 - 0.05
    Dj = CDt * (qdyn * (tocbar * rootChord)^2)
    CDj = Dj / (qdyn * areaRef)

    return CDj, Dj
end

function compute_calmwaterdragbuildup(appendageParams, appendageOptions, solverOptions, qdyn, areaRef, aeroSpan, CL, meanChord, rootChord, chordLengths)
    """
    All pieces of calmwater drag
    """

    CDw, Dw = compute_wavedrag(CL, meanChord, qdyn, areaRef, aeroSpan, appendageParams, solverOptions)

    CDpr, Dpr = compute_profiledrag(aeroSpan, chordLengths, meanChord, qdyn, areaRef, appendageParams, appendageOptions, solverOptions)

    CDj, Dj = compute_junctiondrag(appendageParams, qdyn, rootChord, areaRef)

    CDs, Ds = compute_spraydrag(appendageParams, qdyn)

    return CDw, CDpr, CDj, CDs, Dw, Dpr, Dj, Ds
end

function compute_biplanefreesurface(λ)
    """
    Based on Breslin 1957
    """
    plamsq = 1 + λ^2
    k = 1 / √(1 + λ^2)
    Ek = SpecialFunctions.ellipe(k^2)
    Kk = SpecialFunctions.ellipk(k^2)

    # Biplane function
    σλ = 1 - 4 / π * λ * √(1 + λ^2) * (Kk - Ek)
    # Free surface function
    γλ = 4 / (3π) * ((2 / π) * plamsq^(1.5) * Ek - 1.5 * λ)

    return σλ, γλ
end

function compute_besselInt(Uinf, span, Fnh)

    function compute_integrand(θ)

        J1 = SpecialFunctions.besselj1(GRAV / Uinf^2 * 0.5 * span * (sec(θ))^2 * sin(θ))
        exponent = exp(-2 * (sec(θ))^2 / Fnh^2)


        int = J1^2 * exponent / ((sin(θ))^2 * cos(θ))

        return int
    end

    dθ = 0.01
    # Starting at 0 breaks this, so start close
    θ = 0.001:dθ:π/2
    heights = compute_integrand.(θ)

    # # Trapezoid integration seems to introduce oscillations... wrt Fnc
    # I = 0.5 * dθ * (heights[1] + heights[end] + 2 * sum(heights[2:end-1]))

    # --- Riemann integration ---
    I = sum(heights * dθ)

    return I
end

function compute_dragsFromX(ptVec, gammas, nodeConn, displVec, appendageParams, appendageOptions, solverOptions)


    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = LiftingLine.get_tipnode(LECoords)
    midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    # ---------------------------
    #   Hydrodynamics
    # ---------------------------
    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    displCol = reshape(displVec, 6, length(displVec) ÷ 6)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = setup(Uvec, sweepAng, rootChord, TR, midchords, displCol;
        npt_wing=size(displCol, 2),
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        # airfoilCoordFile=airfoilCoordFile,
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
    _, _, _, _, _, CL, _, _ = LiftingLine.compute_outputs(gammas, TV_influence, FlowCond, LLMesh, LLNLParams)

    meanChord = sum(chordVec) / length(chordVec)
    areaRef = LiftingLine.compute_areas(LECoords, TECoords, nodeConn)
    dynP = 0.5 * FlowCond.rhof * FlowCond.Uinf^2
    aeroSpan = LiftingLine.compute_aeroSpan(midchords, idxTip)
    CDw, CDpr, CDj, CDs, Dw, Dpr, Dj, Ds =
        compute_calmwaterdragbuildup(appendageParams, appendageOptions, solverOptions,
            dynP, areaRef, aeroSpan, CL, meanChord, rootChord, chordVec)

    # NOTE: chordVec type is Vector{Union{Real, Complex}} and that makes CDpr, CDj, Dpr, Dj complex (but 0 for imag part) under compute_totals in hydroelastic derivatives
    #       Zygote doesn't accept complex data type so we convert to real here
    return vec([CDw, real(CDpr), real(CDj), CDs, Dw, real(Dpr), real(Dj), Ds])
end

function compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    """
    This appears to have issues wrt the ptVec inputs
    """
    # backend = AD.ReverseDiffBackend()

    # backend = AD.ZygoteBackend()

    # Since this is a matrix, it needs to be transposed and then unrolled so that the order matches what python needs (this is sneaky)
    displVec = vec(displCol)

    if uppercase(mode) == "RAD" # RAD everything except the ptVec ones because this gave NaNs
        # TODO: BUG here, fix this
        # displacements of the collocation nodes also give NaNs
        ∂Drag∂G, ∂Drag∂xdispl = Zygote.jacobian(
            (xGamma, xDispl) -> compute_dragsFromX(ptVec, xGamma, nodeConn, xDispl, appendageParams, appendageOptions, solverOptions),
            gammas,
            displVec
        )
        # ∂Drag∂Xpt, ∂Drag∂G, ∂Drag∂xdispl = Zygote.jacobian((xPt, xGamma, xDispl) -> compute_dragsFromX(xPt, xGamma, nodeConn, xDispl, appendageParams, appendageOptions, solverOptions), ptVec, gammas, displVec)

        backend = AD.ForwardDiffBackend()

        # Need to FAD displacements too
        # Weird bug, but you can't do the jacobian using multiple inputs at once I guess
        # ∂Drag∂Xpt, ∂Drag∂xdispl = AD.jacobian(backend, (xPt, xDispl) -> compute_dragsFromX(xPt, gammas, nodeConn, xDispl, appendageParams, appendageOptions, solverOptions), ptVec, displVec) # this is bad
        ∂Drag∂Xpt, = AD.jacobian(backend, (xPt) -> compute_dragsFromX(xPt, gammas, nodeConn, displVec, appendageParams, appendageOptions, solverOptions), ptVec)
        ∂Drag∂xdispl, = AD.jacobian(backend, (xDispl) -> compute_dragsFromX(ptVec, gammas, nodeConn, xDispl, appendageParams, appendageOptions, solverOptions), displVec)

    elseif uppercase(mode) == "FIDI"
        outputVector = ["cdw", "cdpr", "cdj", "cds", "dw", "dpr", "dj", "ds"]
        ∂Drag∂Xpt = zeros(DTYPE, length(outputVector), length(ptVec))
        ∂Drag∂xdispl = zeros(DTYPE, length(outputVector), length(displCol))
        dh = 1e-4

        f_i = compute_dragsFromX(ptVec, gammas, nodeConn, displVec, appendageParams, appendageOptions, solverOptions)
        for ii in eachindex(ptVec)

            ptVec[ii] += dh

            f_f = compute_dragsFromX(ptVec, gammas, nodeConn, displVec, appendageParams, appendageOptions, solverOptions)

            ptVec[ii] -= dh

            ∂Drag∂Xpt[:, ii] = (f_f - f_i) / dh
        end
        for ii in eachindex(displVec)

            displVec[ii] += dh

            f_f = compute_dragsFromX(ptVec, gammas, nodeConn, displVec, appendageParams, appendageOptions, solverOptions)

            displVec[ii] -= dh

            ∂Drag∂xdispl[:, ii] = (f_f - f_i) / dh

        end

        backend = AD.ZygoteBackend()
        ∂Drag∂G, = AD.jacobian(backend, (xGamma) -> compute_dragsFromX(ptVec, xGamma, nodeConn, displCol, appendageParams, appendageOptions, solverOptions), gammas)

    elseif uppercase(mode) == "FAD"


        backend = AD.ForwardDiffBackend()
        # @time ∂Drag∂G, = ReverseDiff.jacobian((xGamma) -> compute_dragsFromX(ptVec, xGamma, nodeConn, appendageParams, appendageOptions, solverOptions), gammas)
        ∂Drag∂G, = AD.jacobian(backend, (xGamma) -> compute_dragsFromX(ptVec, xGamma, nodeConn, displVec, appendageParams, appendageOptions, solverOptions), gammas)
        ∂Drag∂xdispl, = AD.jacobian(backend, (xDispl) -> compute_dragsFromX(ptVec, gammas, nodeConn, xDispl, appendageParams, appendageOptions, solverOptions), displVec)
        ∂Drag∂Xpt, = AD.jacobian(backend, (xPt) -> compute_dragsFromX(xPt, gammas, nodeConn, displVec, appendageParams, appendageOptions, solverOptions), ptVec)
    else
        error("Mode not recognized")
    end

    # println("size of ∂I∂xDV: ", (∂I∂xDV))

    # writedlm("ddragdX-$(mode).csv", ∂Drag∂Xpt, ',')
    # writedlm("ddragdG-$(mode).csv", ∂Drag∂G, ',')

    return ∂Drag∂Xpt, ∂Drag∂xdispl, ∂Drag∂G
end

# ************************************************
#     Center of force calculations
# ************************************************
function compute_centerofforce(fHydro, FEMESH)

    Lift = fHydro[WIND:NDOF:end]
    Moments = fHydro[ΘIND:NDOF:end]

    # --- Center of forces ---
    # These calculations are in local appendage frame
    # center of forces in z direction
    xcenter = sum(Lift .* FEMESH.mesh[:, XDIM]) / sum(Lift)
    ycenter = sum(Lift .* FEMESH.mesh[:, YDIM]) / sum(Lift)
    zcenter = sum(Lift .* FEMESH.mesh[:, ZDIM]) / sum(Lift)
    cofz = [xcenter, ycenter, zcenter]


    # center of moments about y axis
    xcenter = sum(Moments .* FEMESH.mesh[:, XDIM]) / sum(Moments)
    ycenter = sum(Moments .* FEMESH.mesh[:, YDIM]) / sum(Moments)
    zcenter = sum(Moments .* FEMESH.mesh[:, ZDIM]) / sum(Moments)
    comy = [xcenter, ycenter, zcenter]

    return cofz, comy
end
