# --- Julia 1.11---
"""
@File          :   ComputeHydroFunctions.jl
@Date created  :   2025/02/10
@Last modified :   2025/02/10
@Author        :   Galen Ng
@Desc          :   Compute hydrodynamic cost functions
"""




function compute_kscl(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
    # function compute_kscl(fHydro, qdyn, chords, solverOptions)
    """
    Compute the KS function for the spanwise lift coefficient
        Needed for the ventilation constraint

    2024-12-24: All of the derivatives for this are good!
    """

    # This way would be off of the lifting line directly
    LLOutputs, _, _ = HydroStrip.compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

    clmax = compute_KS(LLOutputs.cl, solverOptions["rhoKS"])
    # println("cls across span", LLOutputs.cl)

    return clmax
end

function compute_vortexdrag(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    LLOutputs, _, _ = HydroStrip.compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

    Di = LLOutputs.F[XDIM]
    CDi = LLOutputs.CDi

    return CDi, Di
end

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
    # σλ, γλ = HydroStrip.compute_biplanefreesurface(λ)

    # CDw = (
    #     σλ / (π * AR) + γλ / Fnc^2
    # ) * CL^2

    # ************************************************
    #     Arbitrary Fnc approximation
    # ************************************************
    σλ, _ = HydroStrip.compute_biplanefreesurface(λ)
    besselInt = HydroStrip.compute_besselint(solverOptions["Uinf"], aeroSpan, Fnh)
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
