"""
--- Julia ---

@File    :   HydroStrip.jl
@Time    :   2022/05/18
@Author  :   Galen Ng
@Desc    :   Contains hydrodynamic routines and interfaces to other codes
"""

module HydroStrip

# --- Public functions ---
export compute_theodorsen, compute_glauert_circ
export compute_node_mass, compute_node_damp, compute_node_stiff
export compute_AICs, apply_BCs

# --- PACKAGES ---
using SpecialFunctions
using LinearAlgebra
using Statistics
using Zygote
using ChainRulesCore: ChainRulesCore, @ignore_derivatives
using Printf, DelimitedFiles
using Plots
using FLOWMath: norm_cs_safe, abs_cs_safe
# using SparseArrays
# using Debugger
# using Cthulhu

# --- DCFoil modules ---
using ..SolverRoutines
using ..Unsteady: compute_theodorsen, compute_sears, compute_node_stiff_faster, compute_node_damp_faster, compute_node_mass, compute_node_stiff_dcla
using ..GlauertLL: GlauertLL
using ..LiftingLine: LiftingLine, Δα
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE, GRAV, ELEMTYPE
using ..EBBeam: EBBeam as BeamElement, NDOF
using ..DCFoil: RealOrComplex, DTYPE
using ..DesignConstants: CONFIGS
using ..Preprocessing
using ..Utilities
using ..FEMMethods
using ..Interpolation
using ..Rotations

# ==============================================================================
#                         Free surface effects
# ==============================================================================
# The following functions compute the generic force coefficients 'C' for the equation
#     C = Ci  α̈ + Cd α̇ + Cs α
# However, none of the ROM appears to account for heave effects
# The ROM should not be used for k > 0.2 and should DEFINITELY not be used for k > 1.0

function compute_clsROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = 5.268
    p10 = 0.217
    p01 = -6.085
    p20 = -0.0141
    p11 = -0.0425
    p02 = 4.586
    p12 = 0.0
    p03 = 0.0
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

function compute_cldROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = 0.0837
    p10 = -0.0192
    p01 = -5.597
    p20 = 0.0
    p11 = 0.0251
    p02 = 26.662
    p12 = 0.00304
    p03 = -16.218
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

#  NOTE: the moments are about the elastic axis
function compute_cmsROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = 0.0633
    p10 = -0.00883
    p01 = -0.000890
    p20 = 0.000634
    p11 = 0.00106
    p02 = -0.127
    p12 = 0.0
    p03 = 0.0
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

function compute_cmdROM(k, hcRatio, Fnc)
    """
    Compute unsteady force coeff with free-surface effect using a polynomial fit
    Kennedy, R. C., Helfers, D., Young, Y. L. (2015). A Reduced-Order Model for an Oscillating Hydrofoil near the Free Surface. SNAME FAST. http://onepetro.org/snamefast/proceedings-pdf/FAST15/3-FAST15/D031S014R003/2434879/sname-fast-2015-062.pdf/1
    """
    if Fnc < 4
        println("Fnc must be greater than 4 to be independent of free surface")
        # If you're above this, then you can keep using the same added mass formulation
    end
    if k >= 0.2
        println("Error due to higher k")
        # This error is because the vortex sheet is not flat anymore
    end
    p00 = -0.000675
    p10 = 0.000320
    p01 = -1.023#*0.5
    p20 = 0.0
    p11 = -0.00355#*0.25
    p02 = -0.177#*0.25
    p12 = 0.0
    p03 = 0.0
    kSquared = k * k
    CForce = p00 + p10 * hcRatio + p01 * k + p20 * hcRatio * hcRatio + p11 * hcRatio * k + p02 * kSquared + p12 * hcRatio * kSquared + p03 * kSquared * k
    return CForce
end

function compute_pFactor(chordLocal, hLocal)
    """
    Compute the infinite frequency correction factor 'p' for the added mass near a free surface.
    Based on empirical corrections from Besch and Liu 1971
    """

    ARi = (hLocal) / chordLocal

    # if hLocal > 0.5 * h
    #     println("The local depth is greater than half the total depth and the correction factor should depend on the foil configuration...")

    # p_i = 1.0 # for struts of T-foils
    # else
    p_i = ARi / √(1 + ARi^2)
    # end


    return p_i
end
# ==============================================================================
#                         Hydro forces
# ==============================================================================
function compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=false)
    """
    This is a wrapper
    """
    LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = Preprocessing.get_tipnode(LECoords)
    midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    # ---------------------------
    #   Hydrodynamics
    # ---------------------------
    LLOutputs, LLSystem, FlowCond = compute_hydroLLProperties(midchords, chordLengths, Λ; appendageParams=appendageParams, solverOptions=solverOptions, appendageOptions=appendageOptions)

    if return_all
        return LLOutputs, LLSystem, FlowCond
    else
        return LLOutputs.cla
    end
end

function compute_dcladX(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="FiDi")
    """
    Derivative of the lift slope wrt the design variables
    """

    npt_wing = 40 # HARDCODED IN LIFTING LINE CODE

    dcldX_f = zeros(npt_wing, length(ptVec))
    dcldX_i = zeros(npt_wing, length(ptVec))
    appendageParams_da = copy(appendageParams)
    appendageParams_da["alfa0"] = appendageParams["alfa0"] + Δα

    if uppercase(mode) == "FIDI" # very different from what adjoint gives. I think it's because of edge nodes as those show the highest discrepancy in the derivatives

        dh = 1e-7
        idh = 1 / dh

        # ************************************************
        #     First time with current angle of attack
        # ************************************************
        LLOutputs_i, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)
        f_i = LLOutputs_i.cl
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LLOutputs_f, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

            f_f = LLOutputs_f.cl

            dcldX_i[:, ii] = (f_f - f_i) * idh

            ptVec[ii] -= dh
        end

        # writedlm("dcldX_i-$(mode).csv", dcldX_i, ',')

        # ************************************************
        #     Second time with perturbed angle of attack
        # ************************************************

        LLOutputs_i, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions; return_all=true)
        f_i = LLOutputs_i.cl
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LLOutputs_f, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions; return_all=true)
            f_f = LLOutputs_f.cl
            dcldX_f[:, ii] = (f_f - f_i) * idh

            ptVec[ii] -= dh
        end
        # writedlm("dcldX_f-$(mode).csv", dcldX_f, ',')


    elseif uppercase(mode) == "IMPLICIT"
        function compute_directMatrix(∂r∂u, ∂r∂xPt)

            Φ = ∂r∂u \ ∂r∂xPt
            return Φ
        end
        # ************************************************
        #     First time with current angle of attack
        # ************************************************
        LLOutputs_i, _, FlowCond = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)
        Gconv = LLOutputs_i.Γdist / FlowCond.Uinf

        # println("∂r∂Γ") # 2 sec, so it's fast
        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

        # println("∂r∂Xpt") # ACCELERATE THIS!!?
        # Takes about 4 sec in pure julia and about 20sec from python
        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions;
            # mode="FAD",
            mode="FiDi", # fastest
        )

        ∂cl∂Γ = diagm(2 * LLOutputs_i.cl ./ LLOutputs_i.Γdist)
        ∂cl∂X = zeros(npt_wing, length(ptVec)) # There's no dependence

        Φ = compute_directMatrix(∂r∂Γ, ∂r∂xPt)
        dcldX_i = ∂cl∂X - ∂cl∂Γ * Φ


        # ************************************************
        #     Second time with perturbed angle of attack
        # ************************************************
        LLOutputs_f, _, FlowCond = compute_cla_API(ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions; return_all=true)
        Gconv = LLOutputs_f.Γdist / FlowCond.Uinf

        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions)

        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv, ptVec, nodeConn, appendageParams_da, appendageOptions, solverOptions)
        ∂cl∂Γ = diagm(2 * LLOutputs_f.cl ./ LLOutputs_f.Γdist)

        Φ = compute_directMatrix(∂r∂Γ, ∂r∂xPt)
        dcldX_f = ∂cl∂X - ∂cl∂Γ * Φ

    end

    dcladXpt = (dcldX_f - dcldX_i) ./ Δα

    return dcladXpt
end

function compute_dcdidXpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    """
    Derivative of the induced drag wrt the design variables
    """

    npt_wing = 40 # HARDCODED IN LIFTING LINE CODE

    dcdidXpt = zeros(1, length(ptVec))
    LLOutputs_i, LLMesh, FlowCond = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)


    if uppercase(mode) == "FIDI" # very different from what adjoint gives. I think it's because of edge nodes as those show the highest discrepancy in the derivatives

        dh = 1e-5
        idh = 1 / dh

        f_i = LLOutputs_i.CDi
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LLOutputs_f, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

            f_f = LLOutputs_f.CDi

            dcdidXpt[1, ii] = (f_f - f_i) * idh

            ptVec[ii] -= dh
        end


    elseif uppercase(mode) == "ADJOINT"

        function compute_adjointVec(∂r∂u, ∂f∂uT)
            ψ = transpose(∂r∂u) \ ∂f∂uT
            return ψ
        end

        Gconv = LLOutputs_i.Γdist / FlowCond.Uinf

        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

        ∂cdi∂Γ = LiftingLine.compute_∂cdi∂Γ(Gconv, LLMesh, FlowCond) # GOOD
        ∂cdi∂X = LiftingLine.compute_∂cdi∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi") # GOOD

        ∂f∂uT = reshape(∂cdi∂Γ, size(∂cdi∂Γ)..., 1)

        Ψ = compute_adjointVec(∂r∂Γ, ∂f∂uT)
        dcdidXpt = ∂cdi∂X - transpose(Ψ) * ∂r∂xPt

        # println("∂cdi∂X", ∂cdi∂X)
        # println("∂cdi∂Γ", ∂cdi∂Γ)


    elseif uppercase(mode) == "DIRECT"
        function compute_directMatrix(∂r∂u, ∂r∂xPt)
            Φ = ∂r∂u \ ∂r∂xPt
            return Φ
        end
        Gconv = LLOutputs_i.Γdist / FlowCond.Uinf
        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)


        ∂cdi∂Γ = LiftingLine.compute_∂cdi∂Γ(Gconv, LLMesh, FlowCond) # NEW
        ∂cdi∂X = LiftingLine.compute_∂cdi∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions) # NEW
        Φ = compute_directMatrix(∂r∂Γ, ∂r∂xPt)
        dcdidXpt = ∂cdi∂X - reshape(∂cdi∂Γ, 1, length(∂cdi∂Γ)) * Φ
    end

    # writedlm("dcdidXpt-$(mode).csv", dcdidXpt, ',')
    # println("writing dcdidXpt-$(mode).csv")

    return dcdidXpt
end

function compute_dDidXpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    """
    Derivative of the induced drag wrt the design variables
    Dimensional 
    TODO PICKUP WRITING THIS
    """

    npt_wing = 40 # HARDCODED IN LIFTING LINE CODE

    dDidXpt = zeros(1, length(ptVec))
    LLOutputs_i, LLMesh, FlowCond = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)


    if uppercase(mode) == "FIDI" # very different from what adjoint gives. I think it's because of edge nodes as those show the highest discrepancy in the derivatives

        dh = 1e-5
        idh = 1 / dh

        f_i = LLOutputs_i.F[1]
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LLOutputs_f, _, _ = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

            f_f = LLOutputs_f.F[1]

            dDidXpt[1, ii] = (f_f - f_i) * idh

            ptVec[ii] -= dh
        end


    elseif uppercase(mode) == "ADJOINT"

        function compute_adjointVec(∂r∂u, ∂f∂uT)
            ψ = transpose(∂r∂u) \ ∂f∂uT
            return ψ
        end

        Gconv = LLOutputs_i.Γdist / FlowCond.Uinf

        ∂r∂Γ = LiftingLine.compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)
        ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

        ∂Di∂Γ = LiftingLine.compute_∂Di∂Γ(Gconv, LLMesh, FlowCond) # GOOD
        ∂Di∂X = LiftingLine.compute_∂Di∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi") # GOOD

        ∂f∂uT = reshape(∂Di∂Γ, size(∂Di∂Γ)..., 1)

        Ψ = compute_adjointVec(∂r∂Γ, ∂f∂uT)
        dDidXpt = ∂Di∂X - transpose(Ψ) * ∂r∂xPt


    end

    # writedlm("dcdidXpt-$(mode).csv", dcdidXpt, ',')
    # println("writing dcdidXpt-$(mode).csv")

    return dDidXpt
end

function compute_hydroLLProperties(midchords, chordVec, sweepAng; appendageParams, solverOptions, appendageOptions)
    """
    Wrapper function to the hydrodynamic lifting line properties
    In this case, α is the angle by which the flow velocity vector is rotated, not the geometry

    Outputs
    -------
    cla: vector
        Lift slope wrt angle of attack (rad^-1) for each spanwise station
    """

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]
    if solverOptions["use_nlll"]

        @ignore_derivatives() do
            if appendageOptions["config"] == "wing"
                println("WARNING: NL LL is only for symmetric wings")
            end
        end
        # println("Using nonlinear lifting line")

        # Hard-coded NACA0012
        @ignore_derivatives() do
            airfoilCoordFile = "$(pwd())/INPUT/PROFILES/NACA0012.dat"
        end

        airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)

        LLSystem, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords;
            npt_wing=npt_wing,
            npt_airfoil=npt_airfoil,
            rhof=solverOptions["rhof"],
            # airfoilCoordFile=airfoilCoordFile,
            airfoil_ctrl_xy=airfoilCtrlXY,
            airfoil_xy=airfoilXY,
            options=@ignore_derivatives(options),
        )
        LLOutputs = LiftingLine.solve(FlowCond, LLSystem, LLHydro, Airfoils, AirfoilInfluences)

        # Fdist = LLOutputs.Fdist
        # F = LLOutputs.F
        # cla = LLOutputs.cla
        # CDi = LLOutputs.CDi

    else
        span = appendageParams["s"] * 2
        cla, Fxind, CDi = GlauertLL.compute_glauert_circ(0.5 * span, chordVec, deg2rad(α0 + rake), solverOptions["Uinf"];
            h=depth0,
            useFS=solverOptions["use_freeSurface"],
            rho=solverOptions["rhof"],
            # config=foilOptions["config"], # legacy option
            debug=solverOptions["debug"]
        )
        F = [Fxind, 0.0, 0.0, 0.0, 0.0, 0.0]
        uinf = solverOptions["Uinf"]
        ZM = zeros(2, 2)
        ZA = zeros(2)
        ZH = zeros(2, 2, 2)
        LLSystem = LiftingLine.LiftingLineMesh(ZM, ZM, ZM, 1, ZA, ZA, ZM, ZA, ZA, 1, span, 0.0, 0.0, 0.0, 0.0, sweepAng, 0.0, ZH, ZH, ZA, ZM, ZA)
        FlowCond = LiftingLine.FlowConditions([uinf, 0.0, 0.0], uinf, [uinf, 0.0, 0.0], appendageParams["alfa0"], 0.0, solverOptions["rhof"], appendageParams["depth0"])
        LLOutputs = LiftingLine.LiftingLineOutputs(zeros(3, 3), zeros(3), real(cla), zeros(3), F, 0.0, CDi, 0.0)
    end

    return LLOutputs, LLSystem, FlowCond
end



function correct_downwash(
    iComp::Int64, CLMain::DTYPE, DVDictList, solverOptions
)
    """
    """
    DVDict = DVDictList[iComp]
    Uinf = solverOptions["Uinf"]
    depth = DVDict["depth0"]
    xM = solverOptions["appendageList"][1]["xMount"]
    xR = solverOptions["appendageList"][iComp]["xMount"]
    ℓᵣ = xR - xM # distance from main wing AC to downstream wing AC, +ve downstream
    upstreamDict = DVDictList[1]
    sWing = upstreamDict["s"]
    cRefWing = sum(upstreamDict["c"]) / length(upstreamDict["c"])
    chordMMean = cRefWing
    # @ignore_derivatives() do
    #     if solverOptions["debug"]
    #         println(@sprintf("=========================================================================="))
    #         println(@sprintf("Computing downstream flow effects with ℓᵣ = %.2f m, C_L_M = %.1f ", ℓᵣ, CLMain))
    #     end
    # end

    # --- Compute the wake effect ---
    αiWake = compute_wakeDWAng(sWing, cRefWing, CLMain, ℓᵣ)

    # --- Compute the wave pattern effect ---
    Fnc = Uinf / (√(GRAV * chordMMean))
    Fnh = Uinf / (√(GRAV * depth))
    αiWave = compute_wavePatternDWAng(CLMain, chordMMean, Fnc, Fnh, ℓᵣ)

    # --- Correct the downwash ---
    alphaCorrection = αiWake .+ αiWave

    return alphaCorrection
end

function compute_wakeDWAng(sWing, cRefWing, CLWing, ℓᵣ)
    """
    Assume the downstream lifting surface is behind an elliptically loaded wing
    ℓᵣ = xM + xR downstream

    Inputs
    ------
    sWing: float
        span of the upstream wing
    cRefWing: float
        ref chord of the upstream wing
    CLWing: float
        Lift coefficient of the upstream wing
    ℓᵣ: float
        Distance from main wing AC to downstream wing AC, +ve downstream
    """

    ARwing = sWing / cRefWing

    l_div_s = ℓᵣ / sWing
    # THIS IS WRONG
    kappa = 1 + 1 / (√(1 + (l_div_s)^2)) * (1 / (π * l_div_s) + 1)
    kappa = 2.0
    # println("k is", k)
    k = 1 / √(1 + (l_div_s)^2)

    Ek = SpecialFunctions.ellipe(k^2)

    kappa = 1 + 2 / π * Ek / √(1 - k^2)

    ε = kappa * CLWing / (π * ARwing)

    return -ε
end

function compute_wavePatternDWAng(clM, chordM, Fnc, Fnh, ξ)
    """
    Compute 2D wave pattern effect (transverse grav waves only)

    Inputs
    ------
    clM: vector
        Spanwise lift coefficient of the main wing
    chordM: vector
        Ref chord of the main wing
    Fnc: float
        Chord-based Froude number of the main wing
    Fnh: float
        Depth-based Froude number of the main wing
    ξ - Distance from the main wing to the downstream wing
    """

    divFncsq = 1 / (Fnc * Fnc)
    premult = divFncsq * exp(-2 / (Fnh * Fnh))

    # Vectorized computation
    αiwave = -clM .* cos.(divFncsq .* ξ ./ chordM) .* premult

    return αiwave
end

function compute_LL_ventilated(semispan, submergedDepth, α₀, cl_α_FW)
    """
    Slope of the 3D lift coefficient with respect to the angle of attack considering surface-piercing vertical strut
    From Harwood 2019 Part 1

    a0 = π/2 * (1 - √(1 - (2 * submergedDepth / semispan)^2))
    """
    # TODO: get Lc from Casey's paper
    Lc_c = Lc / c

    a0 = ((π / 2) * (Lc_c^3) - 2 * (Lc_c^2) + 4.5 * Lc_c + 1) / ((Lc_c^3) - (Lc_c^2) + 0.75 * Lc_c + 1 / (2π))
    return a0
end

# ==============================================================================
#                         Static drag
# ==============================================================================
function compute_AICs(
    AEROMESH, FOIL, LLSystem, LLOutputs, ϱ, dim, Λ, U∞, ω, elemType="BT2";
    appendageOptions=Dict{String,Any}("config" => "wing"), STRUT=nothing,
    solverOptions=Dict(),
)
    """
    Compute the AIC matrix for a given aeroMesh using LHS convention
        (i.e., -ve force is disturbing, not restoring)
    Inputs
    ------
    aeroMesh: Array
    Mesh of the foil (same as struct)
    stripVecs: 2d Array
    Spanwise tangent vectors for each strip
    FOIL: struct
        Struct containing the foil implicit constants
    elemType: String
        Element type
        
    Returns
    -------
    AIC: Matrix
        Aerodynamic influence coefficient matrix broken up into added mass, damping, and stiffness
        in such a way that
            {F} = -([Mf]{udd} + [Cf]{ud} + [Kf]{u})
        These are matrices
        in the global reference frame
    NOTE: Do not actually call these AIC when talking to other people because this is technically incorrect.
    AIC is the A matrix in potential flow
    """

    # Spline to get lift slope in the right spots if using nonlinear LL
    clαVec = LLOutputs.cla

    globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = build_fluidMat(AEROMESH, FOIL, LLSystem, clαVec, ϱ, dim, Λ, U∞, ω, elemType; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)

    return globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i
end

function build_fluidMat(AEROMESH, FOIL, LLSystem, clαVec, ϱ, dim, Λ, U∞, ω, elemType="BT2";
    appendageOptions=Dict{String,Any}("config" => "wing"), STRUT=nothing,
    solverOptions=Dict())
    """
    Complex step will not work on this routine because we need the real and imag part for unsteady hydro
    # TODO: use the nVec to grab sweep and dihedral effects, then use the external Lambda as inflow angle change

    """
    # --- Initialize global matrices ---
    globalMf_z = Zygote.Buffer(zeros(dim, dim))
    globalCf_r_z = Zygote.Buffer(zeros(dim, dim))
    globalCf_i_z = Zygote.Buffer(zeros(dim, dim))
    globalKf_r_z = Zygote.Buffer(zeros(dim, dim))
    globalKf_i_z = Zygote.Buffer(zeros(dim, dim))
    # Zygote initialization
    globalMf_z[:, :] = zeros(dim, dim)
    globalCf_r_z[:, :] = zeros(dim, dim)
    globalCf_i_z[:, :] = zeros(dim, dim)
    globalKf_r_z[:, :] = zeros(dim, dim)
    globalKf_i_z[:, :] = zeros(dim, dim)

    chordVec = FOIL.chord
    abVec = FOIL.ab
    ebVec = FOIL.eb

    nElemWing = length(chordVec) - 1

    if STRUT != nothing
        strutChordVec = STRUT.chord
        strutabVec = STRUT.ab
        strutebVec = STRUT.eb
        nElemStrut = length(strutChordVec) - 1
        appendageOptions["nNodeStrut"] = nElemStrut + 1
    end
    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    stripVecs = get_strip_vecs(AEROMESH, appendageOptions)
    aeroMesh = AEROMESH.mesh
    junctionNodeX = aeroMesh[1, :]

    for (inode, XN) in enumerate(eachrow(aeroMesh)) # loop aero strips (located at FEM nodes)
        # --- compute strip quantities ---
        yⁿ = XN[YDIM]
        zⁿ = XN[ZDIM]

        nVec = stripVecs[inode, :]

        lᵉ = √(nVec[XDIM]^2 + nVec[YDIM]^2 + nVec[ZDIM]^2) # length of elem
        Δy = lᵉ

        # If we have end point nodes, we need to divide the strip width by 2
        if appendageOptions["config"] == "wing"
            if inode == 1 || inode == FOIL.nNodes
                Δy = 0.5 * lᵉ
            end
        elseif appendageOptions["config"] == "full-wing"
            if inode == FOIL.nNodes || (inode == nElemWing * 2 + 1)
                Δy = 0.5 * lᵉ
            end
        elseif appendageOptions["config"] == "t-foil"
            if inode == FOIL.nNodes || (inode == nElemWing * 2 + 1) || (inode == nElemWing * 2 + nElemStrut + 1)
                Δy = 0.5 * lᵉ
            end
        elseif !(appendageOptions["config"] in CONFIGS)
            error("Invalid configuration")
        end

        nVec = nVec / lᵉ # normalize


        # --- Linearly interpolate values based on y loc ---
        clα, localchord, ab, eb, dR1, dR2, dR3 = compute_stripValues(nVec, LLSystem, clαVec, yⁿ, FOIL, chordVec, abVec, ebVec, aeroMesh, appendageOptions, inode, solverOptions["use_nlll"])
        # # THis chunk of code is super hacky based on assuming wing and t-foil strut order
        # if use_nlll # TODO: FIX LATER TO BE GENERAL
        #     xeval = LLSystem.collocationPts[YDIM, :]
        #     clα = Interpolation.do_linear_interp(xeval, clαVec, yⁿ)
        #     sDomFoil = aeroMesh[1:FOIL.nNodes, YDIM]
        #     if inode <= FOIL.nNodes # STBD WING
        #         c = Interpolation.do_linear_interp(sDomFoil, chordVec, yⁿ)
        #         ab = Interpolation.do_linear_interp(sDomFoil, abVec, yⁿ)
        #         eb = Interpolation.do_linear_interp(sDomFoil, ebVec, yⁿ)
        #         # c = chordVec[inode]
        #         # ab = abVec[inode]
        #         # eb = ebVec[inode]

        #     else
        #         if appendageOptions["config"] in ["t-foil", "full-wing"]
        #             if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
        #                 # Put negative sign on the linear interp routine bc there is a bug!
        #                 sDomFoil = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])

        #                 c = Interpolation.do_linear_interp(sDomFoil, chordVec, -yⁿ)
        #                 ab = Interpolation.do_linear_interp(sDomFoil, abVec, -yⁿ)
        #                 eb = Interpolation.do_linear_interp(sDomFoil, ebVec, -yⁿ)
        #                 # c = chordVec[inode-FOIL.nNodes+1]
        #                 # ab = abVec[inode-FOIL.nNodes+1]
        #                 # eb = ebVec[inode-FOIL.nNodes+1]
        #                 # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
        #                 dR1 = -dR1
        #                 dR2 = -dR2
        #                 dR3 = -dR3
        #             else # strut section
        #                 sDomFoil = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
        #                 c = Interpolation.do_linear_interp(sDomFoil, strutChordVec, zⁿ)
        #                 ab = Interpolation.do_linear_interp(sDomFoil, strutabVec, zⁿ)
        #                 eb = Interpolation.do_linear_interp(sDomFoil, strutebVec, zⁿ)
        #             end
        #         end
        #     end
        #     # println("clα: ", @sprintf("%.4f", clα), "\teb: ", @sprintf("%.4f",eb), "\tyn: $(yⁿ)")
        # else
        #     # if inode <= FOIL.nNodes # STBD WING
        #     #     sDom = aeroMesh[1:FOIL.nNodes, YDIM]
        #     #     clα = Interpolation.do_linear_interp(sDom, clαVec, yⁿ)
        #     #     c = Interpolation.do_linear_interp(sDom, chordVec, yⁿ)
        #     #     ab = Interpolation.do_linear_interp(sDom, abVec, yⁿ)
        #     #     eb = Interpolation.do_linear_interp(sDom, ebVec, yⁿ)
        #     # else
        #     #     if appendageOptions["config"] == "t-foil"
        #     #         if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
        #     #             # Put negative sign on the linear interp routine bc there is a bug!
        #     #             sDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
        #     #             yⁿ = -1 * yⁿ

        #     #             clα = Interpolation.do_linear_interp(sDom, clαVec, yⁿ)
        #     #             c = Interpolation.do_linear_interp(sDom, chordVec, yⁿ)
        #     #             ab = Interpolation.do_linear_interp(sDom, abVec, yⁿ)
        #     #             eb = Interpolation.do_linear_interp(sDom, ebVec, yⁿ)
        #     #             # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
        #     #             dR1 = -dR1
        #     #             dR2 = -dR2
        #     #             dR3 = -dR3
        #     #             # println("I'm a port wing strip")
        #     #         else
        #     #             sDom = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
        #     #             clα = Interpolation.do_linear_interp(sDom, strutclαVec, zⁿ)
        #     #             c = Interpolation.do_linear_interp(sDom, strutChordVec, zⁿ)
        #     #             ab = Interpolation.do_linear_interp(sDom, strutabVec, zⁿ)
        #     #             eb = Interpolation.do_linear_interp(sDom, strutebVec, zⁿ)
        #     #             # println("I'm a strut strip")
        #     #         end
        #     #     elseif appendageOptions["config"] == "full-wing"
        #     #         if inode <= nElemWing * 2 + 1
        #     #             # Put negative sign on the linear interp routine bc there is a bug!
        #     #             sDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
        #     #             yⁿ = -1 * yⁿ

        #     #             clα = Interpolation.do_linear_interp(sDom, clαVec, yⁿ)
        #     #             c = Interpolation.do_linear_interp(sDom, chordVec, yⁿ)
        #     #             ab = Interpolation.do_linear_interp(sDom, abVec, yⁿ)
        #     #             eb = Interpolation.do_linear_interp(sDom, ebVec, yⁿ)
        #     #             # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
        #     #             dR1 = -dR1
        #     #             dR2 = -dR2
        #     #             dR3 = -dR3
        #     #         end
        #     #     end
        #     # end
        # end
        b = 0.5 * localchord # semichord for more readable code

        # --- Precomputes ---
        clambda = cos(Λ)
        slambda = sin(Λ)
        k = ω * b / (U∞ * clambda) # local reduced frequency
        CKVec = compute_theodorsen(k)
        Ck = CKVec[1] + 1im * CKVec[2]

        # --- Compute Compute local AIC matrix for this element ---
        K_f, K̂_f = compute_node_stiff_faster(clα, b, eb, ab, U∞, clambda, slambda, ϱ, Ck)
        C_f, Ĉ_f = compute_node_damp_faster(clα, b, eb, ab, U∞, clambda, slambda, ϱ, Ck)
        M_f = compute_node_mass(b, ab, ϱ)
        p_i = 1.0
        if solverOptions["use_freeSurface"]
            if appendageOptions["config"] == "full-wing"
                localspan = 0.5 * LLSystem.span - abs_cs_safe(yⁿ)
            end
            p_i = compute_pFactor(localchord, localspan)
            # println("local span:\t $(localspan)\nyn:\t$(yⁿ)\np factor:\t$(p_i)")
        end
        M_f *= p_i
        KLocal, CLocal, MLocal = compute_localAIC(K_f, K̂_f, C_f, Ĉ_f, M_f, elemType)

        # ---------------------------
        #   Transformation of AIC
        # ---------------------------
        # Aerodynamics need to happen in global reference frame
        # Γ = SolverRoutines.get_transMat(dR1, dR2, dR3, 1.0, elemType) # yes
        Γ = Rotations.get_transMat(dR1, dR2, dR3, 1.0)
        ΓT = transpose(Γ)
        KLocal_trans = ΓT[1:NDOF, 1:NDOF] * KLocal * Γ[1:NDOF, 1:NDOF]
        CLocal_trans = ΓT[1:NDOF, 1:NDOF] * CLocal * Γ[1:NDOF, 1:NDOF]
        MLocal_trans = ΓT[1:NDOF, 1:NDOF] * MLocal * Γ[1:NDOF, 1:NDOF]

        GDOFIdx::Int64 = NDOF * (inode - 1) + 1

        # Add local AIC to global AIC and remember to multiply by strip width to get the right result
        globalKf_r_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = real(KLocal_trans) * Δy
        globalKf_i_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = imag(KLocal_trans) * Δy
        globalCf_r_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = real(CLocal_trans) * Δy
        globalCf_i_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = imag(CLocal_trans) * Δy
        globalMf_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = MLocal_trans * Δy

        # inode += 1 # increment strip counter
    end

    return copy(globalMf_z), copy(globalCf_r_z), copy(globalCf_i_z), copy(globalKf_r_z), copy(globalKf_i_z)
end

function compute_stripValues(nVec, LLSystem, clαVec, yⁿ, FOIL, chordVec, abVec, ebVec, aeroMesh, appendageOptions, inode, use_nlll)
    """
    This chunk of code is super hacky based on assuming wing and t-foil strut order
    TODO: FIX LATER TO BE GENERAL
    """
    dR1 = nVec[XDIM]
    dR2 = nVec[YDIM]
    dR3 = nVec[ZDIM]
    nElemWing = length(chordVec) - 1

    if use_nlll
        xeval = LLSystem.collocationPts[YDIM, :]
        clα = Interpolation.do_linear_interp(xeval, clαVec, yⁿ)
        sDomFoil = aeroMesh[1:FOIL.nNodes, YDIM]
        if inode <= FOIL.nNodes # STBD WING
            c = chordVec[inode]
            ab = abVec[inode]
            eb = ebVec[inode]

        else
            if appendageOptions["config"] in ["t-foil", "full-wing"]
                if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                    c = chordVec[inode-FOIL.nNodes+1]
                    ab = abVec[inode-FOIL.nNodes+1]
                    eb = ebVec[inode-FOIL.nNodes+1]
                    # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                    dR1 = -dR1
                    dR2 = -dR2
                    dR3 = -dR3
                else # strut section
                    sDomFoil = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
                    c = Interpolation.do_linear_interp(sDomFoil, strutChordVec, zⁿ)
                    ab = Interpolation.do_linear_interp(sDomFoil, strutabVec, zⁿ)
                    eb = Interpolation.do_linear_interp(sDomFoil, strutebVec, zⁿ)
                end
            end
        end
    else
        if inode <= FOIL.nNodes # STBD WING
            sDom = aeroMesh[1:FOIL.nNodes, YDIM]
            clα = Interpolation.do_linear_interp(sDom, clαVec, yⁿ)
            c = Interpolation.do_linear_interp(sDom, chordVec, yⁿ)
            ab = Interpolation.do_linear_interp(sDom, abVec, yⁿ)
            eb = Interpolation.do_linear_interp(sDom, ebVec, yⁿ)
        else
            if appendageOptions["config"] == "t-foil"
                if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                    # Put negative sign on the linear interp routine bc there is a bug!
                    sDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
                    yⁿ = -1 * yⁿ

                    clα = Interpolation.do_linear_interp(sDom, clαVec, yⁿ)
                    c = Interpolation.do_linear_interp(sDom, chordVec, yⁿ)
                    ab = Interpolation.do_linear_interp(sDom, abVec, yⁿ)
                    eb = Interpolation.do_linear_interp(sDom, ebVec, yⁿ)
                    # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                    dR1 = -dR1
                    dR2 = -dR2
                    dR3 = -dR3
                    # println("I'm a port wing strip")
                else
                    sDom = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
                    clα = Interpolation.do_linear_interp(sDom, strutclαVec, zⁿ)
                    c = Interpolation.do_linear_interp(sDom, strutChordVec, zⁿ)
                    ab = Interpolation.do_linear_interp(sDom, strutabVec, zⁿ)
                    eb = Interpolation.do_linear_interp(sDom, strutebVec, zⁿ)
                    # println("I'm a strut strip")
                end
            elseif appendageOptions["config"] == "full-wing"
                if inode <= nElemWing * 2 + 1
                    # Put negative sign on the linear interp routine bc there is a bug!
                    sDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
                    yⁿ = -1 * yⁿ

                    clα = Interpolation.do_linear_interp(sDom, clαVec, yⁿ)
                    c = Interpolation.do_linear_interp(sDom, chordVec, yⁿ)
                    ab = Interpolation.do_linear_interp(sDom, abVec, yⁿ)
                    eb = Interpolation.do_linear_interp(sDom, ebVec, yⁿ)
                    # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                    dR1 = -dR1
                    dR2 = -dR2
                    dR3 = -dR3
                end
            end
        end

    end

    return clα, c, ab, eb, dR1, dR2, dR3
end

function compute_localAIC(K_f, K̂_f, C_f, Ĉ_f, M_f, elemType::String)

    if elemType == "bend-twist"
        println("These aerodynamics are all wrong BTW...")
        KLocal = -1 * [
            0.00000000 0.0 K_f[1, 2] # Lift
            0.00000000 0.0 0.00000000
            0.00000000 0.0 K_f[2, 2] # Pitching moment
        ]
    elseif elemType == "BT2"
        KLocal = [
            0.0 K̂_f[1, 1] K_f[1, 2] K̂_f[1, 2]  # Lift
            0.0 0.0 0.0 0.0
            0.0 K̂_f[2, 1] K_f[2, 2] K̂_f[2, 2] # Pitching moment
            0.0 0.0 0.0 0.0
        ]
        CLocal = [
            C_f[1, 1] Ĉ_f[1, 1] C_f[1, 2] Ĉ_f[1, 2]  # Lift
            0.0 0.0 0.0 0.0
            C_f[2, 1] Ĉ_f[2, 1] C_f[2, 2] Ĉ_f[2, 2] # Pitching moment
            0.0 0.0 0.0 0.0
        ]
        MLocal = [
            M_f[1, 1] 0.0 M_f[1, 2] 0.0  # Lift
            0.0 0.0 0.0 0.0
            M_f[2, 1] 0.0 M_f[2, 2] 0.0 # Pitching moment
            0.0 0.0 0.0 0.0
        ]
    elseif elemType == "COMP2"
        # # NOTE: Done in local aero coordinates which matches the local beam coordinates
        KLocal = [
            # u v   w         phi       theta     psi phi'     theta'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # u
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # v
            0.0 0.0 0.0000000 K_f[1, 2] K̂_f[1, 1] 0.0 K̂_f[1, 2] 0.0 0.0  # w
            0.0 0.0 0.0000000 K_f[2, 2] K̂_f[2, 1] 0.0 K̂_f[2, 2] 0.0 0.0 # phi
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # phi'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi'
        ]
        CLocal = [
            # u v   w         phi       theta     psi phi'     theta'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # u
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # v
            0.0 0.0 C_f[1, 1] C_f[1, 2] Ĉ_f[1, 1] 0.0 Ĉ_f[1, 2] 0.0 0.0  # w
            0.0 0.0 C_f[2, 1] C_f[2, 2] Ĉ_f[2, 1] 0.0 Ĉ_f[2, 2] 0.0 0.0 # phi
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # phi'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # theta'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0000000 0.0 0.0 # psi'
        ]
        MLocal = [
            # u v   w         phi       theta     psi phi'     theta'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # u
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # v
            0.0 0.0 M_f[1, 1] M_f[1, 2] 0.0000000 0.0 0.0 0.0000000 0.0 # w
            0.0 0.0 M_f[2, 1] M_f[2, 2] 0.0000000 0.0 0.0 0.0000000 0.0 # phi
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # theta
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # psi
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # phi'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # theta'
            0.0 0.0 0.0000000 0.0000000 0.0000000 0.0 0.0 0.0000000 0.0 # psi'
        ]
    else
        println("nothing else works")
    end

    return KLocal, CLocal, MLocal
end

function compute_areas(AEROMESH, FOIL;
    appendageOptions=Dict{String,Any}("config" => "wing"), STRUT=nothing)
    """
    Planform area ref (and WSA?) for nondimensionalization
    """

    # --- Initialize planform area counter ---
    planformArea = 0.0
    chordVec = FOIL.chord

    if STRUT != nothing
        strutChordVec = STRUT.chord
    end

    jj = 1 # node index
    nElemWing = length(chordVec) - 1
    # Bit circular logic here
    appendageOptions["nNodes"] = nElemWing + 1
    if STRUT != nothing
        nElemStrut = length(strutChordVec) - 1
        appendageOptions["nNodeStrut"] = nElemStrut + 1
    end
    # ---------------------------
    #   Loop over strips (nodes)
    # ---------------------------
    stripVecs = get_strip_vecs(AEROMESH, appendageOptions)
    aeroMesh = AEROMESH.mesh
    junctionNodeX = aeroMesh[1, :]

    for (inode, XN) in enumerate(eachrow(aeroMesh)) # loop aero strips (located at FEM nodes)
        # --- compute strip quantities ---
        yⁿ = XN[YDIM]
        zⁿ = XN[ZDIM]


        nVec = stripVecs[inode, :]

        lᵉ = √(nVec[XDIM]^2 + nVec[YDIM]^2 + nVec[ZDIM]^2) # length of elem
        Δy = lᵉ

        # If we have end point nodes, we need to divide the strip width by 2
        if appendageOptions["config"] == "wing"
            if inode == 1 || inode == FOIL.nNodes
                Δy = 0.5 * lᵉ
            end
        elseif appendageOptions["config"] == "full-wing"
            if inode == FOIL.nNodes || (inode == nElemWing * 2 + 1)
                Δy = 0.5 * lᵉ
            end
        elseif appendageOptions["config"] == "t-foil"
            if inode == 1 || inode == FOIL.nNodes || (inode == nElemWing * 2 + 1) || (inode == nElemWing * 2 + nElemStrut + 1)
                Δy = 0.5 * lᵉ
            end
        elseif !(appendageOptions["config"] in CONFIGS)
            error("Invalid configuration")
        end

        nVec = nVec / lᵉ # normalize
        dR1 = nVec[XDIM]
        dR2 = nVec[YDIM]
        dR3 = nVec[ZDIM]

        # --- Linearly interpolate values based on y loc ---
        # THis chunk of code is super hacky based on assuming wing and t-foil strut order
        sDomFoil = aeroMesh[1:FOIL.nNodes, YDIM]
        if inode <= FOIL.nNodes # STBD WING
            # c = Interpolation.do_linear_interp(sDomFoil, chordVec, yⁿ)
            c = chordVec[inode]
        else
            if appendageOptions["config"] in ["t-foil", "full-wing"]
                if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                    # Put negative sign on the linear interp routine bc there is a bug!
                    # sDomFoil = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])

                    # c = Interpolation.do_linear_interp(sDomFoil, chordVec, -yⁿ)
                    c = chordVec[inode-FOIL.nNodes+1] # same
                    # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                    dR1 = -dR1
                    dR2 = -dR2
                    dR3 = -dR3
                else # strut section
                    c = 0.0
                end
            end
        end

        # Add rectangle to planform area
        planformArea += c * Δy
    end

    areaRef = planformArea
    # println("Planform area: $(areaRef) m^2")

    return areaRef
end


function compute_∂Kff∂cla(AEROMESH, FOIL, STRUT, dim, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="FiDi")

    # ************************************************
    #     Setup
    # ************************************************
    LLOutputs, LLSystem, FlowCond = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)
    clαVec = copy(LLOutputs.cla)
    n_cla = length(LLOutputs.cla)
    Λ = LLSystem.sweepAng

    # ************************************************
    #     Computation
    # ************************************************
    dKffdcla = zeros(dim^2, n_cla)

    if uppercase(mode) == "FIDI"
        dh = 1e-6

        _, _, _, AIC_i, _ = build_fluidMat(AEROMESH, FOIL, LLSystem, clαVec, FlowCond.rhof, dim, Λ, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)
        Kff_i = vec(-AIC_i)

        for icla in 1:n_cla
            clαVec[icla] += dh

            _, _, _, AIC_f, _ = build_fluidMat(AEROMESH, FOIL, LLSystem, clαVec, FlowCond.rhof, dim, Λ, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)
            Kff_f = vec(-AIC_f)

            clαVec[icla] -= dh

            dKffdcla[:, icla] = (Kff_f - Kff_i) / dh
        end

    end

    return dKffdcla
end

function compute_∂Kff∂Xpt(dim, ptVec, nodeConn, appendageOptions, appendageParams, solverOptions; mode="FiDi")

    # ************************************************
    #     Setup
    # ************************************************
    LLOutputs, LLSystem, FlowCond = compute_cla_API(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; return_all=true)

    LECoords, _ = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
    idxTip = Preprocessing.get_tipnode(LECoords)

    dKffdXpt = zeros(dim^2, length(ptVec))

    if uppercase(mode) == "FIDI"
        dh = 1e-6

        LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
        midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, nodeConn, idxTip, appendageParams, appendageOptions)
        if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
            print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

            α₀ = appendageParams["alfa0"]
            # sweepAng = appendageParams["sweep"]
            rake = appendageParams["rake"]
            # span = appendageParams["s"] * 2
            zeta = appendageParams["zeta"]
            theta_f = appendageParams["theta_f"]
            beta = appendageParams["beta"]
            s_strut = appendageParams["s_strut"]
            c_strut = appendageParams["c_strut"]
            theta_f_strut = appendageParams["theta_f_strut"]
            depth0 = appendageParams["depth0"]

            toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
        else
            α₀ = appendageParams["alfa0"]
            # sweepAng = appendageParams["sweep"]
            rake = appendageParams["rake"]
            # span = appendageParams["s"] * 2
            toc::Vector{RealOrComplex} = appendageParams["toc"]
            ab::Vector{RealOrComplex} = appendageParams["ab"]
            x_ab::Vector{RealOrComplex} = appendageParams["x_ab"]
            zeta = appendageParams["zeta"]
            theta_f = appendageParams["theta_f"]
            beta = appendageParams["beta"]
            s_strut = appendageParams["s_strut"]
            c_strut = appendageParams["c_strut"]
            toc_strut = appendageParams["toc_strut"]
            ab_strut = appendageParams["ab_strut"]
            x_ab_strut = appendageParams["x_ab_strut"]
            theta_f_strut = appendageParams["theta_f_strut"]
            depth0 = appendageParams["depth0"]
        end
        AEROMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))
        FOIL, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)
        # Λ = sweepAng

        _, _, _, AIC_i, _ = compute_AICs(AEROMESH, FOIL, LLSystem, LLOutputs, FlowCond.rhof, dim, Λ, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)
        Kff_i = vec(-AIC_i)

        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            LECoords, TECoords = Utilities.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
            midchords, chordLengths, spanwiseVectors, Λ, pretwistDist = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

            structMesh, elemConn = FEMMethods.make_FEMeshFromCoords(midchords, nodeConn, idxTip, appendageParams, appendageOptions)
            if haskey(appendageOptions, "path_to_geom_props") && !isnothing(appendageOptions["path_to_geom_props"])
                print("Reading geometry properties from file: ", appendageOptions["path_to_geom_props"])

                α₀ = appendageParams["alfa0"]
                # sweepAng = appendageParams["sweep"]
                rake = appendageParams["rake"]
                # span = appendageParams["s"] * 2
                zeta = appendageParams["zeta"]
                theta_f = appendageParams["theta_f"]
                beta = appendageParams["beta"]
                s_strut = appendageParams["s_strut"]
                c_strut = appendageParams["c_strut"]
                theta_f_strut = appendageParams["theta_f_strut"]
                depth0 = appendageParams["depth0"]

                toc, ab, x_ab, toc_strut, ab_strut, x_ab_strut = Preprocessing.get_1DGeoPropertiesFromFile(appendageOptions["path_to_geom_props"])
            else
                α₀ = appendageParams["alfa0"]
                # sweepAng = appendageParams["sweep"]
                rake = appendageParams["rake"]
                # span = appendageParams["s"] * 2
                toc = appendageParams["toc"]
                ab = appendageParams["ab"]
                x_ab = appendageParams["x_ab"]
                zeta = appendageParams["zeta"]
                theta_f = appendageParams["theta_f"]
                beta = appendageParams["beta"]
                s_strut = appendageParams["s_strut"]
                c_strut = appendageParams["c_strut"]
                toc_strut = appendageParams["toc_strut"]
                ab_strut = appendageParams["ab_strut"]
                x_ab_strut = appendageParams["x_ab_strut"]
                theta_f_strut = appendageParams["theta_f_strut"]
                depth0 = appendageParams["depth0"]
            end
            AEROMESH = FEMMethods.StructMesh(structMesh, elemConn, chordLengths, toc, ab, x_ab, theta_f, idxTip, zeros(10, 2))
            FOIL, STRUT = FEMMethods.init_staticStruct(LECoords, TECoords, nodeConn, toc, ab, theta_f, toc_strut, ab_strut, theta_f_strut, appendageParams, appendageOptions, solverOptions)

            _, _, _, AIC_f, _ = compute_AICs(AEROMESH, FOIL, LLSystem, LLOutputs, FlowCond.rhof, dim, Λ, FlowCond.Uinf, 0.0, ELEMTYPE; appendageOptions=appendageOptions, STRUT=STRUT, solverOptions=solverOptions)
            Kff_f = vec(-AIC_f)

            dKffdXpt[:, ii] = (Kff_f - Kff_i) / dh

            ptVec[ii] -= dh

        end
    end

    return dKffdXpt
end


function get_strip_vecs(
    AEROMESH, solverOptions
)
    """

    Compute the spanwise tangent vectors for each strip
        Strips are on the mesh nodes

    Parameters
    ----------
    aeroMesh: array
        The aerodynamic mesh
    elemConn: array
        The element connectivity from the structural mesh
    """
    aeroMesh = AEROMESH.mesh
    elemConn = AEROMESH.elemConn

    nStrips = size(aeroMesh)[1]

    stripVecs = zeros(RealOrComplex, nStrips, 3)
    stripVecs_z = Zygote.Buffer(stripVecs)
    stripVecs_z[:, :] = stripVecs
    nElemWing = solverOptions["nNodes"] - 1
    if haskey(solverOptions, "nNodeStrut")
        nElemStrut = solverOptions["nNodeStrut"] - 1
    else
        nElemStrut = 0
    end
    for istrip in 1:nStrips-1 # loop elements but these are filling aero strips (nodes)
        n1 = elemConn[istrip, 1]
        n2 = elemConn[istrip, 2]
        if istrip > 1 # apply tangency continuation for wing and strut
            if istrip == nElemWing + 1 || istrip == nElemWing * 2 + 1 || istrip == nElemWing * 2 + nElemStrut + 1
                n1 = elemConn[istrip-1, 1]
                n2 = elemConn[istrip-1, 2]
            end
        end

        nvec = aeroMesh[n2, :] - aeroMesh[n1, :]
        stripVecs_z[istrip, :] = nvec

    end

    # Treat last strip separately
    if size(elemConn)[1] == 3 || size(elemConn)[1] == 2
        stripVecs_z[end, :] = aeroMesh[end, :] - aeroMesh[1, :]
    else
        stripVecs_z[end, :] = aeroMesh[end, :] - aeroMesh[end-1, :]
    end

    return copy(stripVecs_z)
end

function compute_genHydroLoadsMatrices(kMax, nk, U∞, b_ref, dim, AEROMESH, Λ, FOIL, LLSystem, LLOutputs, rhof, elemType; appendageOptions, solverOptions)
    """
    Computes the hydrodynamic coefficients for a sweep of reduced frequencies

    Inputs
    ------
        kMax: maximum reduced frequency
        nk: number of reduced frequencies
        U∞: freestream velocity
    """

    linDist = ((LinRange(1, nk, nk)) .- 1) / (nk - 1)
    cubicDist = linDist[2:end] .^ 3 # cubic distribution removing the first point
    kSweep = vcat([1e-13], kMax .* cubicDist)

    # --- Loop over reduced frequencies ---
    Cf_r_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Cf_i_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Kf_r_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Kf_i_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    Mf_sweep_z = Zygote.Buffer(zeros(dim, dim, nk))
    ii = 1
    for k in kSweep
        ω = k * U∞ * (cos(Λ)) / b_ref

        # Compute AIC
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(AEROMESH, FOIL, LLSystem, LLOutputs, rhof, dim, Λ, U∞, ω, elemType; appendageOptions=appendageOptions, solverOptions=solverOptions)

        # Accumulate in frequency sweep matrix
        # @inbounds begin
        Cf_r_sweep_z[:, :, ii] = globalCf_r
        Cf_i_sweep_z[:, :, ii] = globalCf_i
        Kf_r_sweep_z[:, :, ii] = globalKf_r
        Kf_i_sweep_z[:, :, ii] = globalKf_i
        Mf_sweep_z[:, :, ii] = globalMf
        # end
        ii += 1
    end

    return copy(Mf_sweep_z[:, :, 1]), copy(Cf_r_sweep_z), copy(Cf_i_sweep_z), copy(Kf_r_sweep_z), copy(Kf_i_sweep_z), kSweep
end

function interpolate_influenceCoeffs(k, k_sweep, Ar_sweep_r, Ar_sweep_i, Nmr::Int, pkEqnType)
    """
    This function interpolates the influence coefficient matrix for specific reduced frequency
    Here a linear interpolation is applied

    Inputs
    ------
        k_sweep - vector of reduced frequency values that the AIC was generated at. Array(Nk)
        Ar_sweep_r, Ar_sweep_i - The AIC (real/imag parts) evaluated at reduced frequencies in k_sweep. Array(Nmr,Nmr,Nk)
        Nmr - the size of reduced problem

    Outputs
    -------
        Ar_r, Ar_i - The interpolated AIC (real/imag parts) evaluated at reduced frequency k. Array(Nmr,Nmr)
    """

    # Calc the difference of current k and the k_sweep
    kDiff = k .- k_sweep

    # Find where the sign changes for the interpolation
    b1, b2 = SolverRoutines.find_signChange(ChainRulesCore.ignore_derivatives(kDiff))
    # Based on the pk equation we are solving it depends how we handle k=0
    if (pkEqnType == "hassig" || pkEqnType == "ng")
        # Use the lagrange interpolation (L for linear)
        x0L::Vector{DTYPE} = vcat(k_sweep[b1], k_sweep[b2]) # somehow this line is adding a lot to the time

        # This unravels to keep computations in the stack, not the heap

        y0L::Array{DTYPE} = cat(Ar_sweep_r[:, :, b1], Ar_sweep_r[:, :, b2], dims=3)
        Ar_r = Interpolation.lagrangeArrInterp(ChainRulesCore.ignore_derivatives(x0L), y0L, Nmr, Nmr, 2, k)

        y0L = cat(Ar_sweep_i[:, :, b1], Ar_sweep_i[:, :, b2], dims=3)
        Ar_i = Interpolation.lagrangeArrInterp(ChainRulesCore.ignore_derivatives(x0L), y0L, Nmr, Nmr, 2, k)


    elseif (pkEqnType == "rodden") # TODO: get to work with other equation types
        if (k == 0)

            # ! Use the lagrange polynomial for quadratic extrapolation for k = 0
            x0Q[1] = k_sweep[2]
            x0Q[2] = k_sweep[3]
            x0Q[3] = k_sweep[4]

            y0Q[:, :, 1] = Ar_sweep_r[:, :, 2]
            y0Q[:, :, 2] = Ar_sweep_r[:, :, 3]
            y0Q[:, :, 3] = Ar_sweep_r[:, :, 4]
            Ar_r = lagrangeArrInterp(x0Q, y0Q, Nmr, Nmr, 3, k, Ar_r)

            y0Q[:, :, 1] = Ar_sweep_i[:, :, 2] / k_sweep[2]
            y0Q[:, :, 2] = Ar_sweep_i[:, :, 3] / k_sweep[3]
            y0Q[:, :, 3] = Ar_sweep_i[:, :, 4] / k_sweep[4]
            Ar_i = lagrangeArrInterp(x0Q, y0Q, Nmr, Nmr, 3, k, Ar_i)


        else
            # ! Use linear interpolation
            x0L[1] = k_sweep[bound[1]]
            x0L[2] = k_sweep[bound[2]]

            y0L[:, :, 1] = Ar_sweep_r[:, :, bound[1]]
            y0L[:, :, 2] = Ar_sweep_r[:, :, bound[2]]
            Ar_r = lagrangeArrInterp(x0L, y0L, Nmr, Nmr, 2, k, Ar_r)

            y0L[:, :, 1] = Ar_sweep_i[:, :, bound[1]] / k
            y0L[:, :, 2] = Ar_sweep_i[:, :, bound[2]] / k
            Ar_i = lagrangeArrInterp(x0L, y0L, Nmr, Nmr, 2, k, Ar_i)

        end
    end

    return Ar_r, Ar_i
end

function integrate_hydroLoads(
    foilStructuralStates, fullAIC, α₀, rake, dofBlank, downwashAngles::DTYPE, elemType="BT2";
    appendageOptions=Dict(), solverOptions=Dict()
)
    """
    Inputs
    ------
        fullAIC: AIC matrix which in the DCFoil code base is Kf even though it's normally -Kf
        α₀: base angle of attack
        rake: rake angle
        FOIL: FOIL struct
        elemType: element type
    Returns
    -------
        force vector
        abs val of total lift and moment (needed for dynamic mode since they are complex)
    """

    # --- Initializations ---
    # This is dynamic deflection + rigid shape of foil
    DVDict = Dict(
        "alfa0" => α₀,
        "rake" => rake,
        "beta" => 0.0,
    )
    foilTotalStates = SolverRoutines.return_totalStates(foilStructuralStates, DVDict, elemType;
        appendageOptions=appendageOptions, alphaCorrection=downwashAngles)

    # --- Strip theory ---
    # This is the hydro force traction vector
    # The problem is the full AIC matrix build (RHS). These states look good
    # fhydro RHS = -Kf * states
    ForceVector = zeros(typeof(fullAIC[1, 1]), length(foilTotalStates))
    ForceVector_z = Zygote.Buffer(ForceVector)
    ForceVector_z[:] = ForceVector

    # Only compute forces not at the blanked BC node
    Kff = fullAIC[1:end.∉[dofBlank], 1:end.∉[dofBlank]]
    utot = foilTotalStates[1:end.∉[dofBlank]]
    F = -Kff * utot
    ForceVector_z[1:length(foilTotalStates).∉[dofBlank]] = F
    ForceVector = copy(ForceVector_z)

    nDOF = BeamElement.NDOF

    if elemType == "bend-twist"
        My = ForceVector[nDOF:nDOF:end]
    elseif elemType == "BT2"
        My = ForceVector[3:nDOF:end]
        Lift = ForceVector[1:nDOF:end]
    elseif elemType == "COMP2"
        My = ForceVector[3+YDIM:nDOF:end]
        Fz = ForceVector[ZDIM:nDOF:end]
        Fy = ForceVector[YDIM:nDOF:end]
    else
        error("Invalid element type")
    end

    @ignore_derivatives() do
        if solverOptions["debug"]
            config = appendageOptions["config"]
            println("Plotting hydrodynamic loads")
            plot(1:length(Fz), Fz, label="Fz")
            plotTitle = @sprintf("alpha = %.2f, config = %s", α₀, config)
            title!(plotTitle)
            xlabel!("Strip number")
            ylabel!("Lift (N/m)")
            fname = @sprintf("./DebugOutput/hydroloads_lift.png")
            savefig(fname)

            plot(1:length(My), My, label="Moment")
            plotTitle = @sprintf("alpha = %.2f, config = %s", α₀, config)
            title!(plotTitle)
            xlabel!("Strip number")
            ylabel!("Moment (Nm/m)")
            fname = @sprintf("./DebugOutput/hydroloads_moments.png")
            savefig(fname)
        end
    end

    # --- Total dynamic hydro force calcs ---
    CmplxTotalLift = 0.0
    for secLift in Fz
        CmplxTotalLift += secLift
    end
    CmplxTotalMoment = 0.0
    for secMom in My
        CmplxTotalMoment += secMom
    end

    return ForceVector, CmplxTotalLift, CmplxTotalMoment
end

function apply_BCs(K, C, M, globalDOFBlankingList)
    """
    Applies BCs for nodal displacements
    """

    newK = K[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newM = M[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]
    newC = C[1:end.∉[globalDOFBlankingList], 1:end.∉[globalDOFBlankingList]]

    return newK, newC, newM
end


end # end module
