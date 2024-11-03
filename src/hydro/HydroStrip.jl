# --- Julia ---

# @File    :   HydroStrip.jl
# @Time    :   2022/05/18
# @Author  :   Galen Ng
# @Desc    :   Contains hydrodynamic routines and interfaces to other codes

module HydroStrip

# --- Public functions ---
export compute_theodorsen, compute_glauert_circ
export compute_node_mass, compute_node_damp, compute_node_stiff
export compute_AICs, apply_BCs

# --- PACKAGES ---
using SpecialFunctions
using LinearAlgebra
using Statistics
using Zygote, ChainRulesCore
using Printf, DelimitedFiles
using Plots
using FLOWMath: norm_cs_safe
# using SparseArrays
using Debugger

# --- DCFoil modules ---
using ..SolverRoutines
using ..Unsteady: compute_theodorsen, compute_sears, compute_node_stiff_faster, compute_node_damp_faster, compute_node_mass
using ..GlauertLL: GlauertLL
using ..LiftingLine: LiftingLine
using ..SolutionConstants: XDIM, YDIM, ZDIM, MEPSLARGE, GRAV
using ..EBBeam: EBBeam as BeamElement, NDOF
using ..DCFoil: RealOrComplex, DTYPE
using ..DesignConstants: CONFIGS

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
function compute_hydroLLProperties(span, chordVec, α₀, rake, sweepAng, depth0; solverOptions, appendageOptions)
    """
    Wrapper function to the hydrodynamic lifting line properties
    In this case, α is the angle by which the flow velocity vector is rotated, not the geometry

    Outputs
    -------
    cla: vector
        Lift slope wrt angle of attack (rad^-1) for each spanwise station
    """

    if solverOptions["use_nlll"]
        if appendageOptions["config"] == "wing"
            println("WARNING: NL LL is only for symmetric wings")
        end
        # println("Using nonlinear lifting line")

        # Hard-coded NACA0012
        airfoilCoordFile = "$(pwd())/INPUT/PROFILES/NACA0012.dat"
        airfoilX = [1.00000000e+00, 9.98993338e-01, 9.95977406e-01, 9.90964349e-01,
            9.83974351e-01, 9.75035559e-01, 9.64183967e-01, 9.51463269e-01,
            9.36924689e-01, 9.20626766e-01, 9.02635129e-01, 8.83022222e-01,
            8.61867019e-01, 8.39254706e-01, 8.15276334e-01, 7.90028455e-01,
            7.63612734e-01, 7.36135537e-01, 7.07707507e-01, 6.78443111e-01,
            6.48460188e-01, 6.17879468e-01, 5.86824089e-01, 5.55419100e-01,
            5.23790958e-01, 4.92067018e-01, 4.60375022e-01, 4.28842581e-01,
            3.97596666e-01, 3.66763093e-01, 3.36466018e-01, 3.06827437e-01,
            2.77966694e-01, 2.50000000e-01, 2.23039968e-01, 1.97195156e-01,
            1.72569633e-01, 1.49262556e-01, 1.27367775e-01, 1.06973453e-01,
            8.81617093e-02, 7.10082934e-02, 5.55822757e-02, 4.19457713e-02,
            3.01536896e-02, 2.02535132e-02, 1.22851066e-02, 6.28055566e-03,
            2.26403871e-03, 2.51728808e-04, 2.51728808e-04, 2.26403871e-03,
            6.28055566e-03, 1.22851066e-02, 2.02535132e-02, 3.01536896e-02,
            4.19457713e-02, 5.55822757e-02, 7.10082934e-02, 8.81617093e-02,
            1.06973453e-01, 1.27367775e-01, 1.49262556e-01, 1.72569633e-01,
            1.97195156e-01, 2.23039968e-01, 2.50000000e-01, 2.77966694e-01,
            3.06827437e-01, 3.36466018e-01, 3.66763093e-01, 3.97596666e-01,
            4.28842581e-01, 4.60375022e-01, 4.92067018e-01, 5.23790958e-01,
            5.55419100e-01, 5.86824089e-01, 6.17879468e-01, 6.48460188e-01,
            6.78443111e-01, 7.07707507e-01, 7.36135537e-01, 7.63612734e-01,
            7.90028455e-01, 8.15276334e-01, 8.39254706e-01, 8.61867019e-01,
            8.83022222e-01, 9.02635129e-01, 9.20626766e-01, 9.36924689e-01,
            9.51463269e-01, 9.64183967e-01, 9.75035559e-01, 9.83974351e-01,
            9.90964349e-01, 9.95977406e-01, 9.98993338e-01, 1.00000000e+00]

        airfoilY = [1.33226763e-17, -1.41200438e-04, -5.63343432e-04, -1.26208774e-03,
            -2.23030811e-03, -3.45825298e-03, -4.93375074e-03, -6.64245132e-03,
            -8.56808791e-03, -1.06927428e-02, -1.29971013e-02, -1.54606806e-02,
            -1.80620201e-02, -2.07788284e-02, -2.35880799e-02, -2.64660655e-02,
            -2.93884006e-02, -3.23300020e-02, -3.52650469e-02, -3.81669313e-02,
            -4.10082448e-02, -4.37607812e-02, -4.63956016e-02, -4.88831639e-02,
            -5.11935307e-02, -5.32966591e-02, -5.51627743e-02, -5.67628192e-02,
            -5.80689678e-02, -5.90551853e-02, -5.96978089e-02, -5.99761253e-02,
            -5.98729133e-02, -5.93749219e-02, -5.84732545e-02, -5.71636340e-02,
            -5.54465251e-02, -5.33271006e-02, -5.08150416e-02, -4.79241724e-02,
            -4.46719377e-02, -4.10787401e-02, -3.71671601e-02, -3.29610920e-02,
            -2.84848303e-02, -2.37621474e-02, -1.88154040e-02, -1.36647314e-02,
            -8.32732576e-03, -2.81688492e-03, 2.81688492e-03, 8.32732576e-03,
            1.36647314e-02, 1.88154040e-02, 2.37621474e-02, 2.84848303e-02,
            3.29610920e-02, 3.71671601e-02, 4.10787401e-02, 4.46719377e-02,
            4.79241724e-02, 5.08150416e-02, 5.33271006e-02, 5.54465251e-02,
            5.71636340e-02, 5.84732545e-02, 5.93749219e-02, 5.98729133e-02,
            5.99761253e-02, 5.96978089e-02, 5.90551853e-02, 5.80689678e-02,
            5.67628192e-02, 5.51627743e-02, 5.32966591e-02, 5.11935307e-02,
            4.88831639e-02, 4.63956016e-02, 4.37607812e-02, 4.10082448e-02,
            3.81669313e-02, 3.52650469e-02, 3.23300020e-02, 2.93884006e-02,
            2.64660655e-02, 2.35880799e-02, 2.07788284e-02, 1.80620201e-02,
            1.54606806e-02, 1.29971013e-02, 1.06927428e-02, 8.56808791e-03,
            6.64245132e-03, 4.93375074e-03, 3.45825298e-03, 2.23030811e-03,
            1.26208774e-03, 5.63343432e-04, 1.41200438e-04, -1.33226763e-17]
        airfoilCtrlX = [9.99496669e-01, 9.97485372e-01, 9.93470878e-01, 9.87469350e-01,
            9.79504955e-01, 9.69609763e-01, 9.57823618e-01, 9.44193979e-01,
            9.28775727e-01, 9.11630948e-01, 8.92828675e-01, 8.72444620e-01,
            8.50560862e-01, 8.27265520e-01, 8.02652394e-01, 7.76820594e-01,
            7.49874136e-01, 7.21921522e-01, 6.93075309e-01, 6.63451649e-01,
            6.33169828e-01, 6.02351778e-01, 5.71121594e-01, 5.39605029e-01,
            5.07928988e-01, 4.76221020e-01, 4.44608801e-01, 4.13219623e-01,
            3.82179880e-01, 3.51614556e-01, 3.21646728e-01, 2.92397065e-01,
            2.63983347e-01, 2.36519984e-01, 2.10117562e-01, 1.84882395e-01,
            1.60916095e-01, 1.38315166e-01, 1.17170614e-01, 9.75675810e-02,
            7.95850013e-02, 6.32952845e-02, 4.87640235e-02, 3.60497304e-02,
            2.52036014e-02, 1.62693099e-02, 9.28283111e-03, 4.27229719e-03,
            1.25788376e-03, 2.51728808e-04, 1.25788376e-03, 4.27229719e-03,
            9.28283111e-03, 1.62693099e-02, 2.52036014e-02, 3.60497304e-02,
            4.87640235e-02, 6.32952845e-02, 7.95850013e-02, 9.75675810e-02,
            1.17170614e-01, 1.38315166e-01, 1.60916095e-01, 1.84882395e-01,
            2.10117562e-01, 2.36519984e-01, 2.63983347e-01, 2.92397065e-01,
            3.21646728e-01, 3.51614556e-01, 3.82179880e-01, 4.13219623e-01,
            4.44608801e-01, 4.76221020e-01, 5.07928988e-01, 5.39605029e-01,
            5.71121594e-01, 6.02351778e-01, 6.33169828e-01, 6.63451649e-01,
            6.93075309e-01, 7.21921522e-01, 7.49874136e-01, 7.76820594e-01,
            8.02652394e-01, 8.27265520e-01, 8.50560862e-01, 8.72444620e-01,
            8.92828675e-01, 9.11630948e-01, 9.28775727e-01, 9.44193979e-01,
            9.57823618e-01, 9.69609763e-01, 9.79504955e-01, 9.87469350e-01,
            9.93470878e-01, 9.97485372e-01, 9.99496669e-01]
        airfoilCtrlY = [-7.06002188e-05, -3.52271935e-04, -9.12715585e-04, -1.74619792e-03,
            -2.84428054e-03, -4.19600186e-03, -5.78810103e-03, -7.60526962e-03,
            -9.63041534e-03, -1.18449221e-02, -1.42288910e-02, -1.67613504e-02,
            -1.94204243e-02, -2.21834542e-02, -2.50270727e-02, -2.79272331e-02,
            -3.08592013e-02, -3.37975244e-02, -3.67159891e-02, -3.95875880e-02,
            -4.23845130e-02, -4.50781914e-02, -4.76393828e-02, -5.00383473e-02,
            -5.22450949e-02, -5.42297167e-02, -5.59627967e-02, -5.74158935e-02,
            -5.85620766e-02, -5.93764971e-02, -5.98369671e-02, -5.99245193e-02,
            -5.96239176e-02, -5.89240882e-02, -5.78184443e-02, -5.63050796e-02,
            -5.43868129e-02, -5.20710711e-02, -4.93696070e-02, -4.62980550e-02,
            -4.28753389e-02, -3.91229501e-02, -3.50641261e-02, -3.07229612e-02,
            -2.61234889e-02, -2.12887757e-02, -1.62400677e-02, -1.09960286e-02,
            -5.57210534e-03, 0.00000000e+00, 5.57210534e-03, 1.09960286e-02,
            1.62400677e-02, 2.12887757e-02, 2.61234889e-02, 3.07229612e-02,
            3.50641261e-02, 3.91229501e-02, 4.28753389e-02, 4.62980550e-02,
            4.93696070e-02, 5.20710711e-02, 5.43868129e-02, 5.63050796e-02,
            5.78184443e-02, 5.89240882e-02, 5.96239176e-02, 5.99245193e-02,
            5.98369671e-02, 5.93764971e-02, 5.85620766e-02, 5.74158935e-02,
            5.59627967e-02, 5.42297167e-02, 5.22450949e-02, 5.00383473e-02,
            4.76393828e-02, 4.50781914e-02, 4.23845130e-02, 3.95875880e-02,
            3.67159891e-02, 3.37975244e-02, 3.08592013e-02, 2.79272331e-02,
            2.50270727e-02, 2.21834542e-02, 1.94204243e-02, 1.67613504e-02,
            1.42288910e-02, 1.18449221e-02, 9.63041534e-03, 7.60526962e-03,
            5.78810103e-03, 4.19600186e-03, 2.84428054e-03, 1.74619792e-03,
            9.12715585e-04, 3.52271935e-04, 7.06002188e-05]

        airfoilXY = copy(transpose(hcat(airfoilX, airfoilY)))
        airfoilCtrlXY = copy(transpose(hcat(airfoilCtrlX, airfoilCtrlY)))

        npt_wing = 40
        npt_airfoil = 99

        rootChord = chordVec[1]
        TR = chordVec[end] / rootChord

        # println("aoa: $(α₀)")
        Uvec = [cos(deg2rad(α₀)), 0.0, sin(deg2rad(α₀))] * solverOptions["Uinf"]

        # --- Structural span is not the same as aero span ---
        aeroSpan = span * cos(sweepAng)

        options = Dict(
            "translation" => vec([appendageOptions["xMount"], 0, 0]), # of the midchord
            "debug" => true,
        )

        LLSystem, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, aeroSpan, sweepAng, rootChord, TR;
            npt_wing=npt_wing,
            npt_airfoil=npt_airfoil,
            rhof=solverOptions["rhof"],
            # airfoilCoordFile=airfoilCoordFile,
            airfoil_ctrl_xy=airfoilCtrlXY,
            airfoil_xy=airfoilXY,
            options=options,
        )
        LLOutputs = LiftingLine.solve(FlowCond, LLSystem, LLHydro, Airfoils, AirfoilInfluences)

        Fdist = LLOutputs.Fdist
        F = LLOutputs.F
        cla = LLOutputs.cla
        CDi = LLOutputs.CDi

    else
        cla, Fxind, CDi = GlauertLL.compute_glauert_circ(0.5 * span, chordVec, deg2rad(α₀ + rake), solverOptions["Uinf"];
            h=depth0,
            useFS=solverOptions["use_freeSurface"],
            rho=solverOptions["rhof"],
            # config=foilOptions["config"], # legacy option
            debug=solverOptions["debug"]
        )
        F = [Fxind, 0.0, 0.0, 0.0, 0.0, 0.0]
        LLSystem = nothing
        FlowCond = nothing
    end

    return cla, F, CDi, LLSystem, FlowCond
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
    # ChainRulesCore.ignore_derivatives() do
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
    AEROMESH, FOIL, dim, Λ, U∞, ω, elemType="BT2";
    appendageOptions=Dict{String,Any}("config" => "wing"), STRUT=nothing,
    use_nlll=false, LLSystem=nothing
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
    """

    aeroMesh = AEROMESH.mesh
    # elemConn = AEROMESH.elemConn

    # --- Initialize global matrices ---
    globalMf_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalCf_r_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalCf_i_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalKf_r_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    globalKf_i_z = Zygote.Buffer(zeros(RealOrComplex, dim, dim))
    # Zygote initialization
    globalMf_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalCf_r_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalCf_i_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalKf_r_z[:, :] = zeros(RealOrComplex, dim, dim)
    globalKf_i_z[:, :] = zeros(RealOrComplex, dim, dim)

    # --- Initialize planform area counter ---
    planformArea = 0.0
    chordVec = FOIL.chord
    abVec = FOIL.ab
    ebVec = FOIL.eb

    # Spline to get lift slope in the right spots if using nonlinear LL
    clαVec = FOIL.clα
    

    if STRUT != nothing
        strutclαVec = STRUT.clα
        strutChordVec = STRUT.chord
        strutabVec = STRUT.ab
        strutebVec = STRUT.eb
    end

    jj = 1 # node index
    # nElemWing = solverOptions["nNodes"] - 1
    # nElemStrut = solverOptions["nNodeStrut"] - 1
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
    junctionNodeX = aeroMesh[1, :]

    # for inode in eachindex(aeroMesh[:, 1]) # loop aero strips (located at FEM nodes)
    for (inode, XN) in enumerate(eachrow(aeroMesh)) # loop aero strips (located at FEM nodes)
        # @inbounds begin
        # --- compute strip quantities ---
        # XN = aeroMesh[inode, :]
        yⁿ = XN[YDIM]
        zⁿ = XN[ZDIM]
        # println(XN)


        nVec = stripVecs[inode, :]

        # TODO: use the nVec to grab sweep and dihedral effects, then use the external Lambda as inflow angle change
        lᵉ = √(nVec[XDIM]^2 + nVec[YDIM]^2 + nVec[ZDIM]^2) # length of elem
        Δy = lᵉ

        # If we have end point nodes, we need to divide the strip width by 2
        if appendageOptions["config"] == "wing"
            if inode == 1 || inode == FOIL.nNodes
                Δy = 0.5 * lᵉ
            end
        elseif appendageOptions["config"] == "full-wing"
            if inode == 1 || inode == FOIL.nNodes || (inode == nElemWing * 2 + 1)
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
        @bp
        if use_nlll # TODO: FIX LATER TO BE GENERAL
            xeval = LLSystem.collocationPts[YDIM, :]
            clα = SolverRoutines.do_linear_interp(xeval, FOIL.clα, yⁿ)
            sDomFoil = aeroMesh[1:FOIL.nNodes, YDIM]
            if inode <= FOIL.nNodes # STBD WING
                c = SolverRoutines.do_linear_interp(sDomFoil, chordVec, yⁿ)
                ab = SolverRoutines.do_linear_interp(sDomFoil, abVec, yⁿ)
                eb = SolverRoutines.do_linear_interp(sDomFoil, ebVec, yⁿ)
            else
                if appendageOptions["config"] in ["t-foil", "full-wing"]
                    if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                        # Put negative sign on the linear interp routine bc there is a bug!
                        sDomFoil = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])

                        c = SolverRoutines.do_linear_interp(sDomFoil, chordVec, -yⁿ)
                        ab = SolverRoutines.do_linear_interp(sDomFoil, abVec, -yⁿ)
                        eb = SolverRoutines.do_linear_interp(sDomFoil, ebVec, -yⁿ)
                        # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                        dR1 = -dR1
                        dR2 = -dR2
                        dR3 = -dR3
                    else # strut section
                        sDomFoil = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
                        c = SolverRoutines.do_linear_interp(sDomFoil, strutChordVec, zⁿ)
                        ab = SolverRoutines.do_linear_interp(sDomFoil, strutabVec, zⁿ)
                        eb = SolverRoutines.do_linear_interp(sDomFoil, strutebVec, zⁿ)
                    end
                end
            end
            # println("clα: ", @sprintf("%.4f", clα), "\teb: ", @sprintf("%.4f",eb), "\tyn: $(yⁿ)")
        else
            if inode <= FOIL.nNodes # STBD WING
                sDom = aeroMesh[1:FOIL.nNodes, YDIM]
                clα = SolverRoutines.do_linear_interp(sDom, clαVec, yⁿ)
                c = SolverRoutines.do_linear_interp(sDom, chordVec, yⁿ)
                ab = SolverRoutines.do_linear_interp(sDom, abVec, yⁿ)
                eb = SolverRoutines.do_linear_interp(sDom, ebVec, yⁿ)
            else
                if appendageOptions["config"] == "t-foil"
                    if inode <= nElemWing * 2 + 1 # fix this logic for elems based!
                        # Put negative sign on the linear interp routine bc there is a bug!
                        sDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
                        yⁿ = -1 * yⁿ

                        clα = SolverRoutines.do_linear_interp(sDom, clαVec, yⁿ)
                        c = SolverRoutines.do_linear_interp(sDom, chordVec, yⁿ)
                        ab = SolverRoutines.do_linear_interp(sDom, abVec, yⁿ)
                        eb = SolverRoutines.do_linear_interp(sDom, ebVec, yⁿ)
                        # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                        dR1 = -dR1
                        dR2 = -dR2
                        dR3 = -dR3
                        # println("I'm a port wing strip")
                    else
                        sDom = vcat(junctionNodeX[ZDIM], aeroMesh[FOIL.nNodes*2:end, ZDIM])
                        clα = SolverRoutines.do_linear_interp(sDom, strutclαVec, zⁿ)
                        c = SolverRoutines.do_linear_interp(sDom, strutChordVec, zⁿ)
                        ab = SolverRoutines.do_linear_interp(sDom, strutabVec, zⁿ)
                        eb = SolverRoutines.do_linear_interp(sDom, strutebVec, zⁿ)
                        # println("I'm a strut strip")
                    end
                elseif appendageOptions["config"] == "full-wing"
                    if inode <= nElemWing * 2 + 1
                        # Put negative sign on the linear interp routine bc there is a bug!
                        sDom = -1 * vcat(junctionNodeX[YDIM], aeroMesh[FOIL.nNodes+1:FOIL.nNodes*2-1, YDIM])
                        yⁿ = -1 * yⁿ

                        clα = SolverRoutines.do_linear_interp(sDom, clαVec, yⁿ)
                        c = SolverRoutines.do_linear_interp(sDom, chordVec, yⁿ)
                        ab = SolverRoutines.do_linear_interp(sDom, abVec, yⁿ)
                        eb = SolverRoutines.do_linear_interp(sDom, ebVec, yⁿ)
                        # For the PORT wing, we want the AICs to be equal to the STBD wing, just mirrored through the origin
                        dR1 = -dR1
                        dR2 = -dR2
                        dR3 = -dR3
                    end
                end
            end
        end
        b = 0.5 * c # semichord for more readable code

        # --- Precomputes ---
        clambda = cos(Λ)
        slambda = sin(Λ)
        k = ω * b / (U∞ * clambda) # local reduced frequency
        # Do Theodorsen computation once for efficiency
        if abs(ω) <= MEPSLARGE
            Ck = 1.0
        else
            CKVec = compute_theodorsen(k)
            Ck = CKVec[1] + 1im * CKVec[2]
        end

        K_f, K̂_f = compute_node_stiff_faster(clα, b, eb, ab, U∞, clambda, slambda, FOIL.ρ_f, Ck)
        C_f, Ĉ_f = compute_node_damp_faster(clα, b, eb, ab, U∞, clambda, slambda, FOIL.ρ_f, Ck)
        M_f = compute_node_mass(b, ab, FOIL.ρ_f)

        # --- Compute Compute local AIC matrix for this element ---
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

        # ---------------------------
        #   Transformation of AIC
        # ---------------------------
        # Aerodynamics need to happen in global reference frame
        Γ = SolverRoutines.get_transMat(dR1, dR2, dR3, 1.0, elemType)
        # DOUBLE CHECK IF THIS TRANSFORMATION IS CORRECT FOR PORT WING
        # I think yes
        KLocal = Γ'[1:NDOF, 1:NDOF] * KLocal * Γ[1:NDOF, 1:NDOF]
        CLocal = Γ'[1:NDOF, 1:NDOF] * CLocal * Γ[1:NDOF, 1:NDOF]
        MLocal = Γ'[1:NDOF, 1:NDOF] * MLocal * Γ[1:NDOF, 1:NDOF]

        GDOFIdx::Int64 = NDOF * (inode - 1) + 1

        # Add local AIC to global AIC and remember to multiply by strip width to get the right result
        globalKf_r_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = real(KLocal) * Δy
        globalKf_i_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = imag(KLocal) * Δy
        globalCf_r_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = real(CLocal) * Δy
        globalCf_i_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = imag(CLocal) * Δy
        globalMf_z[GDOFIdx:GDOFIdx+NDOF-1, GDOFIdx:GDOFIdx+NDOF-1] = MLocal * Δy

        # Add rectangle to planform area
        if inode <= FOIL.nNodes
            planformArea += c * Δy
            # else
            # println("Not adding planform area for strut or mirrored wing")
        end
        # inode += 1 # increment strip counter
        # end # inbounds
    end

    return copy(globalMf_z), copy(globalCf_r_z), copy(globalCf_i_z), copy(globalKf_r_z), copy(globalKf_i_z), planformArea
end

function get_strip_vecs(
    AEROMESH, solverOptions
)
    """

    Compute the spanwise tangent vectors for each strip

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
        # if solverOptions["debug"]
        #     println("treating last strip (node) tangent vector as a continuation of previous node")
        # end
        stripVecs_z[end, :] = aeroMesh[end, :] - aeroMesh[end-1, :]
    end

    return copy(stripVecs_z)
end

function compute_genHydroLoadsMatrices(kMax, nk, U∞, b_ref, dim, AEROMESH, Λ, FOIL, elemType)
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
        globalMf, globalCf_r, globalCf_i, globalKf_r, globalKf_i = HydroStrip.compute_AICs(AEROMESH, FOIL, dim, Λ, U∞, ω, elemType)

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

function integrate_hydroLoads(
    foilStructuralStates, fullAIC, α₀, rake, dofBlank::Vector{Int64}, downwashAngles::DTYPE, elemType="BT2";
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
    ForceVector = zeros(DTYPE, length(foilTotalStates))
    ForceVector_z = Zygote.Buffer(zeros(DTYPE, length(foilTotalStates)))
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

    ChainRulesCore.ignore_derivatives() do
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
    AbsTotalLift = 0.0
    for secLift in Fz
        AbsTotalLift += abs(secLift)
    end
    AbsTotalMoment = 0.0
    for secMom in My
        AbsTotalMoment += abs(secMom)
    end
    @bp

    return ForceVector, AbsTotalLift, AbsTotalMoment
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
