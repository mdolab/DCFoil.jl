# --- Julia 1.9 ---
"""
@File    :   LiftingLine.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   Modern lifting line from Phillips and Snyder 2000, Reid 2020 appendix
             The major weakness is the discontinuity in the locus of aerodynamic centers (LAC)
             for a highly swept wing at the root AND the mathematical requirement that the LAC 
             be locally perpendicular to the trailing vortex (TV). Reid 2020 overcame this by
             using a blending function at the wing root and a jointed TV
"""

module LiftingLine

# --- PACKAGES ---
using FLOWMath: abs_cs_safe, atan_cs_safe, norm_cs_safe
using Plots
using LinearAlgebra
using AbstractDifferentiation: AbstractDifferentiation as AD
using ChainRulesCore: ChainRulesCore, NoTangent, ZeroTangent, @ignore_derivatives
using Zygote
using FiniteDifferences
using PyCall
# using Debugger

# --- DCFoil modules ---
using ..VPM: VPM
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..Utilities: Utilities, compute_KS
using ..Preprocessing: Preprocessing
using ..DCFoil: DTYPE
using ..SolverRoutines: SolverRoutines, compute_anglesFromVector, compute_vectorFromAngle, normalize_3Dvector, cross3D

const Δα = 1e-3 # [rad] Finite difference step for lift slope calculations

# ==============================================================================
#                         Structs
# ==============================================================================
struct LiftingLineMesh{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF},TH<:AbstractArray{TF,3}}
    """
    Only geometry and mesh information
    """
    nodePts::TM # LL node points
    collocationPts::TM # Control points
    jointPts::TM # TV joint points
    npt_wing::TI # Number of wing points
    localChords::TA # Local chord lengths of the panel edges [m]
    localChordsCtrl::TA # Local chord lengths of the control points [m]
    sectionVectors::TM # Nondimensional section vectors, "dζi" in paper
    sectionLengths::TA # Section lengths
    sectionAreas::TA # Section areas
    # HydroProperties # Hydro properties at the cross sections
    npt_airfoil::TI # Number of airfoil points
    span::TF # Span of one wing
    planformArea::TF
    SRef::TF # Reference area [m^2]
    AR::TF # Aspect ratio
    rootChord::TF # Root chord [m]
    sweepAng::TF # Wing sweep angle [rad]
    rc::TF # Finite-core vortex radius (viscous correction) [m]
    wing_xyz_eff::TH # Effective wing LAC coordinates per control point
    wing_joint_xyz_eff::TH # Effective TV joint locations per control point
    local_sweeps::TA
    local_sweeps_eff::TM
    local_sweeps_ctrl::TA
end

struct LiftingLineHydro{TF,TM<:AbstractMatrix{TF}}
    """
    Hydro section properties
    """
    airfoil_CLa::TF # Airfoil lift slope ∂cl/∂α [1/rad]
    airfoil_aL0::TF # Airfoil zero-lift angle of attack [rad]
    airfoil_xy::TM # Airfoil coordinates
    airfoil_ctrl_xy::TM # Airfoil control points
end

struct FlowConditions{TF,TC,TA<:AbstractVector{TC}}
    Uinfvec::TA # Freestream velocity [m/s] [U, V, W]
    Uinf::TC # Freestream velocity magnitude [m/s]
    uvec::TA # Freestream velocity unit vector
    alpha::TC # Angle of attack [rad]
    beta::TF
    rhof::TF # Freestream density [kg/m^3]
end

struct LiftingLineOutputs{TF,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    """
    Nondimensional and dimensional outputs of interest
    Redimensionalize with the reference area and velocity
    """
    Fdist::TM # Loads distribution vector (Dimensional) [N]
    Γdist::TA # Converged circulation distribution (Γᵢ) [m^2/s]
    cla::TA # Spanwise lift slopes [1/rad]
    cl::TA # Spanwise lift coefficients
    F::TA # Total integrated loads vector [Fx, Fy, Fz, Mx, My, Mz]
    CL::TF # Lift coefficient (perpendicular to freestream in symmetry plane)
    CDi::TF # Induced drag coefficient (aligned w/ freestream)
    CS::TF # Side force coefficient
end

struct LiftingLineNLParams
    """
    Parameters needed in the nonlinear solve of the LL
    """
    TV_influence
    LLSystem # LiftingLineMesh
    LLHydro # LiftingLineHydro
    FlowCond # Flow conditions
    # Stuff for the 2D VPM solve
    Airfoils #
    AirfoilInfluences #
end

function initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)

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

    Uvec = [cos(deg2rad(α0)), 0.0, sin(deg2rad(α0))] * solverOptions["Uinf"]

    # Rotate by RH rule by leeway angle
    Tz = SolverRoutines.get_rotate3dMat(deg2rad(β0); axis="z")
    Uvec = Tz * Uvec

    options = Dict(
        "translation" => vec([appendageOptions["xMount"], 0, 0]), # of the midchord
        "debug" => true,
    )
    return airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options
end

function setup(Uvec, sweepAng, rootChord, taperRatio, midchords;
    npt_wing=99, npt_airfoil=199, blend=0.25, δ=0.15, rc=0.0, rhof=1025.0,
    airfoil_xy=nothing, airfoil_ctrl_xy=nothing, airfoilCoordFile=nothing, options=nothing)
    """
    Initialize and setup the lifting line model for one wing

    Inputs:
    -------
    wingSpan : scalar
        The span of the wing [m] (after sweep is applied, so this is not the structural span!)
    sweepAng : scalar
        The wing sweep angle in rad.
    blend : scalar , optional
        The normalized blending distance, used to calculate the
        effective loci of aerodynamic centers.
    δ : scalar 0.15, optional
        The fraction of the local chord the vortex segment portion of the
        TV extends from the LAC.
    rc : scalar 0.0, optional
        The finite-core vortex radius (viscous correction) [m]
    airfoilCoordFile : filename, optional
        The filename of the airfoil coordinates to use. If not
        provided, the airfoil_xy and airfoil_ctrl_xy arrays are used
    options : dict, optional
        Dictionary of options to pass to the lifting line model regarding debug stuff
    """

    # ************************************************
    #     Airfoil hydro properties
    # ************************************************
    if !isnothing(airfoilCoordFile) && isnothing(airfoil_xy) && isnothing(airfoil_ctrl_xy)

        println("Reading airfoil coordinates from $(airfoilCoordFile) and using MACH...")

        PREFOIL = pyimport("prefoil")

        rawCoords = PREFOIL.utils.readCoordFile(airfoilCoordFile)

        Foil = PREFOIL.Airfoil(rawCoords)

        Foil.normalizeChord()

        airfoil_pts = Foil.getSampledPts(
            nPts=npt_airfoil + 1, # one more to delete the TE knot
            spacingFunc=PREFOIL.sampling.conical,
            func_args=Dict("coeff" => 1),
            TE_knot=false # weird stuff going on with a trailing knot
        )

        airfoil_ctrl_pts = (airfoil_pts[1:end-2, :] .+ airfoil_pts[2:end-1, :]) .* 0.5

        # --- Transpose and reverse since PreFoil is different ---
        airfoil_xy = reverse(transpose(airfoil_pts[1:end-1, :]), dims=2)
        airfoil_ctrl_xy = reverse(transpose(airfoil_ctrl_pts), dims=2)

        # elseif !isnothing(airfoil_xy) && !isnothing(airfoil_ctrl_xy)
        #     println("Using provided airfoil coordinates")
    end
    # The initial hydro properties use zero sweep
    LLHydro, Airfoil, Airfoil_influences = compute_hydroProperties(0.0, airfoil_xy, airfoil_ctrl_xy)

    # ************************************************
    #     Preproc stuff
    # ************************************************
    # --- Structural span is not the same as aero span ---
    aeroWingSpan = Preprocessing.compute_aeroSpan(midchords)

    # wingSpan = span * cos(sweepAng) #no

    # Blending parameter for the LAC
    σ = 4 * cos(sweepAng)^2 / (blend^2 * aeroWingSpan^2)

    alpha, beta, Uinf = compute_anglesFromVector(Uvec)
    uvec = Uvec / Uinf

    # Wing area
    SRef = rootChord * aeroWingSpan * (1 + taperRatio) * 0.5
    AR = aeroWingSpan^2 / SRef

    translation = zeros(3)
    if !isnothing(options) && haskey(options, "translation")
        translation = options["translation"] # [m] 3d translation of the wing
    end
    # Apply translation wrt midchords TODO: this translation is actually incorrect for swept wings. need to revise this
    translatMatCtrl = repeat(reshape(translation, size(translation)..., 1), 1, npt_wing)
    translatMat = repeat(reshape(translation, size(translation)..., 1), 1, npt_wing + 1)
    #     wing_ctrl_xyz[:, ii] .+= translation #- 0.25 * vec([local_chords_ctrl[ii] + rootChord, 0.0, 0.0])

    # ************************************************
    #     Make wing coordinates
    # ************************************************
    # ---------------------------
    #   Y coords (span)
    # ---------------------------
    start = -aeroWingSpan * 0.5
    stop = aeroWingSpan * 0.5

    # --- Even spacing ---
    θ_bound = LinRange(start, stop, npt_wing * 2 + 1)
    wing_xyz_ycomp = reshape(θ_bound[1:2:end], 1, npt_wing + 1)
    wing_ctrl_xyz_ycomp = reshape(θ_bound[2:2:end], 1, npt_wing)

    Zeros = zeros(1, npt_wing + 1)
    ZerosCtrl = zeros(1, npt_wing)
    wing_xyz = cat(Zeros, wing_xyz_ycomp, Zeros, dims=1)
    wing_ctrl_xyz = cat(ZerosCtrl, wing_ctrl_xyz_ycomp, ZerosCtrl, dims=1)

    # --- Cosine spacing ---
    if abs_cs_safe(sweepAng) > 0.0
        # θ_bound = PREFOIL.sampling.cosine(start, stop, npt_wing * 2 + 1, 2π)
        # println("θ_bound: $(θ_bound)")
        θ_bound = LinRange(0.0, 2π, npt_wing * 2 + 1)
        wing_xyz_ycomp = reshape([sign(θ - π) * 0.25 * aeroWingSpan * (1 + cos(θ)) for θ in θ_bound[1:2:end]], 1, npt_wing + 1)
        wing_ctrl_xyz_ycomp = reshape([sign(θ - π) * 0.25 * aeroWingSpan * (1 + cos(θ)) for θ in θ_bound[2:2:end]], 1, npt_wing)
    end

    # ---------------------------
    #   X coords (chord dist)
    # ---------------------------
    iTR = 1.0 - taperRatio

    local_chords = rootChord * (1.0 .- 2.0 * iTR * abs_cs_safe.(wing_xyz_ycomp[1, :]) / aeroWingSpan)
    local_chords_ctrl = rootChord * (1.0 .- 2.0 * iTR * abs_cs_safe.(wing_ctrl_xyz_ycomp[1, :]) / aeroWingSpan)

    # ∂c/∂y
    local_dchords = 2.0 * rootChord * (-iTR) * sign.(wing_xyz_ycomp[1, :]) / aeroWingSpan
    local_dchords_ctrl = 2.0 * rootChord * (-iTR) * sign.(wing_ctrl_xyz_ycomp[1, :]) / aeroWingSpan

    # --- Locus of aerodynamic centers (LAC) ---
    LAC = compute_LAC(AR, LLHydro, wing_xyz_ycomp[1, :], local_chords, rootChord, sweepAng, aeroWingSpan)
    wing_xyz = cat(reshape(LAC, 1, size(LAC)...), wing_xyz_ycomp, Zeros, dims=1) .+ translatMat

    LAC_ctrl = compute_LAC(AR, LLHydro, wing_ctrl_xyz_ycomp[1, :], local_chords_ctrl, rootChord, sweepAng, aeroWingSpan)
    wing_ctrl_xyz = cat(reshape(LAC_ctrl, 1, size(LAC_ctrl)...), wing_ctrl_xyz_ycomp, ZerosCtrl, dims=1) .+ translatMatCtrl

    # Need a mess of LAC's for each control point
    LACeff = compute_LACeffective(AR, LLHydro, wing_xyz[YDIM, :], wing_ctrl_xyz[YDIM, :], local_chords, local_chords_ctrl, local_dchords, local_dchords_ctrl, σ, sweepAng, rootChord, aeroWingSpan)
    # This is a 3D array
    # wing_xyz_eff = zeros(3, npt_wing, npt_wing + 1)
    wing_xyz_eff_xcomp = reshape(LACeff, 1, size(LACeff)...)
    wing_xyz_eff_ycomp = reshape(repeat(transpose(wing_xyz[YDIM, :]), npt_wing, 1), 1, npt_wing, npt_wing + 1)
    wing_xyz_eff = cat(
        wing_xyz_eff_xcomp,
        wing_xyz_eff_ycomp,
        zeros(size(wing_xyz_eff_ycomp)),
        dims=1)

    # --- Compute local sweeps ---
    # Vectors containing local sweep at each coordinate location in wing_xyz
    fprime = compute_dLACds(AR, LLHydro, wing_xyz[YDIM, :], local_chords, local_dchords, sweepAng, aeroWingSpan)
    localSweeps = -atan_cs_safe.(fprime, ones(size(fprime)))

    fprimeCtrl = compute_dLACds(AR, LLHydro, wing_ctrl_xyz[YDIM, :], local_chords_ctrl, local_dchords_ctrl, sweepAng, aeroWingSpan)
    localSweepsCtrl = -atan_cs_safe.(fprimeCtrl, ones(size(fprimeCtrl)))

    fprimeEff = compute_dLACdseffective(AR, LLHydro, wing_xyz[YDIM, :], wing_ctrl_xyz[YDIM, :], local_chords, local_chords_ctrl, local_dchords, local_dchords_ctrl, σ, sweepAng, rootChord, aeroWingSpan)
    localSweepEff = -atan_cs_safe.(fprimeEff, ones(size(fprimeEff)))

    # --- Other section properties ---
    sectionVectors = wing_xyz[:, 1:end-1] - wing_xyz[:, 2:end] # dℓᵢ

    sectionLengths = .√(sectionVectors[XDIM, :] .^ 2 + sectionVectors[YDIM, :] .^ 2 + sectionVectors[ZDIM, :] .^ 2) # ℓᵢ
    sectionAreas = 0.5 * (local_chords[1:end-1] + local_chords[2:end]) .* abs_cs_safe.(wing_xyz[YDIM, 1:end-1] - wing_xyz[YDIM, 2:end]) # dAᵢ

    ζ = sectionVectors ./ reshape(sectionAreas, 1, size(sectionAreas)...) # Normalized section vectors, [3, npt_wing]

    # ---------------------------
    #   Aero section properties
    # ---------------------------
    # Where the 2D VPM comes into play
    Airfoils = Vector(undef, npt_wing)
    AirfoilInfluences = Vector(undef, npt_wing)
    Airfoils_z = Zygote.Buffer(Airfoils)
    AirfoilInfluences_z = Zygote.Buffer(AirfoilInfluences)
    for (ii, sweep) in enumerate(localSweepsCtrl)
        # Pass in copies because this routine was modifying the input
        Airfoil, Airfoil_influences = VPM.setup(copy(airfoil_xy[XDIM, :]), copy(airfoil_xy[YDIM, :]), copy(airfoil_ctrl_xy), sweep)
        Airfoils_z[ii] = Airfoil
        AirfoilInfluences_z[ii] = Airfoil_influences
        # println("Airfoil control: $(airfoil_ctrl_xy[XDIM,:])")
    end
    # # List comprehension version
    # Airfoils, AirfoilInfluences = [VPM.setup(copy(airfoil_xy[XDIM, :]), copy(airfoil_xy[YDIM, :]), copy(airfoil_ctrl_xy), sweep) for sweep in localSweepsCtrl]
    Airfoils = copy(Airfoils_z)
    AirfoilInfluences = copy(AirfoilInfluences_z)

    # ---------------------------
    #   TV joint locations
    # ---------------------------
    # These are where the bound vortex lines kink and then bend to follow the freestream direction

    local_chords_colmat = reshape(local_chords, 1, size(local_chords)...)

    wing_joint_xyz_xcomp = reshape(wing_xyz[XDIM, :] + δ * local_chords .* cos.(localSweeps), 1, npt_wing + 1)
    wing_joint_xyz_eff_xcomp = reshape(wing_xyz_eff[XDIM, :, :] + δ * local_chords_colmat .* cos.(localSweepEff), 1, npt_wing, npt_wing + 1)


    wing_joint_xyz_ycomp = reshape(wing_xyz[YDIM, :] + δ * local_chords .* sin.(localSweeps), 1, npt_wing + 1)
    wing_joint_xyz_eff_ycomp = reshape(transpose(wing_xyz[YDIM, :]) .+ δ * local_chords_colmat .* sin.(localSweepEff), 1, npt_wing, npt_wing + 1)

    wing_joint_xyz = cat(wing_joint_xyz_xcomp, wing_joint_xyz_ycomp, Zeros, dims=1)
    wing_joint_xyz_eff = cat(wing_joint_xyz_eff_xcomp, wing_joint_xyz_eff_ycomp, zeros(1, npt_wing, npt_wing + 1), dims=1)

    # println("wing_joint_xyz_eff y: $(wing_joint_xyz_eff[YDIM,1,2:end])")
    # println("wing_ctrl_xyz x: $(wing_ctrl_xyz[XDIM,:])")

    # println("local sweep values [deg]: $(rad2deg.(localSweeps))")

    # Store all computed quantities here
    LLMesh = LiftingLineMesh(wing_xyz, wing_ctrl_xyz, wing_joint_xyz, npt_wing, local_chords, local_chords_ctrl, ζ, sectionLengths, sectionAreas,
        npt_airfoil, aeroWingSpan, SRef, SRef, AR, rootChord, sweepAng, rc, wing_xyz_eff, wing_joint_xyz_eff,
        localSweeps, localSweepEff, localSweepsCtrl)

    FlowCond = FlowConditions(Uvec, Uinf, uvec, alpha, beta, rhof)

    return LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences
end

function compute_LAC(AR, LLHydro, y, c, cr, Λ, span; model="kuechemann")
    """
    Compute the locus of aerodynamic centers (LAC) for the wing

    Küchemann's 1956 method for the LAC of a constant swept wing with AR effects

    Parameters
    ----------
    y : spanwise coordinate [m]
    c : chord length at location y [m]
    cr : chord length at the root [m]
    Λ : global sweep of the wing [rad]
    span : full span of the wing [m]

    Returns
    -------
    x : location of the aerodynamic center at location y [m]
    """

    if model == "kuechemann"
        Λₖ = Λ / (1.0 + (LLHydro.airfoil_CLa * cos(Λ) / (π * AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LLHydro.airfoil_CLa * cos(Λₖ) / (π * AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        if Λ == 0
            fs = 0.25 * cr .- c * (1.0 - 1.0 / K) / 4.0
        else
            tanl = vec(2π * tan(Λₖ) ./ (Λₖ * c))
            lam = .√(1.0 .+ (tanl .* y) .^ 2) .-
                  tanl .* abs_cs_safe.(y) .-
                  .√(1.0 .+ (tanl .* (0.5 * span .- abs_cs_safe.(y))) .^ 2) .+
                  tanl .* (0.5 * span .- abs_cs_safe.(y))

            fs = 0.25 * cr .+
                 tan(Λ) .* abs_cs_safe.(y) .-
                 c .* (1.0 .- (1.0 .+ 2.0 * lam * Λₖ / π) / K) * 0.25
        end
    else
        println("Model not implemented yet")
    end

    return fs
end

function compute_LACeffective(AR, LLHydro, y, y0, c, c_y0, dc, dc_y0, σ, Λ, cr, span; model="kuechemann")
    """
    The effective LAC, based on Küchemann's equation .

    Parameters
    ----------
    y : spanwise coordinate
    y0 : control point location
    c : chord length at position y
    c_y0 : chord length at control point z0
    dc : change in chord length at location y, dc/dy
    dc_y0 : change in chord length at control point y0 , dc/dy
    σ : blend strength factor
    Λ : global sweep of the wing [rad]
    cr : chord length at the root
    span : full span of the wing
    model : LAC model to blend

    Returns
    -------
    x : location of the effective aerodynamic center at point y
    """

    # This is a matrix
    ywork = reshape(y, 1, size(y)...)
    y0work = reshape(y0, size(y0)..., 1)
    blend = exp.(-σ * (y0work .- ywork) .^ 2)

    if model == "kuechemann"

        LAC = compute_LAC(AR, LLHydro, ywork[1, :], c, cr, Λ, span)
        LACwork = reshape(LAC, 1, size(LAC)...)

        LAC0 = compute_LAC(AR, LLHydro, y0work[:, 1], c_y0, cr, Λ, span)
        LAC0work = reshape(LAC0, size(LAC0)..., 1)

        fprime0 = compute_dLACds(AR, LLHydro, y0work[:, 1], c_y0, dc_y0, Λ, span)

        LACeff = (1.0 .- blend) .* LACwork .+
                 blend .* (fprime0 .* (ywork .- y0work) .+ LAC0work)
        # println("c: $(c)")
        # println("cr: $(cr)")
        # println("sweep: $(Λ)")
        # println("y: $(y)")
        # println("y0: $(y0)")
        # println("LACwork: $(LACwork[1,:])")
        # println("LAC0work: $(LAC0work[:,1])")
        # println("LACeff: $(LACeff[1,:])")
        return LACeff
    else
        println("Model not implemented yet")
    end
end

function compute_dLACds(AR, LLHydro, y, c, ∂c∂y, Λ, span; model="kuechemann")
    """
    Compute the derivative of the LAC curve wrt the spanwise coordinate
    f'(s)
    Parameters
    ----------
    y : spanwise coordinate
    c : chord length at location y
    ∂c∂y : change in chord length at location y
    Λ : global sweep of the wing (rad)
    span : full span of the wing

    Returns
    -------
    dx : change in the location of the aerodynamic center at location y
    """

    if model == "kuechemann"
        Λₖ = Λ / (1.0 + (LLHydro.airfoil_CLa * cos(Λ) / (π * AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LLHydro.airfoil_CLa * cos(Λₖ) / (π * AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        if Λ == 0
            dx = -∂c∂y * (1.0 - 1.0 / K) * 0.25
        else
            tanl = vec(2π * tan(Λₖ) ./ (Λₖ * c))
            lam = .√(1.0 .+ (tanl .* y) .^ 2) .-
                  tanl .* abs_cs_safe.(y) .-
                  .√(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2) .+
                  tanl .* (span / 2.0 .- abs_cs_safe.(y))

            lamp = ((tanl .^ 2 .* (y .* c .- y .^ 2 .* ∂c∂y) ./ c) ./ .√(1.0 .+ (tanl .* y) .^ 2) -
                    tanl .* (sign.(y) .* c .- abs_cs_safe.(y) .* ∂c∂y) ./ c +
                    ((tanl .^ 2 .* (sign.(y) .* (span / 2.0 .- abs_cs_safe.(y)) .* c .+ ∂c∂y .* (span / 2.0 .- abs.(y)) .^ 2) ./ c) ./ .√(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2)) -
                    tanl .* (sign.(y) .* c .+ (span / 2.0 .- abs_cs_safe.(y)) .* ∂c∂y) ./ c)

            dx = tan(Λ) * sign.(y) .+
                 lamp * Λₖ .* c / (2π * K) .-
                 ∂c∂y .* (1.0 .- (1.0 .+ 2.0 * lam * Λₖ / π) / K) * 0.25
        end
    else
        println("Model not implemented yet")
    end

    # println("Λk: $(Λₖ)")
    # println("K: $(K)")
    # println("====================================")
    # println("y: $(y)")
    # println("c: $(c)")
    # println("tanl: $(tanl)")
    # println("lam: $(lam)")
    # println("lamp: $(lamp)")
    # println("dx:\n $(dx)") # good
    # println("====================================")

    return dx
end

function compute_dLACdseffective(AR, LLHydro, y, y0, c, c_y0, dc, dc_y0, σ, Λ, cr, span; model="kuechemann")
    """
    The derivative of the effective LAC , based on Kuchemann 's equation .

    Parameters
    ----------
    y : spanwise coordinate
    y0 : control point location
    c : chord length at position y
    c_y0 : chord length at control point z0
    dc : change in chord length at location y, dc/dy
    dc_y0 : change in chord length at control point y0 , dc/dy
    σ : blend strength factor
    Λ : global sweep of the wing [rad]
    cr : chord length at the root
    span : full span of the wing
    model : LAC model to blend

    Returns
    -------
    x : change in location of the effective aerodynamic center at point y
    """

    # This is a matrix
    ywork = reshape(y, 1, length(y))
    y0work = reshape(y0, length(y0), 1)
    blend = exp.(-σ * (y0work .- ywork) .^ 2)

    if model == "kuechemann"

        LAC = compute_LAC(AR, LLHydro, y, c, cr, Λ, span)
        LACwork = reshape(LAC, 1, length(LAC))
        fprime = compute_dLACds(AR, LLHydro, y, c, dc, Λ, span)
        fprimework = reshape(fprime, 1, length(fprime))
        LAC0 = compute_LAC(AR, LLHydro, y0, c_y0, cr, Λ, span)
        LAC0work = reshape(LAC0, length(LAC0), 1)
        fprime0 = compute_dLACds(AR, LLHydro, y0, c_y0, dc_y0, Λ, span)
        fprime0work = reshape(fprime0, length(fprime0), 1)

        return fprimework .+
               blend .*
               (fprime0work .- fprimework .-
                2 * σ * (y0work .- ywork) .*
                (fprime0work .* (y0work .- ywork) .-
                 (LAC0work .- LACwork)))
    else
        println("Model not implemented yet")
    end
end

function solve(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences; is_verbose=true)
    """
    Execute LL algorithm.
    Top level wrapper to interface with. 
    Taking derivatives is trickier and done analytically

    Inputs:
    -------
    LiftingSystem : LiftingLineSystem
        Lifting line system struct with all necessary parameters

    LLHydro : LiftingLineHydro
        Section properties at the root airfoil
    Returns:
    --------
    LLResults : LiftingLineResults
        Lifting line results struct with all necessary parameters
    """

    # --- Unpack data structs ---
    Uinf = FlowCond.Uinf
    α = FlowCond.alpha
    β = FlowCond.beta
    rhof = FlowCond.rhof
    DimForces, Γdist, ∂cl∂α, cl, IntegratedForces, CL, CDi, CS = compute_solution(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences; is_verbose=is_verbose)

    # --- Pack back up  ---
    LLResults = LiftingLineOutputs(DimForces, Γdist, ∂cl∂α, cl, IntegratedForces, CL, CDi, CS)

    return LLResults
end

function compute_solution(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences; is_verbose=true)

    ∂α = FlowCond.alpha + Δα # FD

    ∂Uinfvec = FlowCond.Uinf * [cos(∂α), 0, sin(∂α)]
    ∂Uinf = norm_cs_safe(∂Uinfvec)
    ∂uvec = ∂Uinfvec / FlowCond.Uinf
    ∂FlowCond = FlowConditions(∂Uinfvec, ∂Uinf, ∂uvec, ∂α, FlowCond.beta, FlowCond.rhof)

    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    uinf = reshape(FlowCond.uvec, 3, 1, 1)
    uinfMat = repeat(uinf, 1, LLMesh.npt_wing, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)
    ∂uinf = reshape(∂FlowCond.uvec, 3, 1, 1)
    ∂uinfMat = repeat(∂uinf, 1, LLMesh.npt_wing, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)

    P1 = LLMesh.wing_joint_xyz_eff[:, :, 2:end]
    P2 = LLMesh.wing_xyz_eff[:, :, 2:end]
    P3 = LLMesh.wing_xyz_eff[:, :, 1:end-1]
    P4 = LLMesh.wing_joint_xyz_eff[:, :, 1:end-1]

    ctrlPts = reshape(LLMesh.collocationPts, size(LLMesh.collocationPts)..., 1)
    ctrlPtMat = repeat(ctrlPts, 1, 1, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)


    # Mask for the bound segment (npt_wing x npt_wing)
    bound_mask = ones(LLMesh.npt_wing, LLMesh.npt_wing) - diagm(ones(LLMesh.npt_wing))

    influence_straightsega = compute_straightSegment(P1, P2, ctrlPtMat, LLMesh.rc)
    influence_straightsegb = compute_straightSegment(P2, P3, ctrlPtMat, LLMesh.rc) .* reshape(bound_mask, 1, size(bound_mask)...)
    influence_straightsegc = compute_straightSegment(P3, P4, ctrlPtMat, LLMesh.rc)

    ∂influence_semiinfa = compute_straightSemiinfinite(P1, ∂uinfMat, ctrlPtMat, LLMesh.rc)
    ∂influence_semiinfb = compute_straightSemiinfinite(P4, ∂uinfMat, ctrlPtMat, LLMesh.rc)

    TV_influence = compute_TVinfluences(FlowCond, LLMesh)

    ∂TV_influence = -∂influence_semiinfa +
                    influence_straightsega +
                    influence_straightsegb +
                    influence_straightsegc +
                    ∂influence_semiinfb


    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    # First guess using root properties
    c_r = LLMesh.rootChord
    clα = LLHydro.airfoil_CLa
    αL0 = LLHydro.airfoil_aL0
    Λ = LLMesh.sweepAng
    # Ux, _, Uz = FlowCond.Uinfvec
    ux, uy, uz = FlowCond.uvec
    ∂ux, ∂uy, ∂uz = ∂FlowCond.uvec
    span = LLMesh.span
    ctrl_pts = LLMesh.collocationPts
    ζi = LLMesh.sectionVectors
    dAi = reshape(LLMesh.sectionAreas, 1, size(LLMesh.sectionAreas)...)
    g0 = 0.5 * c_r * clα * cos(Λ) *
         (uz / ux - αL0) *
         (1.0 .- (2.0 * ctrl_pts[YDIM, :] / span) .^ 4) .^ (0.25)


    # --- Pack up parameters for the NL solve ---
    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
    ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    # --- Nonlinear solve for circulation distribution ---
    Gconv, residuals = SolverRoutines.converge_resNonlinear(compute_LLresiduals, compute_LLJacobian, g0;
        solverParams=LLNLParams, is_verbose=is_verbose,
        # mode="CS" # 
        mode="FiDi" # this is the fastest
        # mode="Analytic"
    )
    # println("Secondary solve for lift slope")
    ∂Gconv, ∂residuals = SolverRoutines.converge_resNonlinear(compute_LLresiduals, compute_LLJacobian, g0;
        solverParams=∂LLNLParams, is_verbose=is_verbose,
        #  is_cmplx=true,
        # mode="CS" # 
        mode="FiDi"  # this is the fastest
        # mode="ANALYTIC"
    )

    Gi = reshape(Gconv, 1, size(Gconv)...) # now it's a (1, npt) matrix
    ∂Gi = reshape(∂Gconv, 1, size(∂Gconv)...) # now it's a (1, npt) matrix
    Gjvji = TV_influence .* Gi
    Gjvjix = TV_influence[XDIM, :, :] * Gconv
    Gjvjiy = TV_influence[YDIM, :, :] * Gconv
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    Gjvjiz = -TV_influence[ZDIM, :, :] * Gconv
    Gjvji = cat(Gjvjix, Gjvjiy, Gjvjiz, dims=2)
    Gjvji = permutedims(Gjvji, [2, 1])
    u∞ = repeat(reshape(FlowCond.uvec, 3, 1), 1, LLMesh.npt_wing)

    ui = Gjvji .+ u∞ # Local velocities (nondimensional)

    # This is the Biot--Savart law but nondimensional
    # fi = 2 | ( ui ) × ζi| Gi dAi / SRef
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    uicrossζi = -cross.(eachcol(ui), eachcol(ζi))
    uicrossζi = hcat(uicrossζi...) # now it's a (3, npt) matrix
    coeff = 2.0 / LLMesh.SRef
    NondimForces = coeff * (uicrossζi .* Gi) .* dAi

    # Integrated = 2 Σ ( u∞ + Gⱼvⱼᵢ ) x ζᵢ * Gᵢ * dAᵢ / SRef
    IntegratedForces = vec(coeff * sum((uicrossζi .* Gi) .* dAi, dims=2))

    Γdist = Gconv * FlowCond.Uinf # dimensionalize the circulation distribution
    # Forces = NondimForces .* LLMesh.SRef * 0.5 * ϱ * FlowCond.Uinf^2 # dimensionalize the forces
    # println(Γdist)

    # --- Dimensional forces ---
    Γi = Gi * FlowCond.Uinf
    Γjvji = TV_influence .* Γi
    Γjvjix = TV_influence[XDIM, :, :] * Γdist
    Γjvjiy = TV_influence[YDIM, :, :] * Γdist
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    Γjvjiz = -TV_influence[ZDIM, :, :] * Γdist
    Γjvji = cat(Γjvjix, Γjvjiy, Γjvjiz, dims=2)
    Γjvji = permutedims(Γjvji, [2, 1])
    U∞ = repeat(reshape(FlowCond.Uinfvec, 3, 1), 1, LLMesh.npt_wing)

    Ui = Γjvji .+ U∞ # Local velocities
    Uicrossdli = -cross.(eachcol(Ui), eachcol(ζi))
    Uicrossdli = hcat(Uicrossdli...) # now it's a (3, npt) matrix
    DimForces = FlowCond.rhof * (Uicrossdli .* Γi) .* dAi

    # --- Vortex core viscous correction ---
    if LLMesh.rc != 0
        println("Vortex core viscous correction not implemented yet")
    end

    # --- Final outputs ---
    CL = -IntegratedForces[XDIM] * uz +
         IntegratedForces[ZDIM] * ux / (ux^2 + uz^2)
    CDi = IntegratedForces[XDIM] * ux +
          IntegratedForces[YDIM] * uy +
          IntegratedForces[ZDIM] * uz
    CS = (
        -IntegratedForces[XDIM] * ux * uy -
        IntegratedForces[ZDIM] * uz * uy +
        IntegratedForces[YDIM] * (uz^2 + ux^2)
    ) / √(ux^2 * uy^2 + uz^2 * uy^2 + (uz^2 + ux^2)^2)

    # --- Compute the lift curve slope ---
    # ∂G∂α = imag(∂Gconv) / Δα # CS
    ∂G∂α = (∂Gconv .- Gconv) / Δα # Forward Difference
    ∂cl∂α = 2 * ∂G∂α ./ LLMesh.localChordsCtrl
    clvec = 2 * Gconv ./ LLMesh.localChordsCtrl

    return DimForces, Γdist, ∂cl∂α, clvec, IntegratedForces, CL, CDi, CS

end

function compute_TVinfluences(FlowCond, LLMesh)
    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    uinf = reshape(FlowCond.uvec, 3, 1, 1)
    uinfMat = repeat(uinf, 1, LLMesh.npt_wing, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)

    P1 = LLMesh.wing_joint_xyz_eff[:, :, 2:end]
    P2 = LLMesh.wing_xyz_eff[:, :, 2:end]
    P3 = LLMesh.wing_xyz_eff[:, :, 1:end-1]
    P4 = LLMesh.wing_joint_xyz_eff[:, :, 1:end-1]

    ctrlPts = reshape(LLMesh.collocationPts, size(LLMesh.collocationPts)..., 1)
    ctrlPtMat = repeat(ctrlPts, 1, 1, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)


    # Mask for the bound segment (npt_wing x npt_wing)
    bound_mask = ones(LLMesh.npt_wing, LLMesh.npt_wing) - diagm(ones(LLMesh.npt_wing))

    influence_semiinfa = compute_straightSemiinfinite(P1, uinfMat, ctrlPtMat, LLMesh.rc)
    influence_straightsega = compute_straightSegment(P1, P2, ctrlPtMat, LLMesh.rc)
    influence_straightsegb = compute_straightSegment(P2, P3, ctrlPtMat, LLMesh.rc) .* reshape(bound_mask, 1, size(bound_mask)...)
    influence_straightsegc = compute_straightSegment(P3, P4, ctrlPtMat, LLMesh.rc)
    influence_semiinfb = compute_straightSemiinfinite(P4, uinfMat, ctrlPtMat, LLMesh.rc)

    TV_influence = -influence_semiinfa +
                   influence_straightsega +
                   influence_straightsegb +
                   influence_straightsegc +
                   influence_semiinfb
    return TV_influence
end

function compute_LLresiduals(G; solverParams=nothing)
    """
    Nonlinear , nondimensional lifting - line equation .
    Parameters
    ----------
    G : vector
    Circulation distribution normalized by the freestream velocity
    magnitude.

    Returns
    -------
    R : array_like
    Array of the residuals between the lift values predicted from
    section properties and from circulation.
    """

    if isnothing(solverParams)
        println("WARNING: YOU NEED TO PASS IN SOLVER PARAMETERS")
    end

    TV_influence = solverParams.TV_influence
    LLSystem = solverParams.LLSystem
    Airfoils = solverParams.Airfoils
    AirfoilInfluences = solverParams.AirfoilInfluences
    FlowCond = solverParams.FlowCond
    ζi = LLSystem.sectionVectors


    # This is a (3 , npt, npt) × (npt,) multiplication
    # PYTHON: _Vi = TV_influence * G .+ transpose(LLSystem.uvec)
    uix = TV_influence[XDIM, :, :] * G .+ FlowCond.uvec[XDIM]
    #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
    uiy = TV_influence[YDIM, :, :] * G .+ FlowCond.uvec[YDIM]
    uiz = -TV_influence[ZDIM, :, :] * G .+ FlowCond.uvec[ZDIM]
    ui = cat(uix, uiy, uiz, dims=2)
    ui = permutedims(ui, [2, 1])


    # Do a curve fit on aero props
    # if self._aero_approx:
    # _CL = self._lift_from_aero(*self._aero_properties, self.local_sweep_ctrl, self.Vinf * _Vi, self.Vinf)
    # else:
    # Actually solve VPM for each local velocity c
    Ui = FlowCond.Uinf * (ui) # dimensionalize the local velocities
    # println("Ui: $(Ui)\n") # OK

    c_l = [
        VPM.solve(Airfoils[ii], AirfoilInfluences[ii], V_local, 1.0, FlowCond.Uinf)[1]
        for (ii, V_local) in enumerate(eachcol(Ui))
    ] # remember to only grab CL out of VPM solve

    ui_cross_ζi = cross.(eachcol(ui), eachcol(ζi)) # this gives a vector of vectors, not a matrix, so we need double indexing --> [][]
    ui_cross_ζi = hcat(ui_cross_ζi...) # now it's a (3, npt) matrix
    ui_cross_ζi_mag = .√(ui_cross_ζi[XDIM, :] .^ 2 + ui_cross_ζi[YDIM, :] .^ 2 + ui_cross_ζi[ZDIM, :] .^ 2)


    dFimag = 2.0 * ui_cross_ζi_mag .* G

    return dFimag - c_l
end

function compute_LLJacobian(Gi; solverParams, mode="Analytic")
    """
    Compute the Jacobian of the nonlinear, nondimensional lifting line equation

    Inputs:
    -------
    Gi - Circulation distribution normalized by freestream velocity Γ / Uinf

    Returns:
    --------
    J - Jacobian matrix, matrix of partial derivatives ∂r/∂G for
        r(G) = Nondim LL eqn

    """

    if uppercase(mode) == "ANALYTIC" # After many hours of debugging, it matches Python but still doesn't converge...robustness issue

        TV_influence = solverParams.TV_influence
        LLSystem = solverParams.LLSystem
        LLHydro = solverParams.LLHydro
        # Airfoils = solverParams.Airfoils
        # AirfoilInfluences = solverParams.AirfoilInfluences
        FlowCond = solverParams.FlowCond
        # ζi = LLSystem.sectionVectors
        vji = TV_influence

        # (u∞ + Σ Gj vji)
        uix = -vji[XDIM, :, :] * Gi .+ FlowCond.uvec[XDIM] # negated...
        uiy = -vji[YDIM, :, :] * Gi .+ FlowCond.uvec[YDIM] # negated...
        uiz = -vji[ZDIM, :, :] * Gi .+ FlowCond.uvec[ZDIM] # negated...


        ui = cat(uix, uiy, uiz, dims=2)
        ui = permutedims(ui, [2, 1])

        ζ = LLSystem.sectionVectors
        # 3d array of ζ
        ζArr = repeat(reshape(ζ, size(ζ)..., 1), 1, 1, size(ζ, 2))
        #   TODO: might come other places too NOTE: Because I use Z as vertical, the influences are negative for ZDIM because the axes point spanwise in the opposite direction
        uxy = -cross.(eachcol(ui), eachcol(ζ))
        uxy = hcat(uxy...) # now it's a (3, npt) matrix
        uxy_norm = .√(uxy[XDIM, :] .^ 2 + uxy[YDIM, :] .^ 2 + uxy[ZDIM, :] .^ 2)

        vxy = cross3D(vji, ζArr)

        # This is downwash contribution
        uxyvxy_xcomp = uxy[XDIM, :] .* vxy[XDIM, :, :]
        uxyvxy_ycomp = uxy[YDIM, :] .* vxy[YDIM, :, :]
        uxyvxy_zcomp = uxy[ZDIM, :] .* vxy[ZDIM, :, :]
        uxzdotvxz = uxyvxy_xcomp .+ uxyvxy_ycomp .+ uxyvxy_zcomp
        numerator = 2.0 * uxzdotvxz .* Gi
        J = numerator ./ uxy_norm .+ 2.0 * diagm(uxy_norm)

        # Along span
        Λ = LLSystem.local_sweeps_ctrl

        _Cs = cos.(Λ)
        _Ss = sin.(Λ)
        αs = atan_cs_safe.(uiz, uix)
        βs = atan_cs_safe.(uiy, uix)
        _aL = atan_cs_safe.(uiz, uix .* _Cs .+ uiy .* _Ss) # GOOD
        _aLMat = reshape(_aL, size(_aL)..., 1)
        _bL = βs .- Λ

        uixMat = reshape(uix, size(uix)..., 1)
        uiyMat = reshape(uiy, size(uiy)..., 1)
        uizMat = reshape(uiz, size(uiz)..., 1)
        _CsMat = reshape(_Cs, size(_Cs)..., 1)
        _SsMat = reshape(_Ss, size(_Ss)..., 1)
        uixviz = uixMat .* (-vji[ZDIM, :, :])
        uizvix = -(uizMat .* vji[XDIM, :, :])
        num_da = uixviz .- uizvix
        denom_da = uixMat .^ 2 .+ uizMat .^ 2
        _da = num_da ./ denom_da

        uixvy = uixMat .* (-vji[YDIM, :, :])
        uiyvx = -uiyMat .* vji[XDIM, :, :]
        num_db = uixvy .- uiyvx
        denom_db = uixMat .^ 2 + uiyMat .^ 2
        _db = num_db ./ denom_db

        uixcos = uixMat .* _CsMat
        uiysin = uiyMat .* _SsMat
        firstTerm_daL = (uixcos .+ uiysin) .* (-vji[ZDIM, :, :])
        uizvixcos = uizvix .* _CsMat
        uizviysin = uizMat .* (-vji[YDIM, :, :]) .* _SsMat
        secondTerm_daL = uizvixcos .+ uizviysin
        denom_daL = uixMat .^ 2 .+ (uixcos .+ uiysin) .^ 2
        _daL = (firstTerm_daL .- secondTerm_daL) ./ denom_daL # OK

        _Ca = cos.(αs)
        _Sa = sin.(αs)
        _Cb = cos.(βs)
        _Sb = sin.(βs)
        SaSquared = _Sa .^ 2
        SbSquared = _Sb .^ 2
        _CaL = cos.(_aL)
        _SaL = sin.(_aL)
        SaLSquared = _SaL .^ 2
        _CbL = cos.(_bL)
        _SbL = sin.(_bL)
        SbLSquared = _SbL .^ 2
        _Rn = .√(_Ca .^ 2 .* _CbL .^ 2 .+ SaSquared .* _Cb .^ 2)
        _Rd = .√(1.0 .- _Sa .^ 2 .* SbSquared)
        iRd = 1.0 ./ _Rd
        iRdSquared = iRd .^ 2
        _RLd = .√(1.0 .- SaLSquared .* SbLSquared)
        _R = _Rn .* iRd
        _RMat = reshape(_R, size(_R)..., 1)
        _RL = _CbL ./ _RLd
        _RLMat = reshape(_RL, size(_RL)..., 1)

        firstTermdR = reshape((_Sa .* _Ca .* (SbSquared .* _Rn .* iRdSquared .+ (_Cb .^ 2 .- _CbL .^ 2) ./ _Rn) ./ _Rd), size(_Sa)..., 1)
        secondTermdR = reshape((SaSquared .* _Sb .* _Cb .* _Rn .* iRdSquared .- (_Ca .^ 2 .* _SbL .* _CbL .+ SaSquared .* _Sb .* _Cb) ./ _Rn) .* iRd, size(_Sa)..., 1)

        _dR = firstTermdR .* _da .+ secondTermdR .* _db # OK

        firstTerm_dRL = reshape(_SaL .* _CaL .* _SbL .^ 2 .* _CbL ./ (_RLd .^ 3), size(_SaL)..., 1)
        secondTerm_dRL = reshape(_CaL .^ 2 .* _SbL ./ (_RLd .^ 3), size(_SaL)..., 1)

        _dRL = firstTerm_dRL .* _daL .- secondTerm_dRL .* _db # OK
        HydroProps = [compute_hydroProperties(sweep, LLHydro.airfoil_xy, LLHydro.airfoil_ctrl_xy)[1] for sweep in Λ]
        _CLa = [HydroProps[ii].airfoil_CLa for (ii, _) in enumerate(Λ)]
        _aL0 = [HydroProps[ii].airfoil_aL0 for (ii, _) in enumerate(Λ)]
        _CLaMat = reshape(_CLa, size(_CLa)..., 1)
        _aL0Mat = reshape(_aL0, size(_aL0)..., 1)

        _dCL = _dR .* _RLMat .* _CLaMat .* (_aLMat .- _aL0Mat) .+ _RMat .* _dRL .* _CLaMat .* (_aLMat .- _aL0Mat) .+ _RMat .* _RLMat .* _CLaMat .* _daL

        J = J .- _dCL
        # println("J: $(J[end,:])") # OK
        # println("\ndCL:")
        # show(stdout, "text/plain", J)
        # println(forceerror)
    elseif mode == "CS" # slow as hell but works
        dh = 1e-100
        ∂r∂G = zeros(DTYPE, length(Gi), length(Gi))

        GiCS = complex(copy(Gi))
        for ii in eachindex(Gi)
            GiCS[ii] += 1im * dh
            r_f = compute_LLresiduals(GiCS; solverParams=solverParams)
            GiCS[ii] -= 1im * dh
            ∂r∂G[:, ii] = imag(r_f) / dh
        end
        J = ∂r∂G
    elseif mode == "FiDi" # currently the best

        # backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))
        # J, = AD.jacobian(backend, x -> compute_LLresiduals(x; solverParams=solverParams), Gi)

        dh = 1e-4
        ∂r∂G = zeros(DTYPE, length(Gi), length(Gi))
        ∂r∂G_z = Zygote.Buffer(∂r∂G)
        r_i = compute_LLresiduals(Gi; solverParams=solverParams)
        for ii in eachindex(Gi)
            @ignore_derivatives(Gi[ii] += dh)
            r_f = compute_LLresiduals(Gi; solverParams=solverParams)
            @ignore_derivatives(Gi[ii] -= dh)
            # ∂r∂G[:, ii] = (r_f - r_i) / dh
            ∂r∂G_z[:, ii] = (r_f - r_i) / dh
        end
        # J = ∂r∂G
        J = copy(∂r∂G_z)
    elseif mode == "RAD" # not working

        backend = AD.ZygoteBackend()
        J, = AD.jacobian(backend, x -> compute_LLresiduals(x; solverParams=solverParams), Gi)

    else
        println("Mode not implemented yet")
    end

    return J
end

function setup_solverparams(xPt, nodeConn, appendageOptions, appendageParams, solverOptions)
    """
    This is a convenience function that sets up the solver parameters for the lifting line algorithm from xPt
    """

    LECoords, TECoords = Utilities.repack_coords(xPt, 3, length(xPt) ÷ 3)
    midchords, chordVec, spanwiseVectors, sweepAng = Preprocessing.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn; appendageOptions=appendageOptions, appendageParams=appendageParams)

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]

    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLSystem, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords;
        npt_wing=npt_wing,
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        # airfoilCoordFile=airfoilCoordFile,
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=@ignore_derivatives(options),
    )

    TV_influence = compute_TVinfluences(FlowCond, LLSystem)

    # --- Pack up parameters for the NL solve ---
    solverParams = LiftingLineNLParams(TV_influence, LLSystem, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    return solverParams, FlowCond
end

function compute_∂cdi∂Γ(Gconv, LLMesh, FlowCond)

    function compute_cdi(Gconv)
        # ---------------------------
        #   Calculate influence matrix
        # ---------------------------
        TV_influence = compute_TVinfluences(FlowCond, LLMesh)

        ux, uy, uz = FlowCond.uvec
        ζi = LLMesh.sectionVectors
        dAi = reshape(LLMesh.sectionAreas, 1, size(LLMesh.sectionAreas)...)


        Gi = reshape(Gconv, 1, size(Gconv)...) # now it's a (1, npt) matrix
        Gjvji = TV_influence .* Gi
        Gjvjix = TV_influence[XDIM, :, :] * Gconv
        Gjvjiy = TV_influence[YDIM, :, :] * Gconv
        Gjvjiz = -TV_influence[ZDIM, :, :] * Gconv
        Gjvji = cat(Gjvjix, Gjvjiy, Gjvjiz, dims=2)
        Gjvji = permutedims(Gjvji, [2, 1])
        u∞ = repeat(reshape(FlowCond.uvec, 3, 1), 1, LLMesh.npt_wing)

        ui = Gjvji .+ u∞ # Local velocities (nondimensional)

        # This is the Biot--Savart law but nondimensional
        # fi = 2 | ( ui ) × ζi| Gi dAi / SRef
        uicrossζi = -cross.(eachcol(ui), eachcol(ζi))
        uicrossζi = hcat(uicrossζi...) # now it's a (3, npt) matrix
        coeff = 2.0 / LLMesh.SRef

        # Integrated = 2 Σ ( u∞ + Gⱼvⱼᵢ ) x ζᵢ * Gᵢ * dAᵢ / SRef
        IntegratedForces = vec(coeff * sum((uicrossζi .* Gi) .* dAi, dims=2))


        # --- Final outputs ---
        CDi = IntegratedForces[XDIM] * ux +
              IntegratedForces[YDIM] * uy +
              IntegratedForces[ZDIM] * uz
        return CDi
    end

    backend = AD.ReverseDiffBackend()
    ∂cdi∂G, = AD.gradient(backend, x -> compute_cdi(x), Gconv)
    ∂cdi∂Γ = ∂cdi∂G / FlowCond.Uinf

    # Compares well with finite difference 2024-12-07
    # backend = AD.FiniteDifferencesBackend(forward_fdm(2, 1))
    # ∂cdi∂G_FD, = AD.gradient(backend, x -> compute_cdi(x), Gconv)
    # println("∂cdi∂Γ: $(∂cdi∂Γ)")
    # println("∂cdi∂Γ_FD: $(∂cdi∂Γ_FD)")

    return ∂cdi∂Γ
end

function compute_∂r∂Γ(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    solverParams, FlowCond = setup_solverparams(ptVec, nodeConn, appendageOptions, appendageParams, solverOptions)

    ∂r∂G = LiftingLine.compute_LLJacobian(Gconv; solverParams=solverParams, mode="CS")
    ∂r∂Γ = ∂r∂G / FlowCond.Uinf

    return ∂r∂Γ
end

function compute_∂r∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    ∂r∂Xpt = zeros(DTYPE, length(Gconv), length(ptVec))

    function compute_resFromXpt(xPt)
        solverParams, _ = setup_solverparams(xPt, nodeConn, appendageOptions, appendageParams, solverOptions)

        resVec = compute_LLresiduals(Gconv; solverParams=solverParams)
        return resVec
    end

    # ************************************************
    #     Finite difference
    # ************************************************
    if uppercase(mode) == "FIDI"
        dh = 1e-5

        resVec_i = compute_resFromXpt(ptVec) # initialize the solver

        # @inbounds begin # no speedup
        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            resVec_f = compute_resFromXpt(ptVec)

            ptVec[ii] -= dh

            ∂r∂Xpt[:, ii] = (resVec_f - resVec_i) / dh
        end
        # end
    elseif uppercase(mode) == "RAD" # This takes nearly 15 seconds compared to 4 sec in pure julia
        # backend = AD.ReverseDiffBackend()
        backend = AD.ZygoteBackend()
        ∂r∂Xpt, = AD.jacobian(backend, x -> compute_resFromXpt(x), ptVec)

    elseif uppercase(mode) == "FAD"
        backend = AD.ForwardDiffBackend()
        ∂r∂Xpt, = AD.jacobian(backend, x -> compute_resFromXpt(x), ptVec)

    end

    return ∂r∂Xpt
end

function compute_∂cdi∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    ∂cdi∂Xpt = zeros(DTYPE, 1, length(ptVec))

    function compute_cdifromxpt(xPt)

        solverParams, FlowCond = setup_solverparams(xPt, nodeConn, appendageOptions, appendageParams, solverOptions)
        TV_influence = solverParams.TV_influence
        LLMesh = solverParams.LLSystem

        ux, uy, uz = FlowCond.uvec
        ζi = LLMesh.sectionVectors
        dAi = reshape(LLMesh.sectionAreas, 1, size(LLMesh.sectionAreas)...)

        Gi = reshape(Gconv, 1, size(Gconv)...) # now it's a (1, npt) matrix
        Gjvji = TV_influence .* Gi
        Gjvjix = TV_influence[XDIM, :, :] * Gconv
        Gjvjiy = TV_influence[YDIM, :, :] * Gconv
        Gjvjiz = -TV_influence[ZDIM, :, :] * Gconv
        Gjvji = cat(Gjvjix, Gjvjiy, Gjvjiz, dims=2)
        Gjvji = permutedims(Gjvji, [2, 1])
        u∞ = repeat(reshape(FlowCond.uvec, 3, 1), 1, LLMesh.npt_wing)

        ui = Gjvji .+ u∞ # Local velocities (nondimensional)

        # This is the Biot--Savart law but nondimensional
        # fi = 2 | ( ui ) × ζi| Gi dAi / SRef
        uicrossζi = -cross.(eachcol(ui), eachcol(ζi))
        uicrossζi = hcat(uicrossζi...) # now it's a (3, npt) matrix
        coeff = 2.0 / LLMesh.SRef

        # Integrated = 2 Σ ( u∞ + Gⱼvⱼᵢ ) x ζᵢ * Gᵢ * dAᵢ / SRef
        IntegratedForces = vec(coeff * sum((uicrossζi .* Gi) .* dAi, dims=2))

        # --- Final outputs ---
        CDi = IntegratedForces[XDIM] * ux +
              IntegratedForces[YDIM] * uy +
              IntegratedForces[ZDIM] * uz

        return CDi
    end
    # ************************************************
    #     Finite difference
    # ************************************************
    if uppercase(mode) == "FIDI"
        dh = 1e-5
        CDi_i = compute_cdifromxpt(ptVec)

        for ii in eachindex(ptVec)
            ptVec[ii] += dh

            CDi_f = compute_cdifromxpt(ptVec)

            ptVec[ii] -= dh

            ∂cdi∂Xpt[1, ii] = (CDi_f - CDi_i) / dh
        end
    elseif uppercase(mode) == "RAD"
        backend = AD.ReverseDiffBackend()
        ∂cdi∂Xpt, = AD.jacobian(backend, x -> compute_cdifromxpt(x), ptVec)
    elseif uppercase(mode) == "FAD"
        backend = AD.ForwardDiffBackend()
        ∂cdi∂Xpt, = AD.jacobian(backend, x -> compute_cdifromxpt(x), ptVec)
    end

    return ∂cdi∂Xpt
end

function compute_straightSemiinfinite(startpt, endvec, pt, rc)
    """
    Compute the influence of a straight semi-infinite vortex filament
    Inputs:
    -------
    startpt : ndarray
        Starting point of the semi-infinite vortex filament
    endvec : Array{Float64, 3}
        Unit vector of the semi-infinite vortex filament
    pt : ndarray
        Point at which the influence is computed (field point)
    rc : scalar
        Vortex core radius (viscous correction)

    Returns:
    --------
    influence : ndarray
        Influence of the semi-infinite vortex filament at the field point. This is everything but the Γ in the induced velocity equation.
    """


    r1 = pt .- startpt
    r1mag = .√(r1[XDIM, :, :] .^ 2 + r1[YDIM, :, :] .^ 2 + r1[ZDIM, :, :] .^ 2)
    uinf = endvec

    r1dotuinf = r1[XDIM, :, :] .* uinf[XDIM, :, :] .+ r1[YDIM, :, :] .* uinf[YDIM, :, :] .+ r1[ZDIM, :, :] .* uinf[ZDIM, :, :]

    r1crossuinf = cross3D(r1, uinf)
    uinfcrossr1 = cross3D(uinf, r1)

    d = .√(r1crossuinf[XDIM, :, :] .^ 2 + r1crossuinf[YDIM, :, :] .^ 2 + r1crossuinf[ZDIM, :, :] .^ 2)
    d = ifelse.(real(r1dotuinf) .< 0.0, r1mag, d)

    # Reshape d to have a singleton dimension for correct broadcasting
    d = reshape(d, 1, size(d)...)

    numerator = uinfcrossr1 .* (d .^ 2 ./ .√(rc^4 .+ d .^ 4))

    denominator = (4π * r1mag .* (r1mag .- r1dotuinf))
    denominator = reshape(denominator, 1, size(denominator)...)

    influence = numerator ./ denominator

    # Replace NaNs and Infs with 0.0
    @ignore_derivatives() do
        influence = replace(influence, NaN => 0.0)
        influence = replace(influence, Inf => 0.0)
        influence = replace(influence, -Inf => 0.0)
    end


    return influence
end

function compute_straightSegment(startpt, endpt, pt, rc)
    """
    Compute the influence of a straight vortex filament segment on a point.

    Parameters
    ----------
    startpt : Array{Float64,3}
        The position vector of the beginning point of the vortex segment ,
        in three dimensions .
    endpt : Array{Float64,3}
        The position vector of the end point of the vortex segment ,
        in three dimensions .
    pt : Array{Float64,3}
        The position vector of the point at which the influence of the
        vortex segment is calculated , in three dimensions .
    rc : Float64
        The radius of the vortex finite core .
    Returns
    -------
    influence : Array{Float64,3}
        The influence of vortex segment at the point, in three dimensions .
    """

    r1 = pt .- startpt
    r1mag = .√(r1[XDIM, :, :] .^ 2 + r1[YDIM, :, :] .^ 2 + r1[ZDIM, :, :] .^ 2)
    r2 = pt .- endpt
    r2mag = .√(r2[XDIM, :, :] .^ 2 + r2[YDIM, :, :] .^ 2 + r2[ZDIM, :, :] .^ 2)
    r1r2 = r1 .- r2

    r1r2mag = .√(r1r2[XDIM, :, :] .^ 2 + r1r2[YDIM, :, :] .^ 2 + r1r2[ZDIM, :, :] .^ 2)

    r1dotr2 = r1[XDIM, :, :] .* r2[XDIM, :, :] + r1[YDIM, :, :] .* r2[YDIM, :, :] + r1[ZDIM, :, :] .* r2[ZDIM, :, :]
    r1dotr1r2 = r1[XDIM, :, :] .* r1r2[XDIM, :, :] + r1[YDIM, :, :] .* r1r2[YDIM, :, :] + r1[ZDIM, :, :] .* r1r2[ZDIM, :, :]
    r2dotr1r2 = r2[XDIM, :, :] .* r1r2[XDIM, :, :] + r2[YDIM, :, :] .* r1r2[YDIM, :, :] + r2[ZDIM, :, :] .* r1r2[ZDIM, :, :]

    r1crossr2 = cross3D(r1, r2)

    d = (r1crossr2[XDIM, :, :] .^ 2 + r1crossr2[YDIM, :, :] .^ 2 + r1crossr2[ZDIM, :, :] .^ 2) ./ r1r2mag
    d = ifelse.(r1dotr1r2 .< 0.0, r1mag, d)
    d = ifelse.(r2dotr1r2 .< 0.0, r2mag, d)

    # Reshape d to have a singleton dimension for correct broadcasting
    d = reshape(d, 1, size(d)...)

    influence = reshape(r1mag .+ r2mag, 1, size(r1mag)...) .* r1crossr2
    influence = influence .* (d .^ 2) ./ .√(rc^4 .+ d .^ 4)
    denominator = (4π * r1mag .* r2mag .* (r1mag .* r2mag .+ r1dotr2))

    # Reshape the denominator to have the same dimensions as the influence    
    denominator = reshape(denominator, 1, size(denominator)...)

    influence = influence ./ denominator

    # Replace NaNs and Infs with 0.0
    @ignore_derivatives() do
        influence = replace(influence, NaN => 0.0)
        influence = replace(influence, Inf => 0.0)
        influence = replace(influence, -Inf => 0.0)
    end

    return influence
end

function compute_hydroProperties(Λ, airfoil_xy_orig, airfoil_ctrl_xy_orig)
    """
    Determines the aerodynamic properties of a swept airfoil 

    Parameters
    ----------
    Λ : scalar , optional
        The local sweep angle of the effective airfoil [rad]
    airfoil_xy_orig : array_like
        The original airfoil coordinates
    airfoil_ctrl_xy_orig : array_like
        The original airfoil control points
    Returns
    -------
    CLa : scalar
        Effective lift slope of the swept airfoil [1/ rad].
    aL0 : scalar
        Effective zero - lift angle of attack of the swept airfoil [rad].
    """


    angles = [deg2rad(-5), 0.0, deg2rad(5)] # three angles to average properties over...
    V1 = compute_vectorFromAngle(angles[1], 0.0, 1.0)
    V2 = compute_vectorFromAngle(angles[2], 0.0, 1.0)
    V3 = compute_vectorFromAngle(angles[3], 0.0, 1.0)

    # println("airfoil x orig:", airfoil_xy_orig[XDIM, :])
    # println("airfoil y orig:", airfoil_xy_orig[YDIM, :])
    airfoil_xy, airfoil_ctrl_xy = compute_scaledAndSweptAirfoilCoords(Λ, airfoil_xy_orig, airfoil_ctrl_xy_orig)

    # --- VPM of airfoil ---
    Airfoil, Airfoil_influences = VPM.setup(airfoil_xy[XDIM, :], airfoil_xy[YDIM, :], airfoil_ctrl_xy, 0.0) # setup with no sweep
    # println("airfoil x:", airfoil_xy[XDIM, :]) #close enough
    # println("airfoil y:", airfoil_xy[YDIM, :]) #close enough
    _, _, Γ1, _ = VPM.solve(Airfoil, Airfoil_influences, V1)
    _, _, Γ2, _ = VPM.solve(Airfoil, Airfoil_influences, V2)
    _, _, Γ3, _ = VPM.solve(Airfoil, Airfoil_influences, V3)
    Γairfoils = [Γ1 Γ2 Γ3]
    Γbar = (Γ1 + Γ2 + Γ3) / 3.0

    airfoil_Γa = (angles[1] * (Γairfoils[1] - Γbar) +
                  angles[2] * (Γairfoils[2] - Γbar) +
                  angles[3] * (Γairfoils[3] - Γbar)) /
                 (angles[1]^2 + angles[2]^2 + angles[3]^2) # this should not be 0.0
    # println("Angles: $(angles)") # correct
    # println("Vectors: $(V1) $(V2) $(V3)")
    # println("Circulation values: $(Γairfoils) ") # close enough
    # println("Γbar: $(Γbar)") # close enough
    # println("Γa: $(airfoil_Γa)") #  close enough

    airfoil_aL0 = -Γbar / airfoil_Γa
    airfoil_CLa = 2.0 * airfoil_Γa / cos(Λ)

    LLHydro = LiftingLineHydro(airfoil_CLa, airfoil_aL0, airfoil_xy, airfoil_ctrl_xy)

    return LLHydro, Airfoil, Airfoil_influences
end

function compute_scaledAndSweptAirfoilCoords(Λ, airfoil_xy, airfoil_ctrl_xy, factor=1.0)
    """
    The effective swept airfoil geometry.
    Parameters
    ----------
    Λ : scalar , optional
        The sweep angle at which the effective airfoil coordinates will be
        calculated ( rad ).
    factor : scalar , optional
        The scaling by which the coordinates are multiplied to match the
        desired chord length .
    airfoil_xy : array_like
        The original airfoil coordinates
    airfoil_ctrl_xy : array_like
        The original airfoil control points
    """
    cosΛ = cos(Λ)

    return factor * transpose(hcat(airfoil_xy[XDIM, :] * cosΛ, airfoil_xy[YDIM, :])),
    factor * transpose(hcat(airfoil_ctrl_xy[XDIM, :] * cosΛ, airfoil_ctrl_xy[YDIM, :]))
end

end