# --- Julia 1.9 ---
"""
@File    :   LiftingLine.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   Modern lifting line from Phillips and Snyder 2000, Reid 2020 appendix
             The major weakness is the discontinuity in the locus of aerodynamic centers
             for a highly swept wing at the root.
             Reid 2020 overcame this using a blending function at the wing root
"""

module LiftingLine

# --- PACKAGES ---
using FLOWMath: abs_cs_safe, atan_cs_safe
using Plots
using LinearAlgebra
# --- DCFoil modules ---
using ..VPM: VPM
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..DCFoil: DTYPE
using ..SolverRoutines: SolverRoutines, compute_anglesFromVector, compute_vectorFromAngle, normalize_3Dvector, cross3D


# ==============================================================================
#                         Structs
# ==============================================================================
struct LiftingLineMesh{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF},TH<:AbstractArray{TF,3}}
    """
    Only geometry and mesh information
    """
    # alphaGeo::TF
    nodePts::TM # LL node points
    collocationPts::TM # Control points
    npt_wing::TI # Number of wing points
    localChords::TA # Local chord lengths [m]
    sectionVectors::TM # Nondimensional section vectors, "dζi" in paper
    sectionLengths::TA # Section lengths
    sectionAreas::TA # Section areas
    HydroProperties # Hydro properties at the cross sections
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
    local_sweep_ctrl::TA
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

struct FlowConditions{TF,TA<:AbstractVector{TF}}
    Uinfvec::TA # Freestream velocity [m/s] [U, V, W]
    Uinf::TF # Freestream velocity magnitude [m/s]
    uvec::TA # Freestream velocity unit vector
    alpha::TF
    beta::TF
end

struct LiftingLineOutputs{TF,TA<:AbstractVector{TF}}
    Γdist::TA # Circulation distribution (Γᵢ) [m^2/s]
    Fdist::TA # Loads distribution vector 
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
    # Stuff for the 2D VPM solve
    Airfoil #
    Airfoil_influences #
end

function setup(Uvec, wingSpan, sweepAng, rootChord, taperRatio;
    npt_wing=99, npt_airfoil=199, blend=0.25, δ=0.15, rc=0.0, airfoil_xy=nothing, airfoil_ctrl_xy=nothing, airfoilCoordFile=nothing, options=nothing)
    """
    Initialize and setup the lifting line model for one wing

    Inputs:
    -------
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
    if airfoilCoordFile != nothing
        println("Reading airfoil coordinates from file")
        airfoil_xy = read_airfoilFile(airfoilCoordFile)
    else
        println("Using provided airfoil coordinates")
    end
    LLHydro, Airfoil, Airfoil_influences = compute_hydroProperties(sweepAng, airfoil_xy, airfoil_ctrl_xy)

    # ************************************************
    #     Preproc stuff
    # ************************************************
    # Blending parameter for the LAC
    σ = 4 * cos(sweepAng)^2 / (blend^2 * wingSpan^2)

    alpha, beta, Uinf = compute_anglesFromVector(Uvec)
    uvec = Uvec / Uinf

    # Wing area
    SRef = rootChord * wingSpan * (1 + taperRatio) * 0.5
    AR = wingSpan^2 / SRef

    # ************************************************
    #     Make wing coordinates
    # ************************************************
    wing_xyz = zeros(3, npt_wing + 1)
    wing_ctrl_xyz = zeros(3, npt_wing)

    # ---------------------------
    #   Y coords (span)
    # ---------------------------
    # Even spacing
    θ_bound = LinRange(-wingSpan / 2, wingSpan / 2, npt_wing * 2 + 1)

    for (ii, θ) in enumerate(θ_bound[1:2:end])
        wing_xyz[YDIM, ii] = θ
    end
    for (ii, θ) in enumerate(θ_bound[2:2:end])
        wing_ctrl_xyz[YDIM, ii] = θ
    end

    # ---------------------------
    #   X coords (chord dist)
    # ---------------------------
    local_chords = zeros(npt_wing + 1)
    local_chords_ctrl = zeros(npt_wing)

    # ∂c/∂y
    local_dchords = zeros(npt_wing + 1)
    local_dchords_ctrl = zeros(npt_wing)

    iTR = 1.0 - taperRatio

    local_chords = rootChord * (1.0 .- 2.0 * iTR * abs_cs_safe.(wing_xyz[YDIM, :]) / wingSpan)
    local_chords_ctrl = rootChord * (1.0 .- 2.0 * iTR * abs_cs_safe.(wing_ctrl_xyz[YDIM, :]) / wingSpan)
    local_dchords = 2.0 * rootChord * (-iTR) * sign.(wing_xyz[YDIM, :]) / wingSpan
    local_dchords_ctrl = 2.0 * rootChord * (-iTR) * sign.(wing_ctrl_xyz[YDIM, :]) / wingSpan

    ZArr = zeros(2, 2, 2)
    ZM = zeros(2, 2)
    ZA = zeros(2)
    LLSystem = LiftingLineMesh(wing_xyz, wing_ctrl_xyz, npt_wing, local_chords, zeros(2, 2), zeros(2), zeros(2), zeros(2),
        npt_airfoil, wingSpan, 0.0, 0.0, AR, rootChord, sweepAng, 0.0, ZArr, ZArr, ZA, ZM, ZA)

    # --- Locus of aerodynamic centers (LAC) ---
    # Default is Küchemann's
    LAC = compute_LAC(LLSystem, LLHydro, wing_xyz[YDIM, :], local_chords, rootChord, sweepAng, wingSpan)
    for (ii, xloc) in enumerate(LAC)
        wing_xyz[XDIM, ii] = xloc
    end
    LAC_ctrl = compute_LAC(LLSystem, LLHydro, wing_ctrl_xyz[YDIM, :], local_chords_ctrl, rootChord, sweepAng, wingSpan)
    for (ii, xloc) in enumerate(LAC_ctrl)
        wing_ctrl_xyz[XDIM, ii] = xloc
    end

    if (!isnothing(options)) && options["make_plot"]
        println("Making plot")
        plot(wing_xyz[XDIM, :], wing_xyz[YDIM, :], label="Wing LAC", marker=:circle)
        plot!(wing_ctrl_xyz[XDIM, :], wing_ctrl_xyz[YDIM, :], label="Control LAC", marker=:cross)
        plot!(xlabel="X", ylabel="Y", title="Locus of Aerodynamic Centers")
        xlims!(0, 1.0)
        savefig("LAC.pdf")
    end
    # Need a mess of LAC's for each control point
    LACeff = compute_LACeffective(LLSystem, LLHydro, wing_xyz[YDIM, :], wing_ctrl_xyz[YDIM, :], local_chords, local_chords_ctrl, local_dchords, local_dchords_ctrl, σ, sweepAng, rootChord, wingSpan)
    # This is a 3D array
    wing_xyz_eff = zeros(3, npt_wing, npt_wing + 1)
    wing_xyz_eff[XDIM, :, :] = LACeff
    wing_xyz_eff[YDIM, :, :] = repeat(transpose(wing_xyz[YDIM, :]), npt_wing, 1)

    # --- Compute local sweeps ---
    # Vectors containing local sweep at each coordinate location in wing_xyz
    fprime = compute_dLACds(LLSystem, LLHydro, wing_xyz[YDIM, :], local_chords, local_dchords, sweepAng, wingSpan)
    localSweeps = -atan_cs_safe.(fprime, ones(size(fprime)))

    fprimeCtrl = compute_dLACds(LLSystem, LLHydro, wing_ctrl_xyz[YDIM, :], local_chords_ctrl, local_dchords_ctrl, sweepAng, wingSpan)
    localSweepsCtrl = -atan_cs_safe.(fprimeCtrl, ones(size(fprimeCtrl)))

    fprimeEff = compute_dLACdseffective(LLSystem, LLHydro, wing_xyz[YDIM, :], wing_ctrl_xyz[YDIM, :], local_chords, local_chords_ctrl, local_dchords, local_dchords_ctrl, σ, sweepAng, rootChord, wingSpan)
    localSweepEff = -atan_cs_safe.(fprimeEff, ones(size(fprimeEff)))

    # --- Other section properties ---
    sectionVectors = wing_xyz[:, 1:end-1] - wing_xyz[:, 2:end] # dℓᵢ

    sectionLengths = sqrt.(sectionVectors[XDIM, :] .^ 2 + sectionVectors[YDIM, :] .^ 2 + sectionVectors[ZDIM, :] .^ 2) # ℓᵢ
    sectionAreas = 0.5 * (local_chords[1:end-1] + local_chords[2:end]) .* abs_cs_safe.(wing_xyz[YDIM, 1:end-1] - wing_xyz[YDIM, 2:end]) # dAᵢ

    ζ = sectionVectors ./ reshape(sectionAreas, 1, length(sectionAreas)) # Normalized section vectors, [3, npt_wing]

    # ---------------------------
    #   Aero section properties
    # ---------------------------
    # Where the 2D VPM comes into play
    aeroProperties = Vector(undef, npt_wing)
    for (ii, sweep) in enumerate(localSweepsCtrl)
        LLHydro = compute_hydroProperties(sweep, airfoil_xy, airfoil_ctrl_xy)
        aeroProperties[ii] = LLHydro
    end

    # ---------------------------
    #   TV joint locations
    # ---------------------------
    # These are where the bound vortex lines kink and then bend to follow the freestream direction
    wing_joint_xyz = zeros(size(wing_xyz))
    wing_joint_xyz_eff = zeros(size(wing_xyz_eff))
    local_chords_colmat = reshape(local_chords, 1, length(local_chords))

    wing_joint_xyz[XDIM, :] = wing_xyz[XDIM, :] + δ * local_chords .* cos.(localSweeps)
    wing_joint_xyz_eff[XDIM, :, :] = wing_xyz_eff[XDIM, :, :] + δ * local_chords_colmat .* cos.(localSweepEff)

    wing_joint_xyz[YDIM, :] = wing_xyz[YDIM, :] + δ * local_chords .* sin.(localSweeps)
    wing_joint_xyz_eff[YDIM, :, :] = transpose(wing_xyz[YDIM, :]) .+ δ * local_chords_colmat .* sin.(localSweepEff)

    # Store all computed quantities here
    LLMesh = LiftingLineMesh(wing_xyz, wing_ctrl_xyz, npt_wing, local_chords, ζ, sectionLengths, sectionAreas, aeroProperties,
        npt_airfoil, wingSpan, SRef, SRef, AR, rootChord, sweepAng, rc, wing_xyz_eff, wing_joint_xyz_eff,
        localSweeps, localSweepEff, localSweepsCtrl)
    FlowCond = FlowConditions(Uvec, Uinf, uvec, alpha, beta)
    return LLMesh, FlowCond, Airfoil, Airfoil_influences
end

function compute_LAC(LLMesh, LLHydro, y, c, cr, Λ, span; model="kuechemann")
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
        Λₖ = Λ / (1.0 + (LLHydro.airfoil_CLa * cos(Λ) / (π * LLMesh.AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LLHydro.airfoil_CLa * cos(Λₖ) / (π * LLMesh.AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        if Λ == 0
            fs = 0.25 * cr .- c * (1.0 - 1.0 / K) / 4.0
        else
            tanl = vec(2π * tan(Λₖ) / (Λₖ * c))
            lam = sqrt.(1.0 .+ (tanl .* y) .^ 2) .-
                  tanl .* abs_cs_safe.(y) .-
                  sqrt.(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2) .+
                  tanl .* (span / 2.0 .- abs_cs_safe.(y))

            fs = 0.25 * cr .+
                 tan(Λ) .* abs_cs_safe.(y) .-
                 c .* (1.0 .- (1.0 .+ 2.0 * lam * Λₖ / π) / K) * 0.25
        end
    else
        println("Model not implemented yet")
    end


    return fs
end

function compute_LACeffective(LLMesh, LLHydro, y, y0, c, c_y0, dc, dc_y0, σ, Λ, cr, span; model="kuechemann")
    """
    The effective LAC , based on Küchemann's equation .

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
    ywork = reshape(y, 1, length(y))
    y0work = reshape(y0, length(y0), 1)
    blend = exp.(-σ * (y0work .- ywork) .^ 2)

    if model == "kuechemann"

        LAC = compute_LAC(LLMesh, LLHydro, y, c, cr, Λ, span)
        LACwork = reshape(LAC, 1, length(LAC))
        LAC0 = compute_LAC(LLMesh, LLHydro, y0, c_y0, cr, Λ, span)
        LAC0work = reshape(LAC0, length(LAC0), 1)
        fprime0 = compute_dLACds(LLMesh, LLHydro, y0, c_y0, dc_y0, Λ, span)

        return (1.0 .- blend) .* LACwork .+
               blend .* (fprime0 .* (ywork .- y0work) .+ LAC0work)
    else
        println("Model not implemented yet")
    end
end

function compute_dLACds(LLMesh, LLHydro, y, c, ∂c∂y, Λ, span; model="kuechemann")
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
        Λₖ = Λ / (1.0 + (LLHydro.airfoil_CLa * cos(Λ) / (π * LLMesh.AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LLHydro.airfoil_CLa * cos(Λₖ) / (π * LLMesh.AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        if Λ == 0
            dx = -∂c∂y * (1.0 - 1.0 / K) * 0.25
        else
            tanl = vec(2π * tan(Λₖ) / (Λₖ * c))
            lam = sqrt.(1.0 .+ (tanl .* y)^2) .-
                  tanl .* abs_cs_safe.(y) .-
                  sqrt.(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2) .+
                  tanl .* (span / 2.0 .- abs_cs_safe.(y))

            lamp = (tanl .^ 2 .* (y .* c - y .^ 2 .* ∂c∂y) ./ c) / sqrt.(1.0 + (tanl .* y) .^ 2) .-
                   tanl .* (sign.(y) * c - abs_cs_safe.(y) .* ∂c∂y) ./ c +
                   (tanl .^ 2 .* (sign.(y) .* (span / 2.0 .- abs_cs_safe.(y)) .* c + ∂c∂y .* (span / 2.0 .- abs_cs_safe.(y)) .^ 2) ./ c) ./
                   sqrt.(1.0 .+ (tanl .* (span / 2.0 .- abs_cs_safe.(y))) .^ 2) .-
                   tanl .* (sign.(y) .* c + (span / 2.0 .- abs_cs_safe.(y)) .* ∂c∂y) ./ c

            dx = tan(Λ) * sign(y) .+
                 lamp .* Λₖ * c / (2π * K) .-
                 ∂c∂y .* (1.0 .- (1.0 .+ 2.0 * lam * Λₖ / π) / K) * 0.25
        end
    else
        println("Model not implemented yet")
    end


    return dx
end

function compute_dLACdseffective(LLMesh, LLHydro, y, y0, c, c_y0, dc, dc_y0, σ, Λ, cr, span; model="kuechemann")
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

        LAC = compute_LAC(LLMesh, LLHydro, y, c, cr, Λ, span)
        LACwork = reshape(LAC, 1, length(LAC))
        fprime = compute_dLACds(LLMesh, LLHydro, y, c, dc, Λ, span)
        fprimework = reshape(fprime, 1, length(fprime))
        LAC0 = compute_LAC(LLMesh, LLHydro, y0, c_y0, cr, Λ, span)
        LAC0work = reshape(LAC0, length(LAC0), 1)
        fprime0 = compute_dLACds(LLMesh, LLHydro, y0, c_y0, dc_y0, Λ, span)
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

function solve(FlowCond, LLMesh, LLHydro, Airfoil, Airfoil_influences)
    """
    Execute LL algorithm
    Inputs:
    -------
    LiftingSystem : LiftingLineSystem
        Lifting line system struct with all necessary parameters

    Returns:
    --------
    LLResults : LiftingLineResults
        Lifting line results struct with all necessary parameters
    """

    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    uinf = reshape(FlowCond.uvec, 3, 1, 1)
    uinfMat = repeat(uinf, 1, LLMesh.npt_wing, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)

    P1 = LLMesh.wing_joint_xyz_eff[:, :, 2:end]
    P2 = LLMesh.wing_xyz_eff[:, :, 2:end]
    P3 = LLMesh.wing_xyz_eff[:, :, 1:end-1]
    P4 = LLMesh.wing_joint_xyz_eff[:, :, 1:end-1]

    ctrlPts = reshape(LLMesh.collocationPts, 3, LLMesh.npt_wing, 1)
    ctrlPtMat = repeat(ctrlPts, 1, 1, LLMesh.npt_wing) # end up with size (3, npt_wing, npt_wing)

    bound_mask = ones(LLMesh.npt_wing, LLMesh.npt_wing) - diagm(ones(LLMesh.npt_wing))

    influence_semiinfa = compute_straightSemiinfinite(P1, uinfMat, ctrlPtMat, LLMesh.rc)
    influence_straightsega = compute_straightSegment(P1, P2, ctrlPtMat, LLMesh.rc)
    influence_straightsegb = compute_straightSegment(P3, P4, ctrlPtMat, LLMesh.rc)
    influence_semiinfb = compute_straightSemiinfinite(P4, uinfMat, ctrlPtMat, LLMesh.rc)
    TV_influence = -influence_semiinfa +
                   influence_straightsega +
                   influence_straightsegb +
                   influence_semiinfb

    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    # First guess
    c_r = LLMesh.rootChord
    # TODO: WEIRD BUG HERE
    clα = LLHydro.airfoil_CLa
    αL0 = LLHydro.airfoil_aL0
    Λ = LLMesh.sweepAng
    Ux, _, Uz = FlowCond.Uinfvec
    span = LLMesh.span
    ctrl_pts = LLMesh.collocationPts
    g0 = 0.5 * c_r * clα * cos(Λ) * (Uz / Ux - αL0) *
         (1.0 .- (2.0 * ctrl_pts[YDIM, :] / span) .^ 4) .^ (0.25)

    # --- Pack up parameters for the NL solve ---
    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, Airfoil, Airfoil_influences)

    # --- Nonlinear solve for circulation distribution ---
    Gconv, residuals = SolverRoutines.converge_resNonlinear(compute_LLresiduals, compute_LLJacobian, g0; solverParams=LLNLParams)

    Gjvji = TV_influence * Gconv
    uvecmat = repeat(reshape(FlowCond.uvec, 3, 1), 1, LLMesh.npt_wing)

    # This is the Biot--Savart law but nondimensional
    Forces = 2.0 * cross(Gjvji + uvecmat, LLMesh.sectionVectors) * Gconv * LLMesh.sectionAreas / LLMesh.SRef
    # Integrated = 2 Σ ( uvec + Gⱼvⱼᵢ ) x ζᵢ * Gᵢ * dAᵢ / SRef
    IntegratedForces = 2.0 * sum(cross(Gjvji + uvecmat, LLMesh.zeta) *
                                 Gconv * LLMesh.sectionAreas / LLMesh.SRef, dims=2)

    Γdist = Gconv * FlowCond.Uinfvec

    # --- Vortex core viscous correction ---
    if LLMesh.rc != 0
        println("Vortex core viscous correction not implemented yet")
    end

    # --- Final outputs ---
    ux, uy, uz = FlowCond.uvec

    CL = -Forces[XDIM] * uz +
         Forces[ZDIM] * ux / (ux^2 + uz^2)
    CDi = Forces[XDIM] * ux +
          Forces[YDIM] * uy +
          Forces[ZDIM] * uz
    CS = (
        -Forces[XDIM] * ux * uy -
        Forces[ZDIM] * uz * uy +
        Forces[YDIM] * (uz^2 + ux^2)
    ) /
         sqrt(ux^2 * uy^2 + uz^2 * uy^2 + (uz^2 + ux^2)^2)

    LLResults = LiftingLineOutputs(Forces, Γdist, IntegratedForces, CL, CDi, CS)

    return LLResults
end

function compute_LLresiduals(G; solverParams=nothing)
    """
    Nonlinear , nondimensional lifting - line equation .
    Parameters
    ----------
    G : array_like
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
    Airfoil = solverParams.Airfoil
    Airfoil_influences = solverParams.Airfoil_influences

    _Vi = TV_influence * G .+ transpose(LLSystem.uvec)

    # Do a curve fit on aero props
    # if self._aero_approx:
    # _CL = self._lift_from_aero(*self._aero_properties, self.local_sweep_ctrl, self.Vinf * _Vi, self.Vinf)
    # else:
    # Actually solve VPM
    _CL = [VPM.solve(Airfoil, Airfoil_influences, V_local) for V_local in FlowCond.Uinfvec * transpose(_Vi)]

    u∞_plus_ΣGjvji_cross_ζi = cross(_Vi, LLSystem.sectionVectors)
    u∞_plus_ΣGjvji_cross_ζi_mag = sqrt.(u∞_plus_ΣGjvji_cross_ζi[XDIM, :] .^ 2 + u∞_plus_ΣGjvji_cross_ζi[YDIM, :] .^ 2, u∞_plus_ΣGjvji_cross_ζi[ZDIM, :] .^ 2)

    _dF = 2.0 * u∞_plus_ΣGjvji_cross_ζi_mag * G

    return _dF - _CL
end

function compute_LLJacobian(LLHydro, FlowCond, LLSystem, Gi, TV_influence)
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
    vji = TV_influence

    # (u∞ + Σ Gj vji)
    u = FlowCond.uvec .+ vji * Gi
    ux, uy, uz = u
    # zetaArr = repeat(reshape(LLSystem.zeta, 3, length(LLSystem.zeta), 1), 1, 1, length(LLSystem.zeta))

    uxy = cross.(u, LLSystem.zeta)
    uxy_norm = sqrt.(uxy[XDIM, :] .^ 2 + uxy[YDIM, :] .^ 2 + uxy[ZDIM, :] .^ 2)
    vxy = cross.(vji, LLSystem.zeta)
    # This is downwash contribution
    J = (2.0 * (transpose(uxy[XDIM, :]) * vxy[XDIM, :, :] .+ transpose(uxy[YDIM, :]) * vxy[YDIM, :, :] + transpose(uxy[ZDIM, :]) * vxy[ZDIM, :, :])
         * transpose(Gi)
         /
         transpose(uxy_norm)
    )
    J .+= 2.0 * diagm(uxy_norm)
    _CLa, _aL0 = LLHydro.CLa, LLHydro.aL0
    Λ = HydroLL.local_sweeps_ctrl
    _Cs = cos.(Λ)
    _Ss = sin.(Λ)
    αs = atan_cs_safe.(uz, ux)
    βs = atan_cs_safe.(uy, ux)
    _aL = atan_cs_safe.(uz, ux * _Cs + uy * _Ss)
    _bL = βs - Λ
    _da = (transpose(ux) * vji[ZDIM, :, :] - transpose(uz) * vji[XDIM, :, :]) / (
        transpose(ux) .^ 2 + transpose(uz) .^ 2)
    _db = (transpose(ux) * vji[YDIM, :, :] - transpose(uy) * vji[XDIM, :, :]) / (
        transpose(ux) .^ 2 + transpose(uy) .^ 2)
    _daL = (
        (transpose(ux) * transpose(_Cs) + transpose(uy) * transpose(_Ss)) * vji[ZDIM, :, :]
        -
        transpose(uz) * (vji[XDIM, :, :] * transpose(_Cs) + vji[YDIM, :, :] * transpose(_Ss))
    ) / (transpose(ux) .^ 2 + (transpose(ux) * transpose(_Cs) + transpose(uy) * transpose(_Ss)) .^ 2)
    _Ca = cos(αs)
    _Sa = sin(αs)
    _Cb = cos(βs)
    _Sb = sin(βs)
    _CaL = cos(_aL)
    _SaL = sin(_aL)
    _CbL = cos(_bL)
    _SbL = sin(_bL)
    _Rn = sqrt.(_Ca^2 * _CbL^2 + _Sa^2 * _Cb^2)
    _Rd = sqrt(1.0 - _Sa^2 * _Sb^2)
    _RLd = sqrt(1.0 - _SaL^2 * _SbL^2)
    _R = _Rn / _Rd
    _RL = _CbL / _RLd
    _dR = transpose(_Sa * _Ca * (_Sb^2 * _Rn / (_Rd^2) + (_Cb^2 - _CbL^2) / _Rn) / _Rd) * _da +
          transpose((_Sa^2 * _Sb * _Cb * _Rn / (_Rd^2) - (_Ca^2 * _SbL * _CbL + _Sa^2 * _Sb * _Cb) / _Rn) / _Rd) * _db
    _dRL = transpose(_SaL * _CaL * _SbL^2 * _CbL / (_RLd^3)) * _daL -
           transpose(_CaL^2 * _SbL / (_RLd^3)) * _db
    _dCL = _dR * transpose(_RL) * transpose(_CLa) * (transpose(_aL) - transpose(_aL0))
    +transpose(_R) * _dRL * tranpose(_CLa) * (transpose(_aL) - transpose(_aL0))
    +transpose(_R) * transpose(_RL) * transpose(_CLa) * _daL
    J -= _dCL
    return J
end

function compute_straightSemiinfinite(startpt, endvec, pt, rc)
    """
    Compute the influence of a straight semi-infinite vortex filament
    Inputs:
    -------
    startpt : ndarray
        Starting point of the semi-infinite vortex filament
    endvec : ndarray
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
    r1 = pt - startpt
    r1mag = sqrt.(r1[XDIM, :, :] .^ 2 + r1[YDIM, :, :] .^ 2 + r1[ZDIM, :, :] .^ 2)
    uinf = endvec

    # TODO: This might be wrong
    r1dotuinf = r1[XDIM, :, :] * uinf[XDIM, :, :] + r1[YDIM, :, :] * uinf[YDIM, :, :] + r1[ZDIM, :, :] * uinf[ZDIM, :, :]

    r1crossuinf = cross3D(r1, uinf)
    uinfcrossr1 = cross3D(uinf, r1)

    d = sqrt.(r1crossuinf[XDIM, :, :] .^ 2 + r1crossuinf[YDIM, :, :] .^ 2 + r1crossuinf[ZDIM, :, :] .^ 2)

    d = ifelse.(r1dotuinf .< 0.0, r1mag, d)

    numx = uinfcrossr1[XDIM, :, :] * (d .^ 2 ./ sqrt.(rc^4 .+ d .^ 4))
    numy = uinfcrossr1[YDIM, :, :] * (d .^ 2 ./ sqrt.(rc^4 .+ d .^ 4))
    numz = uinfcrossr1[ZDIM, :, :] * (d .^ 2 ./ sqrt.(rc^4 .+ d .^ 4))

    denominator = (4π * r1mag * (r1mag .- r1dotuinf))

    infx = numx ./ denominator
    infy = numy ./ denominator
    infz = numz ./ denominator

    influence = cat(infx, infy, infz, dims=3)
    influence = permutedims(influence, [3, 1, 2])

    # influence = replace.(influence, NaN => 0.0)
    return influence
end

function compute_straightSegment(startpt, endpt, pt, rc)
    """
    Compute the influence of a straight vortex filament segment on a point.

    Parameters
    ----------
    startpt : array_like
        The position vector of the beginning point of the vortex segment ,
        in three dimensions .
    endpt : array_like
        The position vector of the end point of the vortex segment ,
        in three dimensions .
    pt : array_like
        The position vector of the point at which the influence of the
        vortex segment is calculated , in three dimensions .
    rc : scalar
        The radius of the vortex finite core .
    Returns
    -------
    influence : array_like
        The influence of vortex segment at the point, in three dimensions .
    """
    r1 = pt - startpt
    r1mag = sqrt.(r1[XDIM, :, :] .^ 2 + r1[YDIM, :, :] .^ 2 + r1[ZDIM, :, :] .^ 2)
    r2 = pt - endpt
    r2mag = sqrt.(r2[XDIM, :, :] .^ 2 + r2[YDIM, :, :] .^ 2 + r2[ZDIM, :, :] .^ 2)
    r1r2 = r1 - r2
    r1r2mag = sqrt.(r1r2[XDIM, :, :] .^ 2 + r1r2[YDIM, :, :] .^ 2 + r1r2[ZDIM, :, :] .^ 2)
    r1dotr2 = r1[XDIM, :, :] .* r2[XDIM, :, :] + r1[YDIM, :, :] .* r2[YDIM, :, :] + r1[ZDIM, :, :] .* r2[ZDIM, :, :]
    r1dotr1r2 = r1[XDIM, :, :] .* r1r2[XDIM, :, :] + r1[YDIM, :, :] .* r1r2[YDIM, :, :] + r1[ZDIM, :, :] .* r1r2[ZDIM, :, :]
    r2dotr1r2 = r2[XDIM, :, :] .* r1r2[XDIM, :, :] + r2[YDIM, :, :] .* r1r2[YDIM, :, :] + r2[ZDIM, :, :] .* r1r2[ZDIM, :, :]

    r1crossr2 = cross3D(r1, r2)

    d = (r1crossr2[XDIM, :, :] .^ 2 + r1crossr2[YDIM, :, :] .^ 2 + r1crossr2[ZDIM, :, :] .^ 2) ./ r1r2mag
    d = ifelse.(r1dotr1r2 .< 0.0, r1mag, d)
    d = ifelse.(r2dotr1r2 .< 0.0, r2mag, d)

    termx = r1crossr2[XDIM, :, :] .* d .^ 2 / sqrt.(rc^4 .+ d .^ 4) /
            (4π * r1mag .* r2mag * (r1mag .* r2mag + r1dotr2))
    termy = r1crossr2[YDIM, :, :] .* d .^ 2 / sqrt.(rc^4 .+ d .^ 4) /
            (4π * r1mag .* r2mag * (r1mag .* r2mag + r1dotr2))
    termz = r1crossr2[ZDIM, :, :] .* d .^ 2 / sqrt.(rc^4 .+ d .^ 4) /
            (4π * r1mag .* r2mag * (r1mag .* r2mag + r1dotr2))
    secondterm = cat(termx, termy, termz, dims=3)
    secondterm = permutedims(secondterm, [3, 1, 2])

    influence = reshape(r1mag + r2mag, 1, size(r1mag, 1), size(r2mag, 2)) .* secondterm

    # influence = replace.(influence, NaN => 0.0)
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

    airfoil_xy, airfoil_ctrl_xy = compute_scaledAndSweptAirfoilCoords(Λ, airfoil_xy_orig, airfoil_ctrl_xy_orig)

    # --- VPM of airfoil ---
    Airfoil, Airfoil_influences = VPM.setup(airfoil_xy[XDIM, :], airfoil_xy[YDIM, :], airfoil_ctrl_xy, 0.0) # setup with no sweep
    _, _, Γ1, _ = VPM.solve(Airfoil, Airfoil_influences, V1)
    _, _, Γ2, _ = VPM.solve(Airfoil, Airfoil_influences, V2)
    _, _, Γ3, _ = VPM.solve(Airfoil, Airfoil_influences, V3)

    Γairfoils = [Γ1, Γ2, Γ3]
    Γbar = (Γ1 + Γ2 + Γ3) / 3.0

    airfoil_Γa = (angles[1] * (Γairfoils[1] - Γbar) +
                  angles[2] * (Γairfoils[2] - Γbar) +
                  angles[3] * (Γairfoils[3] - Γbar)) /
                 (angles[1]^2 + angles[2]^2 + angles[3]^2)

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