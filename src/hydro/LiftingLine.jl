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
# --- DCFoil modules ---
using ..VPM: VPM
using ..SolutionConstants: XDIM, YDIM, ZDIM
using ..DCFoil: DTYPE
using ..SolverRoutines: compute_anglesFromVector, normalize_3Dvector


# ==============================================================================
#                         Structs
# ==============================================================================
struct LiftingLineMesh{TF,TI,TA<:AbstractVector{TF},TM<:AbstractMatrix{TF}}
    # alphaGeo::TF
    nodePts::TM # LL node points
    collocationPts::TM # Control points
    localChords::TA # Local chord lengths [m]
    sectionVectors::TM # Section vectors, "dℓᵢ" in paper
    sectionLengths::TA # Section lengths
    sectionAreas::TA # Section areas
    npt_airfoil::TI # Number of airfoil points
    planformArea::TF
    SRef::TF # Reference area [m^2]
    AR::TF # Aspect ratio
    rootChord::TF # Root chord [m]
    sweepAng::TF # Wing sweep angle [rad]
    rc::TF # Finite-core vortex radius (viscous correction) [m]
end

struct FlowConditions{TF,TA<:AbstractVector{TF}}
    Uinfvec::TA # Freestream velocity [m/s] [U, V, W]
    Uinf::TF # Freestream velocity magnitude [m/s]
    uvec::TA # Freestream velocity unit vector
    alpha::TF
    beta::TF
end

struct LiftingLineOutputs{TF,TA<:AbstractVector{TF}}
    F::TA # Total integrated loads vector [Fx, Fy, Fz, Mx, My, Mz]
    Γdist::TA # Circulation distribution (Γᵢ) [m^2/s]
    CL::TF # Lift coefficient (perpendicular to freestream in symmetry plane)
    CDi::TF # Induced drag coefficient (aligned w/ freestream)
    CS::TF # Side force coefficient
end

function setup(Uvec, wingSpan, sweepAng, rootChord, taperRatio;
    npt_wing=99, npt_airfoil=199, blend=0.25, airfoilCoords="input.dat")
    """
    Initialize and setup the lifting line model

    Inputs:
    -------
    sweepAng : scalar
            The wing sweep angle in rad.
    blend : scalar , optional
            The normalized blending distance, used to calculate the
            effective loci of aerodynamic centers.
    """

    # Blending parameter for the LAC
    σ = 4 * cos(sweepAng)^2 / (blend^2 * wingSpan^2)

    Uinf = sqrt(Uvec[XDIM]^2 + Uvec[YDIM]^2 + Uvec[ZDIM]^2)
    uvec = Uvec / Uinf

    # Wing area
    SRef = rootChord * wingSpan * (1 + taperRatio) * 0.5
    AR = wingSpan^2 / SRef

    # ************************************************
    #     Make wing coordinates
    # ************************************************
    wing_xyz = zeros(npt_wing + 1, 3)
    wing_ctrl_xyz = zeros(npt_wing, 3)

    # ---------------------------
    #   Y coords (span)
    # ---------------------------
    # Even spacing
    θ_bound = LinRange(-wingSpan / 2, wingSpan / 2, npt_wing * 2 + 1)

    wing_xyz[:, YDIM] = θ_bound[1:2:end]
    # TODO: fix spacing later
    wing_ctrl_xyz[:, YDIM] = θ_bound[2:2:end]

    # ---------------------------
    #   X coords (chord dist)
    # ---------------------------
    local_chords = zeros(npt_wing + 1)
    local_chords_ctrl = zeros(npt_wing)
    local_dchords = zeros(npt_wing + 1)
    local_dchords_ctrl = zeros(npt_wing)
    local_chords = rootChord * (1.0 - 2.0 * (1.0 - taperRatio) * abs_cs_safe(wing_xyz[YDIM, :]) / wingSpan)
    local_chords_ctrl = rootChord * (1.0 - 2.0 * (1.0 - taperRatio) * abs_cs_safe(self.wing_ctrl_xyz[YDIM, :]) / wingSpan)
    local_dchords = 2.0 * rootChord * (taperRatio - 1.0) * sign(wing_xyz[YDIM, :]) / wingSpan
    local_dchords_ctrl = 2.0 * rootChord * (taperRatio - 1.0) * sign(wing_ctrl_xyz[YDIM, :]) / wingSpan
    LLSystem = LiftingLineMesh(wing_xyz, wing_ctrl_xyz, local_chords, zeros(1, 1), zeros(1, 1), zeros(1, 1), npt_airfoil, 0.0, 0.0, AR, rootChord, sweepAng, 0.0)

    # --- Locus of aerodynamic centers (LAC) ---
    # Default is Küchemann's
    wing_xyz_eff = compute_LAC(LLSystem, wing_xyz[YDIM], local_chords, rootChord, sweepAng, wingSpan)

    # --- Compute local sweeps ---
    fprime = compute_dLACds(LLSystem, wing_xyz[YDIM], local_chords, local_dchords, sweepAng, wingSpan)
    fprime = compute_dLACds(LLSystem, wing_xyz[YDIM], local_chords, local_dchords, sweepAng, wingSpan)
    fprime = compute_dLACds(LLSystem, wing_xyz[YDIM], local_chords, local_dchords, sweepAng, wingSpan)
    localSweeps = -atan_cs_safe()
    localSweepsCtrl = -atan_cs_safe()

    localSweepEff = -atan_cs_safe()

    # --- Other section properties ---
    sectionVectors = wing_xyz[1:end-1, :] - wing_xyz[2:end, :] # dℓᵢ
    sectionLengths = sqrt(sum(sectionVectors .^ 2, dims=2))
    sectionAreas = 0.5 * (local_chords[1:end-1] + local_chords[2:end]) * abs_cs_safe(wing_xyz[1:end-1, YDIM] - wing_xyz[2:end, YDIM]) # dAᵢ

    ζ = sectionVectors ./ sectionAreas # Normalized section vectors

    # ---------------------------
    #   Aero section properties
    # ---------------------------
    # Where the 2D VPM comes into play
    for sweep in localSweepsCtrl
        matrix = compute_aeroProperties(sweep)
        aeroProperties[] = transpose(matrix)
    end

    # ---------------------------
    #   TV joint locations
    # ---------------------------
    # These are where the bound vortex lines kink and then bend to follow the freestream direction
    wing_joint_xyz = zeros(npt_wing + 1, 3)
    wing_joint_xyz_eff = zeros(npt_wing + 1, npt_wing, 3)

    wing_joint_xyz[:, XDIM] = wing_xyz[:, XDIM] + delta * local_chords * cos(localSweeps)
    wing_joint_xyz_eff[:, :, XDIM] = wing_xyz_eff[:, :, XDIM] + delta * local_chords * cos(local_sweep_eff)

    wing_joint_xyz[:, YDIM] = wing_xyz[:, YDIM] + delta * local_chords * sin(localSweeps)
    wing_joint_xyz_eff[:, :, YDIM] = wing_xyz[:, None, YDIM] + delta * local_chords * sin(local_sweep_eff)


    # Store all computed quantities here
    LLMesh = LiftingLineMesh(wing_xyz, wing_ctrl_xyz, local_chords, sectionVectors, sectionLengths, sectionAreas, npt_airfoil, planformArea, SRef, AR, rootChord, sweepAng, rc)
    FlowCond = FlowConditions(Uvec, Uinf, uvec, alpha, beta)
    return LLMesh, FlowCond
end

function compute_LAC(LiftingSystem, y, c, cr, Λ, span; model="kuechemann")
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
        Λₖ = Λ / (1.0 + (LiftingSystem.airfoil_CLa * cos(Λ) / (π * LiftingSystem.AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LiftingSystem.airfoil_CLa * cos(Λₖ) / (π * LiftingSystem.AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        tanl = 2π * tan(Λₖ) / (Λₖ * c)
        lam = sqrt(1.0 + (tanl * y)^2) -
              tanl * abs_cs_safe(y) -
              sqrt(1.0 + (tanl * (span / 2.0 - abs_cs_safe(y)))^2)
        +tanl * (span / 2.0 - abs_cs_safe(y))

        fs = 0.25 * cr +
             tan(Λ) * abs_cs_safe(y) -
             c * (1.0 - (1.0 + 2.0 * lam * Λₖ / π) / K) * 0.25
    else
        println("Model not implemented yet")
    end


    return fs
end

function compute_dLACds(LiftingSystem, y, c, ∂c∂y, Λ, span; model="kuechemann")
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
        Λₖ = Λ / (1.0 + (LiftingSystem.airfoil_CLa * cos(Λ) / (π * LiftingSystem.AR))^2)^(0.25) # aspect ratio effect
        K = (1.0 + (LiftingSystem.airfoil_CLa * cos(Λₖ) / (π * LiftingSystem.AR))^2)^(π / (4.0 * (π + 2 * abs_cs_safe(Λₖ))))

        tanl = 2π * tan(Λₖ) / (Λₖ * c)
        lam = sqrt(1.0 + (tanl * y)^2) -
              tanl * abs_cs_safe(y) -
              sqrt(1.0 + (tanl * (span / 2.0 - abs_cs_safe(y)))^2)
        +tanl * (span / 2.0 - abs_cs_safe(y))

        lamp = (tanl * tanl * (y * c - y * y * ∂c∂y) / c) / sqrt(1.0 + (tanl * y)^2) -
               tanl * (sign(y) * c - abs_cs_safe(y) * ∂c∂y) / c +
               (tanl * tanl * (sign(y) * (b / 2.0 - abs_cs_safe(y)) * c + ∂c∂y * (b / 2.0 - abs_cs_safe(y))^2) / c) /
               sqrt(1.0 + (tanl * (b / 2.0 - abs_cs_safe(y)))^2) -
               tanl * (sign(y) * c + (b / 2.0 - abs_cs_safe(y)) * ∂c∂y) / c

        dx = tan(Λ) * sign(y)
        +lamp * Λₖ * c / (2π * K)
        -∂c∂y * (1.0 - (1.0 + 2.0 * lam * Λₖ / π) / K) * 0.25
    else
        println("Model not implemented yet")
    end


    return dx
end

function solve(FlowConditions, LiftingSystem)
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
    uinf = reshape(FlowConditions.uvec, 3, 1, 1)
    uinfMat = repeat(uinf, 1, LiftingSystem.npt_wing, LiftingSystem.npt_wing) # end up with size (3, npt_wing, npt_wing)

    P1 = LiftingSystem.wing_joint_xyz_eff[]
    P2 = LiftingSystem.wing_xyz_eff[]
    P3 = LiftingSystem.wing_xyz_eff[]
    P4 = LiftingSystem.wing_joint_xyz_eff[]

    ctrlPts = reshape(LiftingSystem.collocationPts, 3, 1, LiftingSystem.npt_wing)
    ctrlPtMat = repeat(ctrlPts, 1, LiftingLineSystem.npt_wing, LiftingSystem.npt_wing) # end up with size (3, npt_wing, npt_wing)

    bound_mask = ones(LiftingSystem.npt_wing, LiftingSystem.npt_wing) - diagm(ones(LiftingSystem.npt_wing))

    TV_influence = -compute_straightSemiinfinite(P1, uinfMat, ctrlPtMat, LiftingSystem.rc)
    +compute_straightSegment(P1, P2, ctrlPtMat, LiftingSystem.rc)
    +compute_straightSegment(P3, P4, ctrlPtMat, LiftingSystem.rc)
    +compute_straightSemiinfinite(P4, uinfMat, ctrlPtMat, LiftingSystem.rc)

    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    # First guess
    g0 = 0.5 * LiftingSystem.rootChord * LiftingSystem.airfoil_CLa * cos(LiftingSystem.sweepAng) * (FlowConditions.Uinf[ZDIM] / FlowConditions.Uinf[XDIM] - LiftingSystem.airfoil_aL0) * (1 - (2.0 * LiftingSystem.wing_ctrl_xyz[YDIM, :] / LiftingSystem.span)^4)^(1 / 4)

    # Solve for circulation distribution
    Gconv, residuals = SolverRoutines.converge_r(compute_LLresiduals, compute_LLJacobian, g0)

    Gjvji = TV_influence * Gconv
    uvecmat = repeat(reshape(FlowConditions.uvec, 3, 1), 1, LiftingSystem.npt_wing)

    # This is the Biot--Savart law but nondimensional
    # Fvec = 2 Σ ( uvec + Gⱼvⱼᵢ ) x ζᵢ * Gᵢ * dAᵢ / SRef
    Forces = 2.0 * sum(cross(Gjvji + uvecmat, LiftingSystem.zeta) *
                       Gconv * LiftingSystem.sectionAreas / LiftingSystem.SRef, dims=2)

    Γdist = Gconv * FlowConditions.Uinfvec

    # --- Vortex core viscous correction ---
    if LiftingSystem.rc != 0
        println("Vortex core viscous correction not implemented yet")
    end

    # --- Final outputs ---
    ux, uy, uz = FlowConditions.uvec
    CL = -Forces[XDIM] * uz + Forces[ZDIM] * ux / (ux^2 + uz^2)
    CDi = Forces[XDIM] * ux + Forces[YDIM] * uy + Forces[ZDIM] * uz
    CS = (-Forces[XDIM] * ux * uy - Forces[ZDIM] * uz * uy + Forces[YDIM] * (uz^2 + ux^2)) / sqrt(ux^2 * uy^2 + uz^2 * uy^2 + (uz^2 + ux^2)^2)

    LLResults = LiftingLineOutputs(Forces, Γdist, CL, CDi, CS)

    return LLResults
end

function compute_LLresiduals()

end

function compute_LLJacobian(Gi)
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
    # _v = TV_influence
    # _u = _v * Gi + self.u[:, None]
    # _ux, _uy, _uz = _u
    # _uxz = cross(_u, self._zeta, axis=0)
    # _uxz_norm = norm(_uxz, axis=0)
    # _vxz = cross(_v, self._zeta[:, :, None], axis=0)
    # J = (
    #     2.0
    #     * (_uxz[0, :, None] * _vxz[0, :, :] + _uxz[1, :, None] * _vxz[1, :, :] + _uxz[2, :, None] * _vxz[2, :, :])
    #     * Gi[:, None]
    #     / _uxz_norm[:, None]
    # )
    # J += 2.0 * np.diag(_uxz_norm)
    # _CLa, _aL0 = self._aero_properties
    # _sweep = self.local_sweep_ctrl
    # _Cs = np.cos(_sweep)
    # _Ss = np.sin(_sweep)
    # _a = np.arctan2(_uy, _ux)
    # _b = np.arctan2(_uz, _ux)
    # _aL = np.arctan2(_uy, _ux * _Cs + _uz * _Ss)
    # _bL = _b - _sweep
    # _da = (_ux[:, None] * _v[1, :, :] - _uy[:, None] * _v[0, :, :]) / (
    #     _ux[:, None] * _ux[:, None] + _uy[:, None] * _uy[:, None]
    # )
    # _db = (_ux[:, None] * _v[2, :, :] - _uz[:, None] * _v[0, :, :]) / (
    #     _ux[:, None] * _ux[:, None] + _uz[:, None] * _uz[:, None]
    # )
    # _daL = (
    #     (_ux[:, None] * _Cs[:, None] + _uz[:, None] * _Ss[:, None]) * _v[1, :, :]
    #     - _uy[:, None] * (_v[0, :, :] * _Cs[:, None] + _v[2, :, :] * _Ss[:, None])
    # ) / (_ux[:, None] * _ux[:, None] + (_ux[:, None] * _Cs[:, None] + _uz[:, None] * _Ss[:, None]) ^ 2)
    # _Ca = cos(_a)
    # _Sa = sin(_a)
    # _Cb = cos(_b)
    # _Sb = sin(_b)
    # _CaL = np.cos(_aL)
    # _SaL = np.sin(_aL)
    # _CbL = np.cos(_bL)
    # _SbL = np.sin(_bL)
    # _Rn = np.sqrt(_Ca * _Ca * _CbL * _CbL + _Sa * _Sa * _Cb * _Cb)
    # _Rd = np.sqrt(1.0 - _Sa * _Sa * _Sb * _Sb)
    # _RLd = np.sqrt(1.0 - _SaL * _SaL * _SbL * _SbL)
    # _R = _Rn / _Rd
    # _RL = _CbL / _RLd
    # _dR = (_Sa * _Ca * (_Sb * _Sb * _Rn / (_Rd * _Rd) + (_Cb * _Cb - _CbL * _CbL) / _Rn) / _Rd)[:, None] * _da + (
    #     (_Sa * _Sa * _Sb * _Cb * _Rn / (_Rd * _Rd) - (_Ca * _Ca * _SbL * _CbL + _Sa * _Sa * _Sb * _Cb) / _Rn) / _Rd
    # )[:, None] * _db
    # _dRL = (_SaL * _CaL * _SbL * _SbL * _CbL / (_RLd * _RLd * _RLd))[:, None] * _daL - (
    #     _CaL * _CaL * _SbL / (_RLd * _RLd * _RLd)
    # )[:, None] * _db
    # _dCL = (
    #     _dR * _RL[:, None] * _CLa[:, None] * (_aL[:, None] - _aL0[:, None])
    #     + _R[:, None] * _dRL * _CLa[:, None] * (_aL[:, None] - _aL0[:, None])
    #     + _R[:, None] * _RL[:, None] * _CLa[:, None] * _daL
    # )
    # J -= _dCL
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
    r1mag = sqrt(r1[XDIM]^2 + r1[YDIM]^2 + r1[ZDIM]^2)
    uinf = endvec

    r1dotuinf = r1[XDIM] * uinf[XDIM] + r1[YDIM] * uinf[YDIM] + r1[ZDIM] * uinf[ZDIM]

    d = norm(cross(r1, uinf, axis=1), axis=1)
    d = ifelse.(r1dotuinf .< 0.0, r1mag, d)

    influence = cross(uinf, r1, axis=1) * (d^2 / sqrt(rc^4 + d^4)) /
                (4π * r1mag * (r1mag - r1dotuinf))

    #  TODO: may need nan to num here
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
    r1mag = sqrt(r1[XDIM]^2 + r1[YDIM]^2 + r1[ZDIM]^2)
    r2 = pt - endpt
    r2mag = sqrt(r2[XDIM]^2 + r2[YDIM]^2 + r2[ZDIM]^2)
    r1r2 = r1 - r2
    r1r2mag = sqrt(r1r2[XDIM]^2 + r1r2[YDIM]^2 + r1r2[ZDIM]^2)
    r1dotr2 = r1[XDIM] * r2[XDIM] + r1[YDIM] * r2[YDIM] + r1[ZDIM] * r2[ZDIM]
    r1dotr1r2 = r1[XDIM] * r1r2[XDIM] + r1[YDIM] * r1r2[YDIM] + r1[ZDIM] * r1r2[ZDIM]
    r2dotr1r2 = r2[XDIM] * r1r2[XDIM] + r2[YDIM] * r1r2[YDIM] + r2[ZDIM] * r1r2[ZDIM]

    d = norm(cross(r1, r2, axis=1), axis=1) / r1r2mag
    d = ifelse.(r1dotr1r2 .< 0.0, r1mag, d)
    d = ifelse.(r2dotr1r2 .< 0.0, r2mag, d)

    influence = (r1mag + r2mag) + cross(r1, r2, axis=1) * d^2 / sqrt(rc^4 + d^4) /
                                  (4π * r1mag * r2mag * (r1mag * r2mag + r1dotr2))

    return influence
end

end