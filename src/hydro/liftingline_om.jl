# --- Julia 1.9 ---
"""
@File    :   liftingline_om.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   This is the openmdao wrapper for the lifting line
"""

# --- Used in this script ---
using FLOWMath: abs_cs_safe, atan_cs_safe, norm_cs_safe
using LinearAlgebra

# --- Module to om wrap ---
for headerName = [
    "../hydro/LiftingLine",
    # "../ComputeHydroFunctions",
]
    include(headerName * ".jl")
end


using .LiftingLine
# ==============================================================================
#                         OpenMDAO operations for the lifting line
# ==============================================================================
using OpenMDAOCore: OpenMDAOCore

struct OMLiftingLine <: OpenMDAOCore.AbstractImplicitComp
    """
    These are all the options
    I don't know why @concrete is used, but that was in the CCBlade example
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMLiftingLine)
    """
    Setup OpenMDAO data
    """

    # Number of mesh points
    npt = size(self.nodeConn, 2) + 1

    # --- Define inputs and outputs ---
    inputs = [
        OpenMDAOCore.VarData("ptVec", val=zeros(npt * 3 * 2)),
    ]
    # inputs = [OpenMDAOCore.VarData("x", val=0.0), OpenMDAOCore.VarData("y", val=0.0)]
    outputs = [
        OpenMDAOCore.VarData("gammas", val=zeros(LiftingLine.NPT_WING)),
        # OpenMDAOCore.VarData("CL", val=0.0),
        # OpenMDAOCore.VarData("CDi", val=0.0),
        # OpenMDAOCore.VarData("F_x", val=0.0),
        # OpenMDAOCore.VarData("F_y", val=0.0),
        # OpenMDAOCore.VarData("F_z", val=0.0),
        # OpenMDAOCore.VarData("forces_dist", val=zeros(3, LiftingLine.NPT_WING)),
        # OpenMDAOCore.VarData("M_x", val=0.0),
        # OpenMDAOCore.VarData("M_y", val=0.0),
        # OpenMDAOCore.VarData("M_z", val=0.0),
        # OpenMDAOCore.VarData("moments_dist", val=zeros(3, LiftingLine.NPT_WING)),
    ]

    partials = [
        # OpenMDAOCore.PartialsData("*", "*", method="fd"),
        # --- Residuals ---
        OpenMDAOCore.PartialsData("gammas", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("gammas", "gammas", method="exact"),
        # --- Output type functions ---
        # OpenMDAOCore.PartialsData("gammas", "forces", rows=nothing, cols=nothing), # this is the same idea as dependent=false
        # OpenMDAOCore.PartialsData("CL", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("CDi", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("F_x", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("F_y", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("F_z", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("forces_dist", "ptVec", method="fd"),
    ]
    # partials = [OpenMDAOCore.PartialsData("*", "*", method="exact")] # define the partials

    return inputs, outputs, partials
end

# If the discipline can solve itself, use this
function OpenMDAOCore.solve_nonlinear!(self::OMLiftingLine, inputs, outputs)

    # println("solving nonlinear!")

    ptVec = inputs["ptVec"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # ************************************************
    #     Core solver
    # ************************************************
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
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords;
        npt_wing=npt_wing,
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

    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    # First guess using root properties
    c_r = LLMesh.rootChord
    clα = LLHydro.airfoil_CLa
    αL0 = LLHydro.airfoil_aL0
    sweepAng = LLMesh.sweepAng
    ux, uy, uz = FlowCond.uvec
    span = LLMesh.span
    ctrl_pts = LLMesh.collocationPts
    g0 = 0.5 * c_r * clα * cos(sweepAng) *
         (uz / ux - αL0) *
         (1.0 .- (2.0 * ctrl_pts[YDIM, :] / span) .^ 4) .^ (0.25)

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    # --- Nonlinear solve for circulation distribution ---
    Gconv, _, _ = LiftingLine.do_newton_raphson(
        LiftingLine.compute_LLresiduals, LiftingLine.compute_LLresJacobian, g0, nothing;
        maxIters=50, tol=1e-6, mode="FiDi", solverParams=LLNLParams, appendageOptions=appendageOptions, solverOptions=solverOptions)

    # --- Set all values ---
    for (ii, gamma) in enumerate(Gconv)
        outputs["gammas"][ii] = gamma
    end

    # DimForces, Γdist, clvec, cmvec, IntegratedForces, CL, CDi, CS = LiftingLine.compute_outputs(Gconv, TV_influence, FlowCond, LLMesh, LLNLParams)
    # for (ii, DimForce) in enumerate(eachcol(DimForces))
    #     outputs["forces_dist"][:, ii] = DimForce
    # end

    # outputs["F_x"][1] = IntegratedForces[XDIM]
    # outputs["F_y"][1] = IntegratedForces[YDIM]
    # outputs["F_z"][1] = IntegratedForces[ZDIM]
    # outputs["CL"][1] = CL
    # outputs["CDi"][1] = CDi

    return nothing
end

function OpenMDAOCore.linearize!(self::OMLiftingLine, inputs, outputs, partials)
    """
    This defines the derivatives of outputs
    """

    # println("running linearize...")

    ptVec = inputs["ptVec"]
    gammas = outputs["gammas"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Lifting line setup stuff ---    
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
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords;
        npt_wing=npt_wing,
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        # airfoilCoordFile=airfoilCoordFile,
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    ∂α = FlowCond.alpha + LiftingLine.Δα # FD

    ∂Uinfvec = FlowCond.Uinf * [cos(∂α), 0, sin(∂α)]
    ∂Uinf = norm_cs_safe(∂Uinfvec)
    ∂uvec = ∂Uinfvec / FlowCond.Uinf
    ∂FlowCond = LiftingLine.FlowConditions(∂Uinfvec, ∂Uinf, ∂uvec, ∂α, FlowCond.beta, FlowCond.rhof, FlowCond.depth)

    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    # ∂r∂g = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="FiDi")
    ∂r∂g = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="CS") # use very accurate derivatives only when necessary

    ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions)

    # # mode = "RAD"
    # mode = "FiDi"
    # # mode = "CS"
    # ∂fx∂xPt = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, "F_x"; mode=mode)
    # ∂fy∂xPt = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, "F_y"; mode=mode)
    # ∂fz∂xPt = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, "F_z"; mode=mode)
    # ∂CDi∂xPt = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, "CDi"; mode=mode)
    # ∂CLi∂xPt = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions, "CL"; mode=mode)

    # println("shape:\t", size(partials["gammas","gammas"]))

    # This definition really breaks my head but it's basically ∂r / ∂ <second-var>
    for (ii, ∂ri∂g) in enumerate(eachrow(∂r∂g))
        partials["gammas", "gammas"][ii, :] = ∂ri∂g
    end

    for (ii, ∂ri∂Xpt) in enumerate(eachrow(∂r∂xPt))
        partials["gammas", "ptVec"][ii, :] = ∂ri∂Xpt
    end

    # partials["F_x", "ptVec"][1, :] = ∂fx∂xPt[1, :]
    # partials["F_y", "ptVec"][1, :] = ∂fy∂xPt[1, :]
    # partials["F_z", "ptVec"][1, :] = ∂fz∂xPt[1, :]
    # partials["CDi", "ptVec"][1, :] = ∂CDi∂xPt[1, :]
    # partials["CL", "ptVec"][1, :] = ∂CLi∂xPt[1, :]

    return nothing
end

function OpenMDAOCore.guess_nonlinear!(self::OMLiftingLine, inputs, outputs, residuals)
    """
    Provide initial guess for the states
    """
    ptVec = inputs["ptVec"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Lifting line setup stuff ---    
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
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords;
        npt_wing=npt_wing,
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        # airfoilCoordFile=airfoilCoordFile,
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )

    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    # First guess using root properties
    c_r = LLMesh.rootChord
    clα = LLHydro.airfoil_CLa
    αL0 = LLHydro.airfoil_aL0
    sweepAng = LLMesh.sweepAng
    # Ux, _, Uz = FlowCond.Uinfvec
    ux, uy, uz = FlowCond.uvec
    span = LLMesh.span
    ctrl_pts = LLMesh.collocationPts
    g0 = 0.5 * c_r * clα * cos(sweepAng) *
         (uz / ux - αL0) *
         (1.0 .- (2.0 * ctrl_pts[YDIM, :] / span) .^ 4) .^ (0.25)

    # ************************************************
    #     Set the initial guess
    # ************************************************
    for (ii, gamma) in enumerate(g0)
        outputs["gammas"][ii] = gamma
    end
    println("Guessing nonlinear")

    return nothing
end

# Not needed if solve_nonlinear! is defined
function OpenMDAOCore.apply_nonlinear!(self::OMLiftingLine, inputs, outputs, residuals)
    """
    Apply the residuals, routine not needed if solve_nonlinear! is defined
    """

    ptVec = inputs["ptVec"]
    gammas = outputs["gammas"]
    residuals["gammas"] .= 0.0

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Lifting line setup stuff ---    
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
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords;
        npt_wing=npt_wing,
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        # airfoilCoordFile=airfoilCoordFile,
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    ∂α = FlowCond.alpha + LiftingLine.Δα # FD

    ∂Uinfvec = FlowCond.Uinf * [cos(∂α), 0, sin(∂α)]
    ∂Uinf = norm_cs_safe(∂Uinfvec)
    ∂uvec = ∂Uinfvec / FlowCond.Uinf
    ∂FlowCond = LiftingLine.FlowConditions(∂Uinfvec, ∂Uinf, ∂uvec, ∂α, FlowCond.beta, FlowCond.rhof, FlowCond.depth)

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

    influence_straightsega = LiftingLine.compute_straightSegment(P1, P2, ctrlPtMat, LLMesh.rc)
    influence_straightsegb = LiftingLine.compute_straightSegment(P2, P3, ctrlPtMat, LLMesh.rc) .* reshape(bound_mask, 1, size(bound_mask)...)
    influence_straightsegc = LiftingLine.compute_straightSegment(P3, P4, ctrlPtMat, LLMesh.rc)

    ∂influence_semiinfa = LiftingLine.compute_straightSemiinfinite(P1, ∂uinfMat, ctrlPtMat, LLMesh.rc)
    ∂influence_semiinfb = LiftingLine.compute_straightSemiinfinite(P4, ∂uinfMat, ctrlPtMat, LLMesh.rc)

    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

    ∂TV_influence = -∂influence_semiinfa +
                    influence_straightsega +
                    influence_straightsegb +
                    influence_straightsegc +
                    ∂influence_semiinfb


    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    sweepAng = LLMesh.sweepAng

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    resVec = LiftingLine.compute_LLresiduals(gammas; solverParams=LLNLParams)


    # Residuals are of the output state variable
    for (ii, res) in enumerate(resVec)
        residuals["gammas"][ii] = res
    end

    return nothing
end

# ==============================================================================
#                         Lifting line cost functions
# ==============================================================================
# Use the explicit component to handle lifting line cost functions like CL, CDi, etc.
# after the system is solved
struct OMLLOutputs <: OpenMDAOCore.AbstractExplicitComp
    """
    These are all the options
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end