# --- Julia 1.9 ---
"""
@File    :   liftingline_om.jl
@Time    :   2023/12/25
@Author  :   Galen Ng
@Desc    :   This is the openmdao wrapper for the lifting line
            NOTE: When checking partials, the cla derivatives will be wrong from OpenMDAO FD because the
            gammas_d are not perturbed as well
            * Another tricky thing is python and julia store arrays differently so the order of the jacobians for non-vector inputs or outputs will be different. You'll see some transpose flattening code here. 
"""

# --- Used in this script ---
using FLOWMath: abs_cs_safe, atan_cs_safe, norm_cs_safe
using LinearAlgebra
using SpecialFunctions


# --- Module to om wrap ---
for headerName = [
    "../hydro/LiftingLine",
    "../io/TecplotIO",
]
    include(headerName * ".jl")
end


using .LiftingLine
# ==============================================================================
#                         OpenMDAO operations for the lifting line NL solver
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
        OpenMDAOCore.VarData("displacements_col", val=zeros(6, LiftingLine.NPT_WING)), # displacements of collocation pts
        OpenMDAOCore.VarData("alfa0", val=0.0),
    ]
    outputs = [
        OpenMDAOCore.VarData("gammas", val=zeros(LiftingLine.NPT_WING)),
        OpenMDAOCore.VarData("gammas_d", val=zeros(LiftingLine.NPT_WING)), # perturbed gammas
    ]

    partials = [
        # --- Residuals ---
        OpenMDAOCore.PartialsData("gammas", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("gammas", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("gammas", "alfa0", method="fd"),
        OpenMDAOCore.PartialsData("gammas", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("gammas_d", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("gammas_d", "gammas_d", method="exact"),
        OpenMDAOCore.PartialsData("gammas_d", "alfa0", method="fd"),
        OpenMDAOCore.PartialsData("gammas_d", "displacements_col", method="exact"),
    ]
    # partials = [OpenMDAOCore.PartialsData("*", "*", method="exact")] # define the partials

    return inputs, outputs, partials
end

# If the discipline can solve itself, use this
function OpenMDAOCore.solve_nonlinear!(self::OMLiftingLine, inputs, outputs)

    println("solving nonlinear lifting line")

    ptVec = inputs["ptVec"]
    alfa0 = inputs["alfa0"][1]
    displCol = inputs["displacements_col"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Set hydro vars ---
    appendageParams["alfa0"] = alfa0

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
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol;
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
    ∂TV_influence = LiftingLine.compute_TVinfluences(∂FlowCond, LLMesh)
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

    ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    # --- Nonlinear solve for circulation distribution ---
    Gconv0, _, _ = LiftingLine.do_newton_raphson(
        LiftingLine.compute_LLresiduals, LiftingLine.compute_LLresJacobian, g0, nothing;
        # is_verbose=true,
        maxIters=50, tol=1e-6, mode="FiDi", solverParams=LLNLParams, appendageOptions=appendageOptions, solverOptions=solverOptions)

    Gconv_d, _, _ = LiftingLine.do_newton_raphson(
        LiftingLine.compute_LLresiduals, LiftingLine.compute_LLresJacobian, Gconv0, nothing;
        maxIters=50, tol=1e-6, mode="FiDi", solverParams=∂LLNLParams, appendageOptions=appendageOptions, solverOptions=solverOptions)

    # --- Set all values ---
    for (ii, gamma) in enumerate(Gconv0)
        outputs["gammas"][ii] = gamma
    end

    for (ii, gamma) in enumerate(Gconv_d)
        outputs["gammas_d"][ii] = gamma
    end

    return nothing
end

function OpenMDAOCore.linearize!(self::OMLiftingLine, inputs, outputs, partials)
    """
    This defines the derivatives of outputs
    """

    ptVec = inputs["ptVec"]
    alfa0 = inputs["alfa0"][1]
    gammas = outputs["gammas"]
    gammas_d = outputs["gammas_d"]
    displCol = inputs["displacements_col"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Set hydro vars ---
    appendageParams["alfa0"] = alfa0

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
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol;
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
    ∂TV_influence = LiftingLine.compute_TVinfluences(∂FlowCond, LLMesh)

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
    ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    # ∂r∂g = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="FiDi")
    ∂r∂g = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="CS") # use very accurate derivatives only when necessary

    ∂r∂xPt, ∂r∂xdispl = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FAD")

    # This definition really breaks my head but it's basically ∂r / ∂ <second-var>
    for (ii, ∂ri∂g) in enumerate(eachrow(∂r∂g))
        partials["gammas", "gammas"][ii, :] = ∂ri∂g
    end

    for (ii, ∂ri∂Xpt) in enumerate(eachrow(∂r∂xPt))
        partials["gammas", "ptVec"][ii, :] = ∂ri∂Xpt
    end

    for (ii, ∂ri∂xdispl) in enumerate(eachrow(∂r∂xdispl))
        # transpose reshape flatten stuff bc julia and python store arrays differently
        ∂ri∂xdispl = reshape(∂ri∂xdispl, 6, length(∂ri∂xdispl) ÷ 6)
        partials["gammas", "displacements_col"][ii, :] = vec(transpose(∂ri∂xdispl))
    end

    ∂r∂g_d = LiftingLine.compute_LLresJacobian(gammas_d; solverParams=∂LLNLParams, mode="CS") # use very accurate derivatives only when necessary

    for (ii, ∂ri∂g) in enumerate(eachrow(∂r∂g_d))
        partials["gammas_d", "gammas_d"][ii, :] = ∂ri∂g
    end

    ∂r∂xPt_d, ∂r∂xdispl_d = LiftingLine.compute_∂r∂Xpt(gammas_d, ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="FAD")

    for (ii, ∂ri∂Xpt) in enumerate(eachrow(∂r∂xPt_d))
        partials["gammas_d", "ptVec"][ii, :] = ∂ri∂Xpt
    end

    for (ii, ∂ri∂xdispl) in enumerate(eachrow(∂r∂xdispl_d))
        # transpose reshape flatten stuff bc julia and python store arrays differently
        ∂ri∂xdispl = reshape(∂ri∂xdispl, 6, length(∂ri∂xdispl) ÷ 6)
        partials["gammas_d", "displacements_col"][ii, :] = vec(transpose(∂ri∂xdispl))
    end

    return nothing
end

function OpenMDAOCore.guess_nonlinear!(self::OMLiftingLine, inputs, outputs, residuals)
    """
    Provide initial guess for the states
    """
    ptVec = inputs["ptVec"]
    alfa0 = inputs["alfa0"][1]
    displCol = inputs["displacements_col"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Set hydro vars ---
    appendageParams["alfa0"] = alfa0

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
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol;
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
    alfa0 = inputs["alfa0"][1]
    gammas = outputs["gammas"]
    gammas_d = outputs["gammas_d"]
    residuals["gammas"] .= 0.0
    displCol = inputs["displacements_col"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Set hydro vars ---
    appendageParams["alfa0"] = alfa0

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
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol;
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
    ∂TV_influence = LiftingLine.compute_TVinfluences(∂FlowCond, LLMesh)
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

    # ---------------------------
    #   Solve for circulation
    # ---------------------------
    sweepAng = LLMesh.sweepAng

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
    ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    resVec = LiftingLine.compute_LLresiduals(gammas; solverParams=LLNLParams)
    resVec_d = LiftingLine.compute_LLresiduals(gammas_d; solverParams=∂LLNLParams)


    # Residuals are of the output state variable
    for (ii, res) in enumerate(resVec)
        residuals["gammas"][ii] = res
    end

    for (ii, res) in enumerate(resVec_d)
        residuals["gammas_d"][ii] = res
    end

    return nothing
end

# ==============================================================================
#                         Lifting line cost functions
# ==============================================================================
# Use the explicit component to handle lifting line cost functions like CL, CDi, etc.
# after the system is solved
struct OMLiftingLineFuncs <: OpenMDAOCore.AbstractExplicitComp
    """
    These are all the options
    """
    nodeConn
    appendageParams
    appendageOptions
    solverOptions
end

function OpenMDAOCore.setup(self::OMLiftingLineFuncs)

    # Number of mesh points
    npt = size(self.nodeConn, 2) + 1
    NPT_WING = LiftingLine.NPT_WING
    if self.appendageOptions["config"] == "wing"
        NPT_WING = LiftingLine.NPT_WING ÷ 2
        println("Using half wing for explicit comp")
    end

    inputs = [
        OpenMDAOCore.VarData("ptVec", val=zeros(3 * 2 * npt)),
        OpenMDAOCore.VarData("alfa0", val=0.0),
        OpenMDAOCore.VarData("gammas", val=zeros(LiftingLine.NPT_WING)),
        OpenMDAOCore.VarData("gammas_d", val=zeros(LiftingLine.NPT_WING)),
        OpenMDAOCore.VarData("displacements_col", val=zeros(6, LiftingLine.NPT_WING)),
    ]

    outputs = [
        OpenMDAOCore.VarData("CL", val=0.0),
        OpenMDAOCore.VarData("CDi", val=0.0),
        OpenMDAOCore.VarData("CS", val=0.0),
        OpenMDAOCore.VarData("F_x", val=0.0),
        OpenMDAOCore.VarData("F_y", val=0.0),
        OpenMDAOCore.VarData("F_z", val=0.0),
        OpenMDAOCore.VarData("forces_dist", val=zeros(3, NPT_WING)),
        OpenMDAOCore.VarData("M_x", val=0.0),
        OpenMDAOCore.VarData("M_y", val=0.0),
        OpenMDAOCore.VarData("M_z", val=0.0),
        OpenMDAOCore.VarData("moments_dist", val=zeros(3, NPT_WING)), # not really going to use this since we have forces at collocation pts
        OpenMDAOCore.VarData("collocationPts", val=zeros(3, NPT_WING)), # collocation points 
        OpenMDAOCore.VarData("clmax", val=0.0), # KS aggregated clmax
        OpenMDAOCore.VarData("cl", val=zeros(NPT_WING)), # all cl along the wing
        # Empirical drag build up
        OpenMDAOCore.VarData("CDw", val=0.0),
        OpenMDAOCore.VarData("CDpr", val=0.0),
        OpenMDAOCore.VarData("CDj", val=0.0),
        OpenMDAOCore.VarData("CDs", val=0.0),
        OpenMDAOCore.VarData("Dw", val=0.0),
        OpenMDAOCore.VarData("Dpr", val=0.0),
        OpenMDAOCore.VarData("Dj", val=0.0),
        OpenMDAOCore.VarData("Ds", val=0.0),
        # --- lift slopes for dynamic solution ---
        OpenMDAOCore.VarData("cla_col", val=zeros(NPT_WING)),
    ]

    partials = [
        # --- WRT ptVec ---
        OpenMDAOCore.PartialsData("CL", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CDi", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CS", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("clmax", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("cl", "ptVec", method="exact"), # good
        OpenMDAOCore.PartialsData("F_x", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("F_y", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("F_z", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("forces_dist", "ptVec", method="exact"),
        # OpenMDAOCore.PartialsData("moments_dist", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("M_x", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("M_y", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("M_z", "ptVec", method="exact"),
        # Empirical drag build up
        OpenMDAOCore.PartialsData("CDw", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CDpr", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CDj", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CDs", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("Dw", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("Dpr", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("Dj", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("Ds", "ptVec", method="exact"),
        # --- lift slopes for dynamic solution ---
        OpenMDAOCore.PartialsData("cla_col", "ptVec", method="exact"), # good, but the FD check will be wrong because of how the cla is calculated. See test script
        # --- Hydro mesh ---
        OpenMDAOCore.PartialsData("collocationPts", "ptVec", method="exact"),
        # --- WRT gammas ---
        OpenMDAOCore.PartialsData("CL", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("CDi", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("CS", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("clmax", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("cl", "gammas", method="exact"), # good
        OpenMDAOCore.PartialsData("F_x", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("F_y", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("F_z", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("forces_dist", "gammas", method="exact"),
        # OpenMDAOCore.PartialsData("moments_dist", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("M_x", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("M_y", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("M_z", "gammas", method="exact"),
        # Empirical drag build up
        OpenMDAOCore.PartialsData("CDw", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("Dw", "gammas", method="exact"),
        # --- lift slopes for dynamic solution ---
        OpenMDAOCore.PartialsData("cla_col", "gammas", method="exact"),
        # --- WRT displacements col ---
        OpenMDAOCore.PartialsData("CL", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("CDi", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("CS", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("clmax", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("cl", "displacements_col", method="exact"), # good
        OpenMDAOCore.PartialsData("F_x", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("F_y", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("F_z", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("forces_dist", "displacements_col", method="exact"),
        # OpenMDAOCore.PartialsData("moments_dist", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("M_x", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("M_y", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("M_z", "displacements_col", method="exact"),
        # Empirical drag build up
        OpenMDAOCore.PartialsData("CDw", "displacements_col", method="exact"),
        OpenMDAOCore.PartialsData("Dw", "displacements_col", method="exact"),
        # --- lift slopes for dynamic solution ---
        OpenMDAOCore.PartialsData("cla_col", "displacements_col", method="exact"), # good, but the FD check will be wrong because of how the cla is calculated. See test script
        # --- Hydro mesh ---
        OpenMDAOCore.PartialsData("collocationPts", "displacements_col", method="exact"),
        # --- AOA DVs ---
        OpenMDAOCore.PartialsData("*", "alfa0", method="fd"),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMLiftingLineFuncs, inputs, outputs)

    Gconv = inputs["gammas"]
    Gconv_d = inputs["gammas_d"]
    ptVec = inputs["ptVec"]
    alfa0 = inputs["alfa0"][1]
    displCol = inputs["displacements_col"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Set hydro vars ---
    appendageParams["alfa0"] = alfa0

    # ************************************************
    #     Core solver
    # ************************************************
    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = LiftingLine.get_tipnode(LECoords)
    LLNLParams, FlowCond = LiftingLine.setup_solverparams(ptVec, nodeConn, idxTip, displCol, appendageOptions, appendageParams, solverOptions)

    DimForces, Γdist, clvec, cmvec, IntegratedForces, CL, CDi, CS = LiftingLine.compute_outputs(Gconv, LLNLParams.TV_influence, FlowCond, LLNLParams.LLSystem, LLNLParams)

    ksclmax = compute_KS(clvec, solverOptions["rhoKS"])

    # ---------------------------
    #   Drag build up
    # ---------------------------
    dragOutputs = LiftingLine.compute_dragsFromX(ptVec, Gconv, nodeConn, displCol, appendageParams, appendageOptions, solverOptions)
    CDw, CDpr, CDj, CDs, Dw, Dpr, Dj, Ds = dragOutputs

    START = 1
    STOP = LiftingLine.NPT_WING
    if appendageOptions["config"] == "wing"
        START = LiftingLine.NPT_WING ÷ 2 + 1
    end

    outputs["F_x"][1] = IntegratedForces[XDIM]
    outputs["F_y"][1] = IntegratedForces[YDIM]
    outputs["F_z"][1] = IntegratedForces[ZDIM]
    outputs["CL"][1] = CL
    outputs["CDi"][1] = CDi
    outputs["CS"][1] = CS
    outputs["clmax"][1] = ksclmax
    outputs["cl"][:] = clvec[START:STOP]

    for (ii, fi) in enumerate(eachrow(DimForces[:, START:STOP]))
        outputs["forces_dist"][ii, :] = fi
    end

    size(outputs["collocationPts"]) == size(LLNLParams.LLSystem.collocationPts[:, START:STOP]) || error("Size mismatch for collocationPts")
    for (ii, collocationi) in enumerate(eachrow(LLNLParams.LLSystem.collocationPts[:, START:STOP]))
        outputs["collocationPts"][ii, :] = collocationi
    end

    outputs["CDw"][1] = CDw
    outputs["CDpr"][1] = CDpr
    outputs["CDj"][1] = CDj
    outputs["CDs"][1] = CDs
    outputs["Dw"][1] = Dw
    outputs["Dpr"][1] = Dpr
    outputs["Dj"][1] = Dj
    outputs["Ds"][1] = Ds

    # ---------------------------
    #   Lift slope solution
    # ---------------------------
    cla = LiftingLine.compute_liftslopes(Gconv, Gconv_d, LLNLParams.LLSystem, FlowCond, LLNLParams.LLHydro, LLNLParams.Airfoils, LLNLParams.AirfoilInfluences, appendageOptions, solverOptions)
    outputs["cla_col"][:] = cla[START:STOP]


    # outputDir = "./"
    # write_hydromesh(LLNLParams.LLSystem, FlowCond.uvec, outputDir)

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMLiftingLineFuncs, inputs, partials)
    """
    """

    Gconv = inputs["gammas"]
    Gconv_d = inputs["gammas_d"]
    ptVec = inputs["ptVec"]
    alfa0 = inputs["alfa0"][1]
    displCol = inputs["displacements_col"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # --- Set hydro vars ---
    appendageParams["alfa0"] = alfa0

    # ************************************************
    #     Derivatives wrt ptVec (2025-03-02 agree)
    #     for the half-wing, only finite differences work
    # ************************************************
    mode = "FiDi" # slow 
    # mode = "CS" # broken
    # mode = "RAD" # broken
    # mode = "FAD" # use this mode for full wing

    START = 1
    STOP = LiftingLine.NPT_WING
    NPT_WING = LiftingLine.NPT_WING
    DIV = 1
    if appendageOptions["config"] == "wing"
        START = LiftingLine.NPT_WING ÷ 2 + 1
        NPT_WING = LiftingLine.NPT_WING ÷ 2
        DIV = 2
        mode = "FiDi"
    end

    costFuncsInOrder = ["F_x", "F_y", "F_z", "CL", "CDi", "CS", "clmax", "forces_dist", "cl"]
    FXIND = 1
    CLMAXIND = 7

    ∂f∂x, ∂f∂xdispl = LiftingLine.compute_∂I∂Xpt(Gconv, ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode=mode)


    for (ii, ∂fi∂x) in enumerate(eachrow(∂f∂x[FXIND:CLMAXIND, :]))
        partials[costFuncsInOrder[ii], "ptVec"][1, :] = ∂fi∂x
    end

    for (ii, ∂fi∂xdispl) in enumerate(eachrow(∂f∂xdispl[FXIND:CLMAXIND, :]))
        partials[costFuncsInOrder[ii], "displacements_col"][1, :] = ∂fi∂xdispl
    end

    for (ii, ∂fi∂x) in enumerate(eachrow(∂f∂x)[CLMAXIND+1+(START-1)*6:end-LiftingLine.NPT_WING, :])
        partials["forces_dist", "ptVec"][ii, :] = ∂fi∂x
    end

    ctr = 1
    for (ii, ∂fi∂xdispl) in enumerate(eachrow(∂f∂xdispl)[CLMAXIND+1:end-LiftingLine.NPT_WING, :])

        if appendageOptions["config"] == "wing"

            whichHalf = div(ii - 1, LiftingLine.NPT_WING ÷ 2) # divisor
            if !iseven(whichHalf)
                partials["forces_dist", "displacements_col"][ctr, :] = ∂fi∂xdispl
                ctr += 1
            end

        else
            partials["forces_dist", "displacements_col"][ii, :] = ∂fi∂xdispl
        end
    end

    for (ii, ∂fi∂x) in enumerate(eachrow(∂f∂x)[end-LiftingLine.NPT_WING+START:end, :])
        partials["cl", "ptVec"][ii, :] = ∂fi∂x
    end

    for (ii, ∂fi∂xdispl) in enumerate(eachrow(∂f∂xdispl)[end-LiftingLine.NPT_WING+START:end, :])
        # transpose reshape flatten stuff bc julia and python store arrays differently
        # ∂fi∂xdispl = reshape(∂fi∂xdispl, 6, length(∂fi∂xdispl) ÷ 6)
        partials["cl", "displacements_col"][ii, :] = vec(transpose(∂fi∂xdispl))
    end

    # println("size ∂f∂xdispl: ", size(∂f∂xdispl))
    # println("size ∂f∂x: ", size(partials[costFuncsInOrder[1], "displacments_col"][:, :]))

    # ---------------------------
    #   Drag build up derivatives
    # --------------------------- 
    dragMode = "RAD"
    dragMode = "FiDi"
    if appendageOptions["config"] == "wing"
        dragMode = "FiDi"
    end
    ∂Drag∂Xpt, ∂Drag∂xdispl, ∂Drag∂G = LiftingLine.compute_∂EmpiricalDrag(ptVec, Gconv, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode=dragMode)
    partials["CDw", "ptVec"][1, :] = ∂Drag∂Xpt[1, :]
    partials["CDpr", "ptVec"][1, :] = ∂Drag∂Xpt[2, :]
    partials["CDj", "ptVec"][1, :] = ∂Drag∂Xpt[3, :]
    partials["CDs", "ptVec"][1, :] = ∂Drag∂Xpt[4, :]
    partials["Dw", "ptVec"][1, :] = ∂Drag∂Xpt[5, :]
    partials["Dpr", "ptVec"][1, :] = ∂Drag∂Xpt[6, :]
    partials["Dj", "ptVec"][1, :] = ∂Drag∂Xpt[7, :]
    partials["Ds", "ptVec"][1, :] = ∂Drag∂Xpt[8, :]
    dCdwdxdisp = vec(transpose(reshape(∂Drag∂xdispl[1, :], 6, LiftingLine.NPT_WING))) # transpose reshape flatten stuff bc julia and python store arrays differently
    partials["CDw", "displacements_col"][1, :] = dCdwdxdisp
    dDwdxdisp = vec(transpose(reshape(∂Drag∂xdispl[5, :], 6, LiftingLine.NPT_WING))) # transpose reshape flatten stuff bc julia and python store arrays differently
    partials["Dw", "displacements_col"][1, :] = dDwdxdisp

    # ---------------------------
    #   Hydro mesh points
    # ---------------------------
    nodeMode = "FAD"

    ∂collocationPt∂Xpt = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode=nodeMode)
    ctr = 1
    for (ii, ∂cPti∂xPt) in enumerate(eachrow(∂collocationPt∂Xpt))
        if appendageOptions["config"] == "wing"
            whichHalf = div(ii - 1, LiftingLine.NPT_WING ÷ 2) # divisor
            if !iseven(whichHalf)
                partials["collocationPts", "ptVec"][ctr, :] = ∂cPti∂xPt
                ctr += 1
            end
        else
            partials["collocationPts", "ptVec"][ii, :] = ∂cPti∂xPt
        end
    end

    partials["collocationPts", "displacements_col"] .= 0.0
    ∂collocationPt∂displCol = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displCol, appendageParams, appendageOptions, solverOptions; mode="Analytic")
    ctr = 1
    for (ii, ∂cPti∂xdispl) in enumerate(eachrow(∂collocationPt∂displCol))
        if appendageOptions["config"] == "wing"
            whichHalf = div(ii - 1, LiftingLine.NPT_WING ÷ 2) # divisor
            if !iseven(whichHalf)
                partials["collocationPts", "displacements_col"][ctr, :] = ∂cPti∂xdispl
                ctr += 1
            end
        else
            partials["collocationPts", "displacements_col"][ii, :] = ∂cPti∂xdispl
        end
    end

    # ---------------------------
    #   Lift slopes
    # ---------------------------
    dcldXpt_i = partials["cl", "ptVec"]
    appendageParams_d = copy(appendageParams)
    appendageParams_d["alfa0"] = alfa0 + LiftingLine.Δα
    ∂f∂x_d, ∂f∂xdispl_d = LiftingLine.compute_∂I∂Xpt(Gconv_d, ptVec, nodeConn, displCol, appendageParams_d, appendageOptions, solverOptions; mode=mode)
    dcldXpt_f = ∂f∂x_d[end-LiftingLine.NPT_WING+START:end, :]

    dcldxdispl_i = partials["cl", "displacements_col"]
    dcldxdispl_f = zeros(size(dcldxdispl_i))
    for (ii, ∂fi∂xdispl) in enumerate(eachrow(∂f∂xdispl_d)[end-LiftingLine.NPT_WING+START:end, :])
        # transpose reshape flatten stuff bc julia and python store arrays differently
        ∂fi∂xdispl = reshape(∂fi∂xdispl, 6, length(∂fi∂xdispl) ÷ 6)
        dcldxdispl_f[ii, :] = vec(transpose(∂fi∂xdispl))
    end

    dcladXpt = (dcldXpt_f - dcldXpt_i) / LiftingLine.Δα
    dcladxdispl = (dcldxdispl_f - dcldxdispl_i) / LiftingLine.Δα

    for (ii, ∂claidXpt) in enumerate(eachrow(dcladXpt))
        partials["cla_col", "ptVec"][ii, :] = ∂claidXpt
    end
    for (ii, ∂claidxdispl) in enumerate(eachrow(dcladxdispl))
        partials["cla_col", "displacements_col"][ii, :] = ∂claidxdispl
    end

    # ************************************************
    #     Derivatives wrt gammas (2025-03-02 these are all good)
    # clmax is off when setting FD step to 1e-4 --> agrees when using other step size
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
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displCol;
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
    ∂TV_influence = LiftingLine.compute_TVinfluences(∂FlowCond, LLMesh)

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
    ∂LLNLParams = LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

    ∂f∂g = LiftingLine.compute_∂I∂G(Gconv, LLMesh, FlowCond, LLNLParams, solverOptions) # Forward Diff by default

    for (ii, ∂fi∂g) in enumerate(eachrow(∂f∂g[FXIND:CLMAXIND, :]))
        partials[costFuncsInOrder[ii], "gammas"][:] = ∂fi∂g
    end

    for (ii, ∂fi∂g) in enumerate(eachrow(∂f∂g)[8+(START-1)*6:end-LiftingLine.NPT_WING, :])
        partials["forces_dist", "gammas"][ii, :] = ∂fi∂g
    end

    for (ii, ∂fi∂g) in enumerate(eachrow(∂f∂g)[end-LiftingLine.NPT_WING+START:end, :])
        partials["cl", "gammas"][ii, :] = ∂fi∂g
    end

    partials["CDw", "gammas"][1, :] = ∂Drag∂G[1, :]
    partials["Dw", "gammas"][1, :] = ∂Drag∂G[5, :]

    # ---------------------------
    #   Lift slopes
    # ---------------------------
    dcldg_i = partials["cl", "gammas"]
    ∂f∂g_d = LiftingLine.compute_∂I∂G(Gconv_d, LLMesh, ∂FlowCond, ∂LLNLParams, solverOptions) # Forward Diff by default
    dcldg_f = ∂f∂g_d[end-LiftingLine.NPT_WING+START:end, :]

    # println(size(dcldg_i))
    # println(size(dcldg_f))
    dcladg = (dcldg_f - dcldg_i) / LiftingLine.Δα
    for (ii, dclaidg) in enumerate(eachrow(dcladg))
        partials["cla_col", "gammas"][ii, :] = dclaidg
    end

    return nothing
end