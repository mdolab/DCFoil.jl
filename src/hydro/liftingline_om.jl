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
using SpecialFunctions


# --- Module to om wrap ---
for headerName = [
    "../hydro/LiftingLine",
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
        # TODO: add angle of attack and other appendage params here
    ]
    outputs = [
        OpenMDAOCore.VarData("gammas", val=zeros(LiftingLine.NPT_WING)),]

    partials = [
        # OpenMDAOCore.PartialsData("*", "*", method="fd"),
        # --- Residuals ---
        OpenMDAOCore.PartialsData("gammas", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("gammas", "gammas", method="exact"),
    ]
    # partials = [OpenMDAOCore.PartialsData("*", "*", method="exact")] # define the partials

    return inputs, outputs, partials
end

# If the discipline can solve itself, use this
function OpenMDAOCore.solve_nonlinear!(self::OMLiftingLine, inputs, outputs)

    println("solving nonlinear lifting line")

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

    ∂r∂xPt = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")


    # This definition really breaks my head but it's basically ∂r / ∂ <second-var>
    for (ii, ∂ri∂g) in enumerate(eachrow(∂r∂g))
        partials["gammas", "gammas"][ii, :] = ∂ri∂g
    end

    for (ii, ∂ri∂Xpt) in enumerate(eachrow(∂r∂xPt))
        partials["gammas", "ptVec"][ii, :] = ∂ri∂Xpt
    end


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

    # ---------------------------
    #   Calculate influence matrix
    # ---------------------------
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

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

    inputs = [
        OpenMDAOCore.VarData("ptVec", val=zeros(3 * 2 * npt)),
        OpenMDAOCore.VarData("gammas", val=zeros(LiftingLine.NPT_WING)),
    ]

    outputs = [
        OpenMDAOCore.VarData("CL", val=0.0),
        OpenMDAOCore.VarData("CDi", val=0.0),
        OpenMDAOCore.VarData("CS", val=0.0),
        OpenMDAOCore.VarData("F_x", val=0.0),
        OpenMDAOCore.VarData("F_y", val=0.0),
        OpenMDAOCore.VarData("F_z", val=0.0),
        OpenMDAOCore.VarData("forces_dist", val=zeros(3, LiftingLine.NPT_WING)),
        OpenMDAOCore.VarData("M_x", val=0.0),
        OpenMDAOCore.VarData("M_y", val=0.0),
        OpenMDAOCore.VarData("M_z", val=0.0),
        OpenMDAOCore.VarData("moments_dist", val=zeros(3, LiftingLine.NPT_WING)),
        OpenMDAOCore.VarData("collocationPts", val=zeros(3, LiftingLine.NPT_WING)), # collocation points 
        OpenMDAOCore.VarData("clmax", val=0.0), # KS aggregated clmax
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
        OpenMDAOCore.VarData("cla", val=zeros(LiftingLine.NPT_WING)),
    ]

    partials = [
        # --- WRT ptVec ---
        OpenMDAOCore.PartialsData("CL", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CDi", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("CS", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("clmax", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("F_x", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("F_y", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("F_z", "ptVec", method="exact"),
        OpenMDAOCore.PartialsData("forces_dist", "ptVec", method="exact"),
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
        OpenMDAOCore.PartialsData("cla", "ptVec", method="exact"),
        # --- WRT gammas ---
        OpenMDAOCore.PartialsData("CL", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("CDi", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("CS", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("clmax", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("F_x", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("F_y", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("F_z", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("forces_dist", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("M_x", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("M_y", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("M_z", "gammas", method="exact"),
        # Empirical drag build up
        OpenMDAOCore.PartialsData("CDw", "gammas", method="exact"),
        OpenMDAOCore.PartialsData("Dw", "gammas", method="exact"),
        # --- lift slopes for dynamic solution ---
        OpenMDAOCore.PartialsData("cla", "gammas", method="exact"),
    ]
    # partials = [OpenMDAOCore.PartialsData("*", "*", method="fd")] # define the partials

    return inputs, outputs, partials
end

function OpenMDAOCore.compute!(self::OMLiftingLineFuncs, inputs, outputs)

    Gconv = inputs["gammas"]
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

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    DimForces, Γdist, clvec, cmvec, IntegratedForces, CL, CDi, CS = LiftingLine.compute_outputs(Gconv, TV_influence, FlowCond, LLMesh, LLNLParams)

    for (ii, DimForce) in enumerate(eachcol(DimForces))
        outputs["forces_dist"][:, ii] = DimForce
    end

    ksclmax = compute_KS(clvec, solverOptions["rhoKS"])

    # ---------------------------
    #   Drag build up
    # ---------------------------
    dragOutputs = LiftingLine.compute_dragsFromX(ptVec, Gconv, nodeConn, appendageParams, appendageOptions, solverOptions)
    CDw, CDpr, CDj, CDs, Dw, Dpr, Dj, Ds = dragOutputs

    outputs["F_x"][1] = IntegratedForces[XDIM]
    outputs["F_y"][1] = IntegratedForces[YDIM]
    outputs["F_z"][1] = IntegratedForces[ZDIM]
    outputs["CL"][1] = CL
    outputs["CDi"][1] = CDi
    outputs["CS"][1] = CS
    outputs["clmax"][1] = ksclmax
    outputs["forces_dist"][:] = DimForces

    outputs["collocationPts"][:] = LLMesh.collocationPts

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
    # TODO: PICKUP HERE

    return nothing
end

function OpenMDAOCore.compute_partials!(self::OMLiftingLineFuncs, inputs, partials)
    """
    """

    Gconv = inputs["gammas"]
    ptVec = inputs["ptVec"]

    # --- Deal with options here ---
    nodeConn = self.nodeConn
    appendageParams = self.appendageParams
    solverOptions = self.solverOptions
    appendageOptions = self.appendageOptions

    # ************************************************
    #     Derivatives wrt ptVec
    # ************************************************
    # mode = "FiDi"
    # mode = "CS" # broken
    # mode = "RAD" # broken
    mode = "FAD"
    costFuncsInOrder = ["F_x", "F_y", "F_z", "CL", "CDi", "CS", "clmax", "forces_dist"]
    ∂f∂x = LiftingLine.compute_∂I∂Xpt(Gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode=mode)

    for (ii, ∂fi∂x) in enumerate(eachrow(∂f∂x[1:7, :]))
        # println("shape: ", size(∂fi∂x))
        partials[costFuncsInOrder[ii], "ptVec"][1, :] = ∂fi∂x
    end

    for (ii, ∂fi∂x) in enumerate(eachrow(∂f∂x)[8:end])
        partials["forces_dist", "ptVec"][ii, :] = ∂fi∂x
    end

    # ---------------------------
    #   Drag build up derivatives
    # --------------------------- 
    ∂Drag∂Xpt, ∂Drag∂G = LiftingLine.compute_∂EmpiricalDrag(ptVec, Gconv, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")
    partials["CDw", "ptVec"][1, :] = ∂Drag∂Xpt[1, :]
    partials["CDpr", "ptVec"][1, :] = ∂Drag∂Xpt[2, :]
    partials["CDj", "ptVec"][1, :] = ∂Drag∂Xpt[3, :]
    partials["CDs", "ptVec"][1, :] = ∂Drag∂Xpt[4, :]
    partials["Dw", "ptVec"][1, :] = ∂Drag∂Xpt[5, :]
    partials["Dpr", "ptVec"][1, :] = ∂Drag∂Xpt[6, :]
    partials["Dj", "ptVec"][1, :] = ∂Drag∂Xpt[7, :]
    partials["Ds", "ptVec"][1, :] = ∂Drag∂Xpt[8, :]


    # ************************************************
    #     Derivatives wrt gammas
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

    LLNLParams = LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    ∂f∂g = LiftingLine.compute_∂I∂G(Gconv, LLMesh, FlowCond, LLNLParams, solverOptions) # Forward Diff by default
    for (ii, ∂fi∂g) in enumerate(eachrow(∂f∂g[1:7, :]))
        partials[costFuncsInOrder[ii], "gammas"][1, :] = ∂fi∂g
    end

    for (ii, ∂fi∂g) in enumerate(eachrow(∂f∂g)[8:end])
        partials["forces_dist", "gammas"][ii, :] = ∂fi∂g
    end

    partials["CDw", "gammas"][1, :] = ∂Drag∂G[1, :]
    # partials["CDpr", "gammas"][1, :] = ∂Drag∂G[2, :]
    # partials["CDj", "gammas"][1, :] = ∂Drag∂G[3, :]
    # partials["CDs", "gammas"][1, :] = ∂Drag∂G[4, :]
    partials["Dw", "gammas"][1, :] = ∂Drag∂G[5, :]
    # partials["Dpr", "gammas"][1, :] = ∂Drag∂G[6, :]
    # partials["Dj", "gammas"][1, :] = ∂Drag∂G[7, :]
    # partials["Ds", "gammas"][1, :] = ∂Drag∂G[8, :]

    return nothing
end