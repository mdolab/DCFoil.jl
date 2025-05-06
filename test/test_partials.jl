"""
Boiler plate code to test partials of the various modules
"""

for headerName in [
    "../src/hydro/LiftingLine",
    "../src/struct/FEMMethods",
    "../src/solvers/SolveFlutter",
    "../src/solvers/SolveForced",
]
    include("$(headerName).jl")
end

using .LiftingLine
using .FEMMethods

# ==============================================================================
#                         Common input
# ==============================================================================
nNodes = 4
nNodesStrut = 2
NPT_WING = 6

appendageParams = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg] (angle of flow vector)
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "ab" => 0.0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.06 * ones(nNodes), # thickness-to-chord ratio (mean) #FLAGSTAFF
    "x_ab" => 0.0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0, # rake angle about top of strut [deg]
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 1.0, # [m]
    "c_strut" => 0.14 * collect(LinRange(1.0, 1.0, nNodesStrut)), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0.0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)
appendageOptions = Dict(
    "compName" => "rudder",
    "config" => "full-wing",
    "nNodes" => nNodes,
    "nNodeStrut" => nNodesStrut,
    "use_tipMass" => false,
    "xMount" => 0.0,
    "material" => "cfrp", # preselect from material library
    "strut_material" => "cfrp",
    "path_to_geom_props" => "./INPUT/1DPROPS/",
    "path_to_struct_props" => nothing,
    "path_to_geom_props" => nothing,
)
solverOptions = Dict(
    # ---------------------------
    #   I/O
    # ---------------------------
    "name" => "test",
    "debug" => false,
    "writeTecplotSolution" => true,
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList" => [appendageOptions],
    "gravityVector" => [0.0, 0.0, -9.81],
    # ---------------------------
    #   Flow
    # ---------------------------
    "Uinf" => 18.0, # free stream velocity [m/s]
    # "Uinf" => 11.0, # free stream velocity [m/s]
    # "Uinf" => 1.0, # free stream velocity [m/s]
    "rhof" => 1025.0, # fluid density [kg/m³]
    "nu" => 1.1892E-06,
    "use_nlll" => true, # use non-linear lifting line code
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # "use_dwCorrection" => true,
    # ---------------------------
    #   Solver modes
    # ---------------------------
    # --- Static solve ---
    "run_static" => true,
    "res_jacobian" => "analytic",
    # --- Forced solve ---
    "run_forced" => true,
    "fRange" => [0.01, 2.0], # forcing frequency sweep [Hz]
    # "df" => 0.05, # frequency step size
    "df" => 0.005, # frequency step size
    "tipForceMag" => 1.0,
    # --- p-k (Eigen) solve ---
    "run_modal" => true,
    "run_flutter" => true,
    "nModes" => 4,
    "uRange" => [29.0, 30],
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 500.0,
)
displacementsCol = zeros(6, NPT_WING)

# ==============================================================================
#                         Lifting line partials
# ==============================================================================
function test_LLresidualJacobians()

    # ************************************************
    #     Setups
    # ************************************************
    gammas = ones(NPT_WING) * 0.1 # vortex strengths

    LECoords = zeros(3, 10)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.1:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)
    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = LiftingLine.get_tipnode(LECoords)
    midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displacementsCol;
        npt_wing=NPT_WING, # OVERWRITE
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)
    LLNLParams = LiftingLine.LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    # ************************************************
    #     Test residual Jacobian ∂r / ∂ u
    # ************************************************
    ∂r∂g_CS = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="CS")
    ∂r∂g_FD = LiftingLine.compute_LLresJacobian(gammas; solverParams=LLNLParams, mode="FiDi")

    println("∂r∂g_CS: ")
    show(stdout, "text/plain", ∂r∂g_CS)
    println("")
    println("∂r∂g_FD: ")
    show(stdout, "text/plain", ∂r∂g_FD)
    println("")

    # ************************************************
    #     Test residual Jacobian ∂r / ∂ xPt
    # ************************************************
    ∂r∂xPt_FAD, ∂r∂xdispl_FAD = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂r∂xPt_FD, ∂r∂xdispl_FD = LiftingLine.compute_∂r∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    println("∂r∂xPt_FAD: ")
    show(stdout, "text/plain", ∂r∂xPt_FAD)
    println("")
    println("∂r∂xPt_FD: ")
    show(stdout, "text/plain", ∂r∂xPt_FD)
    println("")

    println("∂r∂xdispl_FAD: ")
    show(stdout, "text/plain", ∂r∂xdispl_FAD)
    println("")
    println("∂r∂xdispl_FD: ")
    show(stdout, "text/plain", ∂r∂xdispl_FD)
    println("")
end

test_LLresidualJacobians()

function test_LLcostFuncJacobians()

    # ************************************************
    #     Setups
    # ************************************************
    LECoords = zeros(3, 10)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.1:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)

    gammas = ones(NPT_WING) * 0.1 # vortex strengths
    LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)

    idxTip = LiftingLine.get_tipnode(LECoords)
    midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

    α0 = appendageParams["alfa0"]
    β0 = appendageParams["beta"]
    rake = appendageParams["rake"]
    depth0 = appendageParams["depth0"]
    airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
    LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displacementsCol;
        npt_wing=NPT_WING, # OVERWRITE
        npt_airfoil=npt_airfoil,
        rhof=solverOptions["rhof"],
        airfoil_ctrl_xy=airfoilCtrlXY,
        airfoil_xy=airfoilXY,
        options=options,
    )
    TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)
    LLNLParams = LiftingLine.LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)

    # ************************************************
    #     wrt ptVec
    # ************************************************
    ∂f∂x, ∂f∂xdispl = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂f∂x_FD, ∂f∂xdispl_FD = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    println("∂f∂x: ", ∂f∂x)
    println("∂f∂x_FD: ", ∂f∂x_FD)
    println("∂f∂xdispl: ", ∂f∂xdispl)
    println("∂f∂xdispl_FD: ", ∂f∂xdispl_FD)

    # ************************************************
    #     Drag
    # ************************************************
    ∂Drag∂Xpt_fd, ∂Drag∂xdispl_fd, ∂Drag∂G_fd = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    ∂Drag∂Xpt_fad, ∂Drag∂xdispl_fad, ∂Drag∂G_fad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂Drag∂Xpt_rad, ∂Drag∂xdispl_rad, ∂Drag∂G_rad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")

    println("∂Drag∂Xpt_fad: ", ∂Drag∂Xpt_fad)
    println("∂Drag∂Xpt_fd: ", ∂Drag∂Xpt_fd)
    println("∂Drag∂Xpt_rad: ", ∂Drag∂Xpt_rad)
    # ************************************************
    #     wrt vortex strengths
    # ************************************************
    ∂f∂g_FAD = LiftingLine.compute_∂I∂G(gammas, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FAD")
    ∂f∂g_FiDi = LiftingLine.compute_∂I∂G(gammas, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FiDi")

    println("∂f∂g_FAD: ", ∂f∂g_FAD)
    println("∂f∂g_FiDi: ", ∂f∂g_FiDi)

    # ************************************************
    #     Collocation points wrt ptVec
    # ************************************************
    ∂collocationPt∂Xpt_fad = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂collocationPt∂Xpt_fd = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    println("∂collocationPt∂Xpt_fad: ", ∂collocationPt∂Xpt_fad)
    println("∂collocationPt∂Xpt_fd: ", ∂collocationPt∂Xpt_fd)

    # ************************************************
    #     Collocation points wrt displacement of collocation points
    # ************************************************
    ∂collocationPt∂displCol_an = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="Analytic")
    ∂collocationPt∂displCol_fd = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    println("∂collocationPt∂displCol_an: ", ∂collocationPt∂displCol_an)
    println("∂collocationPt∂displCol_fd: ", ∂collocationPt∂displCol_fd)
end

test_LLcostFuncJacobians()
# ==============================================================================
#                         Beam partials
# ==============================================================================
function test_BeamResidualJacobians()
    ∂rs∂xPt, ∂rs∂xParams = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="analytic",
        # mode="FiDi",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    ∂rs∂xPt, ∂rs∂xParams = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="FiDi",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)
end

function test_BeamCostFuncJacobians()
    reference = [0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0 0.0 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0 0.0 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.551115123125783e-17 0.0 0.0 -5.551115123125783e-17 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.551115123125783e-17 0.0 0.0 -5.551115123125783e-17 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0 0.0 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.25 0.0 0.0 0.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 0.4999999999999999 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 0.4999999999999999 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5; 2.75 0.0 0.0 -2.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.75 0.0 0.0 -2.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 4.440892098500626e-16 0.0 0.0 -4.440892098500626e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.25 0.0 0.0 4.440892098500626e-16 0.0 0.0 -4.440892098500626e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.25 0.0; 0.0 0.0 2.75 0.0 0.0 -2.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.75 0.0 0.0 -2.25 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 8.881784197001252e-16 0.0 0.0 -8.881784197001252e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 0.0 0.0 8.881784197001252e-16 0.0 0.0 -8.881784197001252e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 0.0; 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    # ************************************************
    #     Setup
    # ************************************************
    LECoords = zeros(3, 10)
    LECoords[1, :] .= -0.5
    LECoords[2, :] .= 0.0:0.1:0.9
    nodeConn = transpose([1 2; 2 3; 3 4; 4 5; 5 6; 6 7; 7 8; 8 9; 9 10])
    TECoords = copy(LECoords)
    TECoords[1, :] .= 0.5
    ptVec, m, n = FEMMethods.unpack_coords(LECoords, TECoords)



    # ************************************************
    #     Evaluate Jacobians
    # ************************************************
    ∂nodes∂x_rad = FEMMethods.compute_∂nodes∂x(ptVec, nodeConn, [appendageParams], appendageOptions; mode="RAD")
    ∂nodes∂x_fd = FEMMethods.compute_∂nodes∂x(ptVec, nodeConn, [appendageParams], appendageOptions; mode="FiDi")

    println("FD difference: ", maximum(abs.(∂nodes∂x_fd .- reference)))
    return maximum(abs.(∂nodes∂x_rad .- reference))
end