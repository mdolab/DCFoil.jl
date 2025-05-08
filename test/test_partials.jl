"""
Boiler plate code to test partials of the various modules
"""

using DelimitedFiles

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

    err = maximum(abs.(∂r∂g_FD - ∂r∂g_CS))
    println("max ∂r∂g error: $(err)")
    if err >= 1e-3
        error("∂r∂g CS and FD do not match!")
    end

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

    err1 = maximum(abs.(∂r∂xPt_FD - ∂r∂xPt_FAD))
    println("max ∂r∂xPt error: $(err1)")
    if err1 >= 1e-3
        error("∂r∂xPt FAD and FD do not match!")
    end
    err2 = maximum(abs.(∂r∂xdispl_FD - ∂r∂xdispl_FAD))
    if err2 >= 1e-3
        error("∂r∂xdispl FAD and FD do not match!")
    end

    # # Acceptable
    # println("∂r∂xdispl_FAD: ")
    # show(stdout, "text/plain", ∂r∂xdispl_FAD)
    # println("")
    # println("∂r∂xdispl_FD: ")
    # show(stdout, "text/plain", ∂r∂xdispl_FD)
    # println("")

    # println("differences")
    # show(stdout, "text/plain", abs.(∂r∂xdispl_FD - ∂r∂xdispl_FAD))
    # println("")

end

test_LLresidualJacobians()

function test_LLcostFuncJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol)

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
    println("Running partial derivatives wrt ptVec...")
    # TODO: these don't agree well...
    ∂f∂x, ∂f∂xdispl = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂f∂x_FD, ∂f∂xdispl_FD = LiftingLine.compute_∂I∂Xpt(gammas, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    println("∂f∂x: ")
    show(stdout, "text/plain", ∂f∂x)
    println("")
    println("∂f∂x_FD: ")
    show(stdout, "text/plain", ∂f∂x_FD)
    println("")
    # println("∂f∂xdispl: ")
    # show(stdout, "text/plain", ∂f∂xdispl)
    # println("")
    # println("∂f∂xdispl_FD: ")
    # show(stdout, "text/plain", ∂f∂xdispl_FD)
    # println("")

    err = maximum(abs.(∂f∂x_FD - ∂f∂x))
    idx = argmax(abs.(∂f∂x_FD - ∂f∂xdispl))
    relerr = err / ∂f∂xdispl[idx]
    println("max ∂f∂x error: $(err)")

    err = maximum(abs.((∂f∂xdispl_FD - ∂f∂xdispl)))
    idx = argmax(abs.(∂f∂xdispl_FD - ∂f∂xdispl))
    relerr = err / ∂f∂xdispl[idx]
    println("max ∂f∂xdispl error: $(err)")
    println("indices $(idx)")
    println("max ∂f∂xdispl rel. error: $(relerr)")
    if relerr >= 5e-3
        error("∂f∂xdispl FAD and FD do not match within tolerance!")
    end

    # ************************************************
    #     Drag
    # ************************************************
    # TODO: GGGGGG FIGURE POUT WHY THE ZYGOTE ERROR HAPPENS HERE
    ptVec = [-0.135, 0.0, 0.0, -0.135, 0.04736842, 0.0, -0.135, 0.09473684, 0.0, -0.135, 0.14210526, 0.0, -0.135, 0.18947368, 0.0, -0.135, 0.23684211, 0.0, -0.135, 0.28421053, 0.0, -0.135, 0.33157895, 0.0, -0.135, 0.37894737, 0.0, -0.135, 0.42631579, 0.0, -0.135, 0.47368421, 0.0, -0.135, 0.52105263, 0.0, -0.135, 0.56842105, 0.0, -0.135, 0.61578947, 0.0, -0.135, 0.66315789, 0.0, -0.135, 0.71052632, 0.0, -0.135, 0.75789474, 0.0, -0.135, 0.80526316, 0.0, -0.135, 0.85263158, 0.0, -0.135, 0.9, 0.0, -0.135, -0.04736842, 0.0, -0.135, -0.09473684, 0.0, -0.135, -0.14210526, 0.0, -0.135, -0.18947368, 0.0, -0.135, -0.23684211, 0.0, -0.135, -0.28421053, 0.0, -0.135, -0.33157895, 0.0, -0.135, -0.37894737, 0.0, -0.135, -0.42631579, 0.0, -0.135, -0.47368421, 0.0, -0.135, -0.52105263, 0.0, -0.135, -0.56842105, 0.0, -0.135, -0.61578947, 0.0, -0.135, -0.66315789, 0.0, -0.135, -0.71052632, 0.0, -0.135, -0.75789474, 0.0, -0.135, -0.80526316, 0.0, -0.135, -0.85263158, 0.0, -0.135, -0.9, 0.0, 0.135, 0.0, 0.0, 0.135, 0.04736842, 0.0, 0.135, 0.09473684, 0.0, 0.135, 0.14210526, 0.0, 0.135, 0.18947368, 0.0, 0.135, 0.23684211, 0.0, 0.135, 0.28421053, 0.0, 0.135, 0.33157895, 0.0, 0.135, 0.37894737, 0.0, 0.135, 0.42631579, 0.0, 0.135, 0.47368421, 0.0, 0.135, 0.52105263, 0.0, 0.135, 0.56842105, 0.0, 0.135, 0.61578947, 0.0, 0.135, 0.66315789, 0.0, 0.135, 0.71052632, 0.0, 0.135, 0.75789474, 0.0, 0.135, 0.80526316, 0.0, 0.135, 0.85263158, 0.0, 0.135, 0.9, 0.0, 0.135, -0.04736842, 0.0, 0.135, -0.09473684, 0.0, 0.135, -0.14210526, 0.0, 0.135, -0.18947368, 0.0, 0.135, -0.23684211, 0.0, 0.135, -0.28421053, 0.0, 0.135, -0.33157895, 0.0, 0.135, -0.37894737, 0.0, 0.135, -0.42631579, 0.0, 0.135, -0.47368421, 0.0, 0.135, -0.52105263, 0.0, 0.135, -0.56842105, 0.0, 0.135, -0.61578947, 0.0, 0.135, -0.66315789, 0.0, 0.135, -0.71052632, 0.0, 0.135, -0.75789474, 0.0, 0.135, -0.80526316, 0.0, 0.135, -0.85263158, 0.0, 0.135, -0.9, 0.0]
    displCol = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
    Gconv = [0.04690876722640871, 0.06108229189190107, 0.06841831614902917, 0.0728857725585854, 0.07582480121164972, 0.07783230392550569, 0.07921538705136152, 0.08014528603314902, 0.08072053216903123, 0.08099586081021691, 0.08099586081021694, 0.08072053216903126, 0.08014528603314902, 0.07921538705136152, 0.07783230392550569, 0.07582480121164975, 0.07288577255858546, 0.06841831614902923, 0.06108229189190113, 0.04690876722640872]
    nodeConn = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 1 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38; 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39]
    appendageParams = Dict("alfa0" => 6.0, "zeta" => 0.04, "ab" => [0.0, 0.0, 0.0, 0.0, 0.0], "toc" => [1.0, 1.0, 1.0, 1.0, 1.0], "x_ab" => [0.0, 0.0, 0.0, 0.0, 0.0], "theta_f" => 0.0, "depth0" => 0.4, "rake" => 0.0, "beta" => 0.0, "s_strut" => 1.0, "c_strut" => [0.14, 0.14, 0.14], "toc_strut" => [0.095, 0.095, 0.095], "ab_strut" => [0.0, 0.0, 0.0], "x_ab_strut" => [0.0, 0.0, 0.0], "theta_f_strut" => 0.0)
    appendageOptions = Dict("compName" => "rudder", "config" => "full-wing", "nNodes" => 5, "nNodeStrut" => 3, "use_tipMass" => false, "xMount" => 0.0, "material" => "cfrp", "strut_material" => "cfrp", "path_to_geom_props" => nothing, "path_to_struct_props" => nothing)
    solverOptions = Dict("name" => "test", "debug" => false, "writeTecplotSolution" => true, "outputDir" => "output", "appendageList" => [Dict("compName" => "rudder", "config" => "full-wing", "nNodes" => 5, "nNodeStrut" => 3, "use_tipMass" => false, "xMount" => 0.0, "material" => "cfrp", "strut_material" => "cfrp", "path_to_geom_props" => nothing, "path_to_struct_props" => nothing)], "gravityVector" => Any[0.0, 0.0, -9.81], "Uinf" => 18.0, "rhof" => 1025.0, "nu" => 1.1892e-6, "use_nlll" => true, "use_freeSurface" => false, "use_cavitation" => false, "use_ventilation" => false, "use_dwCorrection" => false, "run_static" => true, "res_jacobian" => "analytic", "run_forced" => false, "fRange" => Any[0.1, 1000.0], "tipForceMag" => 1.0, "run_body" => false, "run_modal" => false, "run_flutter" => false, "nModes" => 4, "uRange" => Any[5.144562197756971, 7.7168432966354565], "maxQIter" => 100, "rhoKS" => 500.0, "alphaConst" => 0.0, "betaConst" => 0.0001415253502820411)

    ∂Drag∂Xpt_fd, ∂Drag∂xdispl_fd, ∂Drag∂G_fd = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    ∂Drag∂Xpt_fad, ∂Drag∂xdispl_fad, ∂Drag∂G_fad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂Drag∂Xpt_rad, ∂Drag∂xdispl_rad, ∂Drag∂G_rad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gammas, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")

    println("∂Drag∂Xpt_fad: ")
    show(stdout, "text/plain", ∂Drag∂Xpt_fad)
    println("")
    println("∂Drag∂Xpt_fd: ")
    show(stdout, "text/plain", ∂Drag∂Xpt_fd)
    println("")
    println("∂Drag∂Xpt_rad: ")
    show(stdout, "text/plain", ∂Drag∂Xpt_rad)
    println("")
    writedlm("output/∂Drag∂Xpt_fad.csv", ∂Drag∂Xpt_fad, ",")
    writedlm("output/∂Drag∂Xpt_fd.csv", ∂Drag∂Xpt_fd, ",")
    writedlm("output/∂Drag∂Xpt_rad.csv", ∂Drag∂Xpt_rad, ",")
    println("writing ∂Drag∂Xpt_fad and ∂Drag∂Xpt_fd to file")

    # ************************************************
    #     wrt vortex strengths
    # ************************************************
    println("Running partial derivatives wrt gammas...")
    ∂f∂g_FAD = LiftingLine.compute_∂I∂G(gammas, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FAD")
    ∂f∂g_FiDi = LiftingLine.compute_∂I∂G(gammas, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FiDi")

    err = maximum(abs.(∂f∂g_FiDi - ∂f∂g_FAD))
    println("max ∂f∂g error: $(err)")
    if err >= 1e-3
        error("∂f∂g FAD and FD do not match!")
    end

    # ************************************************
    #     Collocation points wrt ptVec
    # ************************************************
    println("Running partial derivatives wrt ptVec...")
    ∂collocationPt∂Xpt_fad = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂collocationPt∂Xpt_fd = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")

    err = maximum(abs.(∂collocationPt∂Xpt_fd - ∂collocationPt∂Xpt_fad))
    println("max ∂collocationPt∂Xpt error: $(err)")
    if err >= 1e-3
        writedlm("OUTPUT/∂collocationPt∂Xpt_fad.csv", ∂collocationPt∂Xpt_fad, ",")
        writedlm("OUTPUT/∂collocationPt∂Xpt_fd.csv", ∂collocationPt∂Xpt_fd, ",")
        error("∂collocationPt∂Xpt do not match! Wrote to file")
    end

    # ************************************************
    #     Collocation points wrt displacement of collocation points
    # ************************************************
    println("Running partial derivatives wrt displacements of collocation points...")
    ∂collocationPt∂displCol_an = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="Analytic")
    ∂collocationPt∂displCol_fd = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
    ∂collocationPt∂displCol_fad = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
    ∂collocationPt∂displCol_rad = LiftingLine.compute_∂collocationPt∂displCol(ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")

    err1 = maximum(abs.(∂collocationPt∂displCol_fd - ∂collocationPt∂displCol_an))
    err2 = maximum(abs.(∂collocationPt∂displCol_fad - ∂collocationPt∂displCol_an))
    err3 = maximum(abs.(∂collocationPt∂displCol_rad - ∂collocationPt∂displCol_an))
    println("max ∂collocationPt∂displCol error: $(maximum([err1, err2, err3]))")
    if maximum([err1, err2, err3]) >= 1e-10
        writedlm("OUTPUT/∂collocationPt∂displCol_an.csv", ∂collocationPt∂displCol_an, ",")
        writedlm("OUTPUT/∂collocationPt∂displCol_fd.csv", ∂collocationPt∂displCol_fd, ",")
        writedlm("OUTPUT/∂collocationPt∂displCol_fad.csv", ∂collocationPt∂displCol_fad, ",")
        writedlm("OUTPUT/∂collocationPt∂displCol_rad.csv", ∂collocationPt∂displCol_rad, ",")
        error("∂collocationPt∂displCol disagreement! Wrote to file")
    end

    return 0.0
end

test_LLcostFuncJacobians(appendageParams, appendageOptions, solverOptions, displacementsCol)

# ==============================================================================
#                         Beam partials
# ==============================================================================
function test_BeamResidualJacobians(appendageParams, appendageOptions, solverOptions)



    allStructStates = ones()
    ∂rs∂xPt, ∂rs∂xParams = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="analytic",
        # mode="FiDi",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)
    ∂rs∂xPt_fd, ∂rs∂xParams_fd = FEMMethods.compute_∂r∂x(allStructStates, traction_forces, [appendageParams], LECoords, TECoords, nodeConn;
        mode="FiDi",
        # mode="RAD",
        appendageOptions=appendageOptions, solverOptions=solverOptions)

    err = maximum(abs.(∂rs∂xPt_fd - ∂rs∂xPt))
    err2 = maximum(abs.(∂rs∂xParams_fd - ∂rs∂xParams))
    if err >= 1e-3
        error("∂rs∂xPt CS and FD do not match!")
    end
    if err2 >= 1e-3
        error("∂rs∂xParams CS and FD do not match!")
    end

    return err
end
test_BeamResidualJacobians()

function test_BeamCostFuncJacobians()

    reference = [0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.16666666666666666 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.16666666666666666 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3333333333333333 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 -1.1102230246251565e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3333333333333333 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 0.4999999999999999 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.1102230246251565e-16 0.0 0.0 0.4999999999999999 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5; 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 2.220446049250313e-16 0.0 0.0 -2.220446049250313e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.16666666666666666 0.0 0.0 2.220446049250313e-16 0.0 0.0 -2.220446049250313e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.16666666666666666 0.0; 0.0 0.0 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 -1.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 4.440892098500626e-16 0.0 0.0 -4.440892098500626e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.3333333333333333 0.0 0.0 4.440892098500626e-16 0.0 0.0 -4.440892098500626e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.3333333333333333 0.0; 0.0 0.0 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.5 0.0 0.0 -3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 8.881784197001252e-16 0.0 0.0 -8.881784197001252e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 0.0 0.0 8.881784197001252e-16 0.0 0.0 -8.881784197001252e-16 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -0.5 0.0; 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 5.0 0.0 0.0 -4.5 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
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

    println("FD difference: ", maximum(abs.(∂nodes∂x_fd .- ∂nodes∂x_rad)))
    return maximum(abs.(∂nodes∂x_rad .- reference))
end

# ==============================================================================
#                         Flutter partials
# ==============================================================================
function test_FlutterJacobians()

    # ************************************************
    #     Setups
    # ************************************************


    dIdxDV = Zygote.gradient((xpt, xdispl, xcla, xtheta, xtoc, xalpha) ->
            SolveFlutter.cost_funcsFromDVsOM(xpt, nodeConn, xdispl, xcla, xtheta, xtoc, xalpha, appendageParams, solverOptions),
        ptVec,
        displacementsCol,
        claVecMod,
        appendageParams["theta_f"],
        appendageParams["toc"],
        appendageParams["alfa0"],
    )
    dIdxDV_fd = FiniteDifferences.jacobian(forward_fdm(2, 1), (xpt, xdispl, xcla, xtheta, xtoc, xalpha) ->
            SolveFlutter.cost_funcsFromDVsOM(xpt, nodeConn, xdispl, xcla, xtheta, xtoc, xalpha, appendageParams, solverOptions),
        ptVec,
        displacementsCol,
        claVecMod,
        appendageParams["theta_f"],
        appendageParams["toc"],
        appendageParams["alfa0"],
    )

    err = maximum(abs.(dIdxDV_fd .- dIdxDV))
end