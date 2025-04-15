# THIS IS THE JULIA CENTRIC RUN SCRIPT
using Plots
using OpenMDAO: om, make_component
include("../src/hydro/LiftingLine.jl")
include("../src/hydro/liftingline_om.jl")
include("../src/struct/beam_om.jl")
using .LiftingLine
using .FEMMethods
include("../src/solvers/solveflutter_om.jl")
include("../src/io/MeshIO.jl")
const RealOrComplex = Union{Real,Complex}
# const RealOrComplex = AbstractFloat

# ==============================================================================
#                         Setup stuff
# ==============================================================================
files = Dict(
    "gridFile" => ["./INPUT/flagstaff_foil_stbd_mesh.dcf"],
)
GridStruct = add_meshfiles(files["gridFile"], Dict("junction-first" => true))

LECoords = (GridStruct.LEMesh)
TECoords = (GridStruct.TEMesh)
nodeConn = (GridStruct.nodeConn)
ptVec, m, n = FEMMethods.unpack_coords(GridStruct.LEMesh, GridStruct.TEMesh)
nNodes = 5
nNodesStrut = 3
appendageParams = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg] (angle of flow vector)
    # "sweep" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    # "c" => collect(LinRange(0.14, 0.095, nNodes)), # chord length [m]
    "ab" => 0.0 * ones(RealOrComplex, nNodes), # dist from midchord to EA [m]
    "toc" => 0.075 * ones(RealOrComplex, nNodes), # thickness-to-chord ratio (mean)
    "x_ab" => 0.0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(0), # fiber angle global [rad]
    # --- Strut vars ---
    "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
    "rake" => 0.0, # rake angle about top of strut [deg]
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 1.0, # [m]
    "c_strut" => 0.14 * collect(LinRange(1.0, 1.0, nNodesStrut)), # chord length [m]
    "toc_strut" => 0.095 * ones(RealOrComplex, nNodesStrut), # thickness-to-chord ratio (mean)
    "ab_strut" => 0.0 * ones(RealOrComplex, nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0.0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
)

paramsList = [appendageParams]

appendageOptions = Dict(
    "compName" => "rudder",
    # "config" => "t-foil",
    # "config" => "full-wing",
    "config" => "wing",
    "nNodes" => nNodes,
    "nNodeStrut" => nNodesStrut,
    "use_tipMass" => false,
    "xMount" => 3.355,
    "material" => "cfrp", # preselect from material library
    "strut_material" => "cfrp",
    # "path_to_struct_props" => "./INPUT/1DPROPS/", # path to 1D properties
    "path_to_geom_props" => "./INPUT/1DPROPS/",
    "path_to_struct_props" => nothing,
    "path_to_geom_props" => nothing,
)

appendageList = [appendageOptions]
solverOptions = Dict(
    # ---------------------------
    #   I/O
    # ---------------------------
    # "name" => "R3E6",
    "name" => "mothrudder-nofs",
    "debug" => false,
    # "gridFile" => ["./INPUT/mothrudder_foil_stbd_mesh.dcf", "./INPUT/mothrudder_foil_port_mesh.dcf", "./INPUT/mothrudder_foil_strut_mesh.dcf"],
    "gridFile" => ["./INPUT/mothrudder_foil_stbd_mesh.dcf", "./INPUT/mothrudder_foil_port_mesh.dcf"], #, "./INPUT/mothrudder_foil_strut_mesh.dcf"],
    "writeTecplotSolution" => true,
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList" => appendageList,
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
    # "res_jacobian" => "CS",
    "res_jacobian" => "analytic",
    # "res_jacobian" => "RAD",
    # "onlyStructDerivs" => true,
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
displacementsCol = zeros(6, LiftingLine.NPT_WING)

solverOptions = FEMMethods.set_structDamping(ptVec, nodeConn, appendageParams, solverOptions, appendageList[1])
# ==============================================================================
#                         Derivatives
# ==============================================================================


# ************************************************
#     Testing cla d xpt
# ************************************************

# gconv = vec([0.0322038 0.03231678 0.03253403 0.03284775 0.03324736 0.03371952 0.03425337 0.03484149 0.03547895 0.03616117 0.03688162 0.03762993 0.03839081 0.03914407 0.03986539 0.04052803 0.04110476 0.04156866 0.04189266 0.04205787 0.04205787 0.04189266 0.04156866 0.04110476 0.04052803 0.03986539 0.03914407 0.03839081 0.03762993 0.03688162 0.03616117 0.03547895 0.03484149 0.03425337 0.03371952 0.03324736 0.03284775 0.03253403 0.03231678 0.0322038])
dalfa = LiftingLine.Δα
gconv = vec([0.00310651 0.00886824 0.01383426 0.01797466 0.02138239 0.02419296 0.02653531 0.02851189 0.03019559 0.03163361 0.03285384 0.03387137 0.03469469 0.03533145 0.03579343 0.03610042 0.0362815 0.03637183 0.03640545 0.03641115 0.03641115 0.03640545 0.03637183 0.0362815 0.03610042 0.03579343 0.03533145 0.03469469 0.03387137 0.03285384 0.03163361 0.03019559 0.02851189 0.02653531 0.02419296 0.02138239 0.01797466 0.01383426 0.00886824 0.00310651])
gconv_d = vec([0.00313608 0.00895266 0.013966 0.01814589 0.02158614 0.02442354 0.02678823 0.02878367 0.03048342 0.03193516 0.03316702 0.03419425 0.03502543 0.03566826 0.03613464 0.03644456 0.03662737 0.03671856 0.0367525 0.03675826 0.03675826 0.0367525 0.03671856 0.03662737 0.03644456 0.03613464 0.03566826 0.03502543 0.03419425 0.03316702 0.03193516 0.03048342 0.02878367 0.02678823 0.02442354 0.02158614 0.01814589 0.013966 0.00895266 0.00313608])
dfdxpt = LiftingLine.compute_∂I∂Xpt(gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")
appendageParams_d = copy(appendageParams)
appendageParams_d["alfa0"] += dalfa
dfdxpt_f = LiftingLine.compute_∂I∂Xpt(gconv_d, ptVec, nodeConn, appendageParams_d, appendageOptions, solverOptions; mode="FAD")
# dfdxpt = LiftingLine.compute_∂I∂Xpt(gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")

# Check the dcldx derivatives
dcldx = dfdxpt[end-LiftingLine.NPT_WING+1:end, :]
dcldx_f = dfdxpt_f[end-LiftingLine.NPT_WING+1:end, :]
dcladx = (dcldx_f - dcldx) / dalfa

function compute_dcladx(dh)

    function compute_clafromX(ptVec)

        LECoords, TECoords = LiftingLine.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
        idxTip = LiftingLine.get_tipnode(LECoords)
        midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

        α0 = appendageParams["alfa0"]
        β0 = appendageParams["beta"]
        rake = appendageParams["rake"]
        depth0 = appendageParams["depth0"]
        airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
        LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, zeros(6, LiftingLine.NPT_WING);
            npt_wing=npt_wing,
            npt_airfoil=npt_airfoil,
            rhof=solverOptions["rhof"],
            # airfoilCoordFile=airfoilCoordFile,
            airfoil_ctrl_xy=airfoilCtrlXY,
            airfoil_xy=airfoilXY,
            options=options,
        )

        cla = LiftingLine.compute_liftslopes(gconv, gconv_d, LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences, appendageOptions, solverOptions)
        return cla
    end

    # FD
    dcladx = zeros(LiftingLine.NPT_WING, length(ptVec))
    cla_p = compute_clafromX(ptVec)
    for ii in 1:length(ptVec)
        ptVec[ii] += dh
        cla_m = compute_clafromX(ptVec)
        ptVec[ii] -= dh
        dcladx[:, ii] = (cla_m - cla_p) / (dh)
    end

    # ForwardDiff.jacobian(compute_clafromX, ptVec)

    return dcladx
end

dcladx_fidi = compute_dcladx(dh)

dh = 1e-5

∂α = FlowCond.alpha + LiftingLine.Δα # FD
∂Uinfvec = FlowCond.Uinf * [cos(∂α), 0, sin(∂α)]
∂Uinf = norm_cs_safe(∂Uinfvec)
∂uvec = ∂Uinfvec / FlowCond.Uinf
∂FlowCond = LiftingLine.FlowConditions(∂Uinfvec, ∂Uinf, ∂uvec, ∂α, FlowCond.beta, FlowCond.rhof, FlowCond.depth)
∂TV_influence = LiftingLine.compute_TVinfluences(∂FlowCond, LLMesh)
∂LLNLParams = LiftingLine.LiftingLineNLParams(∂TV_influence, LLMesh, LLHydro, ∂FlowCond, Airfoils, AirfoilInfluences)

dfdg = LiftingLine.compute_∂I∂G(gconv, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FAD")
dfdg_f = LiftingLine.compute_∂I∂G(gconv_d, LLMesh, ∂FlowCond, ∂LLNLParams, solverOptions; mode="FAD")
dcldg = dfdg[end-LiftingLine.NPT_WING+1:end, :]
dcldg_f = dfdg_f[end-LiftingLine.NPT_WING+1:end, :]

dcladg = (dcldg_f - dcldg) / dalfa # this makes sense 

# ************************************************
#     Test twist into lifting line
# ************************************************
idxTip = LiftingLine.get_tipnode(LECoords)
midchords, chordVec, spanwiseVectors, sweepAng, pretwistDist = LiftingLine.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip; appendageOptions=appendageOptions, appendageParams=appendageParams)

# Here's a twist distribution
displacementsCol[5, :] .= vcat(LinRange(deg2rad(10.0), deg2rad(0.0), LiftingLine.NPT_WING ÷ 2), LinRange(deg2rad(0.0), deg2rad(10.0), LiftingLine.NPT_WING ÷ 2))

# Vertically displace them
dist = 0.1
displacementsCol[3, :] .= vcat(LinRange(dist, 0.0, LiftingLine.NPT_WING ÷ 2), LinRange(0.0, dist, LiftingLine.NPT_WING ÷ 2))

α0 = appendageParams["alfa0"]
β0 = appendageParams["beta"]
rake = appendageParams["rake"]
depth0 = appendageParams["depth0"]
airfoilXY, airfoilCtrlXY, npt_wing, npt_airfoil, rootChord, TR, Uvec, options = LiftingLine.initialize_LL(α0, β0, rake, sweepAng, chordVec, depth0, appendageOptions, solverOptions)
LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, displacementsCol;
    npt_wing=npt_wing,
    npt_airfoil=npt_airfoil,
    rhof=solverOptions["rhof"],
    # airfoilCoordFile=airfoilCoordFile,
    airfoil_ctrl_xy=airfoilCtrlXY,
    airfoil_xy=airfoilXY,
    options=options,
)
TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

LLNLParams = LiftingLine.LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
Forces, gammaDist, cla, cl, Forces, CL, CDi, CS = LiftingLine.compute_solution(FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences)
plot(LLMesh.collocationPts[2, :], cl)
p2 = plot(LLMesh.collocationPts[2, :], gammaDist)
p3 = plot(LLMesh.collocationPts[2, :], LLMesh.collocationPts[3, :])
p3 = plot(LLMesh.collocationPts[2, :], displacementsCol[5, :])
plot(p2, p3, layout=(2, 1))

println("CDi:\t", CDi)

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
LLMesh, FlowCond, LLHydro, Airfoils, AirfoilInfluences = LiftingLine.setup(Uvec, sweepAng, rootChord, TR, midchords, zeros(6, LiftingLine.NPT_WING);
    npt_wing=npt_wing,
    npt_airfoil=npt_airfoil,
    rhof=solverOptions["rhof"],
    # airfoilCoordFile=airfoilCoordFile,
    airfoil_ctrl_xy=airfoilCtrlXY,
    airfoil_xy=airfoilXY,
    options=options,
)
TV_influence = LiftingLine.compute_TVinfluences(FlowCond, LLMesh)

LLNLParams = LiftingLine.LiftingLineNLParams(TV_influence, LLMesh, LLHydro, FlowCond, Airfoils, AirfoilInfluences)
dfdg = LiftingLine.compute_∂I∂G(gconv, LLMesh, FlowCond, LLNLParams, solverOptions)
dfdg_FD = LiftingLine.compute_∂I∂G(gconv, LLMesh, FlowCond, LLNLParams, solverOptions; mode="FiDi")


# smaller gconv
gconv = vec([0.0061781 0.01608609 0.02289767 0.02758199 0.03094478 0.03337678 0.03501358 0.03592766 0.03627758 0.03633065 0.03633065 0.03627758 0.03592766 0.03501358 0.03337678 0.03094478 0.02758199 0.02289767 0.01608609 0.0061781])
# GOOD
drdx, drdxdisp = LiftingLine.compute_∂r∂Xpt(gconv, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
drdx_FD, drdxdisp_FD = LiftingLine.compute_∂r∂Xpt(gconv, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")
# @time LiftingLine.compute_∂r∂Xpt(gconv, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")

# GOOD
dfdxpt, dfdxdispl = LiftingLine.compute_∂I∂Xpt(gconv, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
dfdxpt_fd, dfdxdispl_fd = LiftingLine.compute_∂I∂Xpt(gconv, ptVec, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

# GOOD
∂Drag∂Xpt, ∂Drag∂xdispl, ∂Drag∂G = LiftingLine.compute_∂EmpiricalDrag(ptVec, gconv, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FAD")
∂Drag∂Xpt, ∂Drag∂xdispl, ∂Drag∂G = LiftingLine.compute_∂EmpiricalDrag(ptVec, gconv, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="RAD")
∂Drag∂Xpt_fd, ∂Drag∂xdispl_fd, ∂Drag∂G_fd = LiftingLine.compute_∂EmpiricalDrag(ptVec, gconv, nodeConn, displacementsCol, appendageParams, appendageOptions, solverOptions; mode="FiDi")

# These derivatives are good
# @time dcolldXpt = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")
# dcolldXpt = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="RAD")
# @time dcolldXpt = LiftingLine.compute_∂collocationPt∂Xpt(ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")
cla = LiftingLine.compute_liftslopes(gconv, FlowCond, LLMesh, LLHydro, Airfoils, AirfoilInfluences)

# ans1, ans2 = LiftingLine.compute_∂EmpiricalDrag(ptVec, gconv, nodeConn, appendageParams, appendageOptions, solverOptions; mode="RAD")
# @time ans1fad, ans2fad = LiftingLine.compute_∂EmpiricalDrag(ptVec, gconv, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")

allStructStates = [0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 8.05204097e-09 2.07663899e-07 0.00000000e+00 0.00000000e+00 3.41657011e-06 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 3.81207259e-08 5.18382685e-07 0.00000000e+00 0.00000000e+00 4.29963911e-06 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 9.94491169e-08 9.64236765e-07 0.00000000e+00 0.00000000e+00 6.52539579e-06 0.00000000e+00 0.00000000e+00 0.00000000e+00 0.00000000e+00 2.06784693e-07 1.62203281e-06 0.00000000e+00 0.00000000e+00 8.36014003e-06 0.00000000e+00 0.00000000e+00]
fu = zeros(length(allStructStates))
fu[end-5] = 1.0
LECoords, TECoords = FEMMethods.repack_coords(ptVec, 3, length(ptVec) ÷ 3)
∂r∂Xpt, drdxParams = FEMMethods.compute_∂r∂x(allStructStates, fu, [appendageParams], LECoords, TECoords, nodeConn; mode="analytic", appendageOptions=appendageOptions, solverOptions=solverOptions)
∂r∂Xpt, drdxParamsFD = FEMMethods.compute_∂r∂x(allStructStates, fu, [appendageParams], LECoords, TECoords, nodeConn; mode="FiDi", appendageOptions=appendageOptions, solverOptions=solverOptions)
# ==============================================================================
#                         OpenMDAO executables
# ==============================================================================

prob = om.Problem()
comp = make_component(OMLiftingLine(nodeConn, appendageParams, appendageOptions, solverOptions))

model = om.Group()
model.add_subsystem("liftingline", comp)

prob = om.Problem(model)

prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP")
# TODO: Install pyoptsparse!!
# prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
outputDir = "output"

prob.setup()


prob.set_val("liftingline.ptVec", ptVec)
prob.set_val("liftingline.gammas", zeros(AbstractFloat, 100))

@time prob.run_model()
@time prob.run_model()

println(prob.get_val("liftingline.gammas"))


# ==============================================================================
#                         Testing flutter
# ==============================================================================

using .SolveFlutter
claVec = vec([0.62203061 1.76578511 2.72440491 3.48331132 4.05915881 4.48205682 4.78369524 4.99179699 5.12856967 5.21100599 5.25185966 5.26075176 5.24519846 5.21150891 5.16553872 5.11324235 5.06088288 5.01468345 4.98003015 4.96115104 4.96115104 4.98003015 5.01468345 5.06088288 5.11324235 5.16553872 5.21150891 5.24519846 5.26075176 5.25185966 5.21100599 5.12856967 4.99179699 4.78369524 4.48205682 4.05915881 3.48331132 2.72440491 1.76578511 0.62203061])
mesh = [[0.0 0.0 0.0]
    [0.0 0.08325 0.0]
    [0.0 0.1665 0.0]
    [0.0 0.24975 0.0]
    [0.0 0.333 0.0]
    [0.0 -0.08325 0.0]
    [0.0 -0.1665 0.0]
    [0.0 -0.24975 0.0]
    [0.0 -0.333 0.0]]
elemConn = [[1 2]
    [2 3]
    [3 4]
    [4 5]
    [1 6]
    [6 7]
    [7 8]
    [8 9]]
# TODO: GPICKUP HERE GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
claVecMod = claVec[end-4:end] .+ 1.1
SolveFlutter.cost_funcsFromDVsOM(ptVec, nodeConn, displacementsCol, claVecMod, appendageParams["theta_f"], appendageParams["toc"], appendageParams["alfa0"], appendageParams, solverOptions)
evalFuncsSensList = ["ksflutter"]
dfdx_rad = SolveFlutter.evalFuncsSens(evalFuncsSensList, appendageParams, GridStruct, displacementsCol, claVecMod, solverOptions; mode="RAD")
dfdx_fd = SolveFlutter.evalFuncsSens(evalFuncsSensList, appendageParams, GridStruct, displacementsCol, claVecMod, solverOptions; mode="FiDi")