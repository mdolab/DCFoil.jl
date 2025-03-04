# THIS IS THE JULIA CENTRIC RUN SCRIPT

using OpenMDAO: om, make_component
include("../src/hydro/LiftingLine.jl")
include("../src/hydro/liftingline_om.jl")
include("../src/struct/beam_om.jl")
const RealOrComplex = Union{Real, Complex}
# const RealOrComplex = AbstractFloat

# ==============================================================================
#                         Setup stuff
# ==============================================================================
ptVec = [-0.07,
    0.0,
    0.0,
    -0.0675,
    0.037,
    0.0,
    -0.065,
    0.074,
    0.0,
    -0.0625,
    0.111,
    0.0,
    -0.06,
    0.148,
    0.0,
    -0.0575,
    0.185,
    0.0,
    -0.055,
    0.222,
    0.0,
    -0.0525,
    0.259,
    0.0,
    -0.05,
    0.296,
    0.0,
    -0.0475,
    0.333,
    0.0,
    -0.0675,
    -0.037,
    0.0,
    -0.065,
    -0.074,
    0.0,
    -0.0625,
    -0.111,
    0.0,
    -0.06,
    -0.148,
    0.0,
    -0.0575,
    -0.185,
    0.0,
    -0.055,
    -0.222,
    0.0,
    -0.0525,
    -0.259,
    0.0,
    -0.05,
    -0.296,
    0.0,
    -0.0475,
    -0.333,
    0.0,
    0.07,
    0.0,
    0.0,
    0.0675,
    0.037,
    0.0,
    0.065,
    0.074,
    0.0,
    0.0625,
    0.111,
    0.0,
    0.06,
    0.148,
    0.0,
    0.0575,
    0.185,
    0.0,
    0.055,
    0.222,
    0.0,
    0.0525,
    0.259,
    0.0,
    0.05,
    0.296,
    0.0,
    0.0475,
    0.333,
    0.0,
    0.0675,
    -0.037,
    0.0,
    0.065,
    -0.074,
    0.0,
    0.0625,
    -0.111,
    0.0,
    0.06,
    -0.148,
    0.0,
    0.0575,
    -0.185,
    0.0,
    0.055,
    -0.222,
    0.0,
    0.0525,
    -0.259,
    0.0,
    0.05,
    -0.296,
    0.0,
    0.0475,
    -0.333,
    0.0]
nodeConn = [1 2 3 4 5 6 7 8 9 1 11 12 13 14 15 16 17 18;
    2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19]
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
    "config" => "full-wing",
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
    # "nModes" => nModes,
    "uRange" => [1.0,30],
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 500.0,
)

# ==============================================================================
#                         Derivatives
# ==============================================================================
using .LiftingLine
using .FEMMethods
gconv = vec([0.0322038 0.03231678 0.03253403 0.03284775 0.03324736 0.03371952 0.03425337 0.03484149 0.03547895 0.03616117 0.03688162 0.03762993 0.03839081 0.03914407 0.03986539 0.04052803 0.04110476 0.04156866 0.04189266 0.04205787 0.04205787 0.04189266 0.04156866 0.04110476 0.04052803 0.03986539 0.03914407 0.03839081 0.03762993 0.03688162 0.03616117 0.03547895 0.03484149 0.03425337 0.03371952 0.03324736 0.03284775 0.03253403 0.03231678 0.0322038])

# LiftingLine.compute_∂I∂Xpt(gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FAD")
# LiftingLine.compute_∂I∂Xpt(gconv, ptVec, nodeConn, appendageParams, appendageOptions, solverOptions; mode="FiDi")

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