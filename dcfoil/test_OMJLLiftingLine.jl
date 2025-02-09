# THIS IS THE JULIA CENTRIC RUN SCRIPT
# TODO: now do this whole wrapping in julia to see if it is faster

using OpenMDAO: om, make_component
include("../src/hydro/OMLiftingLine.jl")
# const RealOrComplex = Union{Real, Complex}
const RealOrComplex = AbstractFloat

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
    # "uRange" => uRange,
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 500.0,
)

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
# optOptions = Dict(
#     "Major feasibility tolerance" => 1e-4,
#     "Major optimality tolerance" => 1e-4,
#     "Difference interval" => 1e-4,
#     "Hessian full memory" => None,
#     "Function precision" => 1e-8,
#     "Print file" => outputDir * "SNOPT_print.out",
#     "Summary file" => outputDir * "SNOPT_summary.out",
#     "Verify level" => -1,  # NOTE=> verify level 0 is pretty useless; just use level 1--3 when testing a new feature
#     # "Linesearch tolerance"=> 0.99,  # all gradients are known so we can do less accurate LS
#     # "Nonderivative linesearch"=> None,  # Comment out to specify yes nonderivative (nonlinear problem)
#     # "Major Step Limit"=> 5e-3,
#     # "Major iterations limit"=> 1,  # NOTE=> for debugging; remove before runs if left active by accident
# )

prob.setup()


prob.set_val("liftingline.ptVec", ptVec)
prob.set_val("liftingline.gammas", zeros(AbstractFloat,100))

@time prob.run_model()
@time prob.run_model()

println(prob.get_val("liftingline.gammas"))