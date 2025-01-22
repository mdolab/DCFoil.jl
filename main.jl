# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Desc    :   Main executable for running DCFoil

using Printf, Dates, Profile
# using Debugger: @run


# This is the way to import it manually in dev mode
include("src/DCFoil.jl")
using .DCFoil: RealOrComplex
# Import static package
# using DCFoil

# ==============================================================================
# Setup hydrofoil model and solver settings
# ==============================================================================
# ************************************************
#     Task type
# ************************************************
# Set task you want to true
# Defaults
run = true # run the solver for a single point
run_static = false
run_forced = false
run_modal = false
run_flutter = false
debug = false
tipMass = false

# Uncomment here
# run_static = true
run_forced = true
# run_modal = true
# run_flutter = true
# debug = true
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 10 # spatial nodes
nNodesStrut = 5 # spatial nodes
nModes = 4 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
fSweep = [1e-4, 10.0] # forcing frequency sweep [Hz]
# fSweep = [1e-4, 10.0] # forcing frequency sweep [Hz] SMALLER TEST
uRange = [20.0, 60.0] / 1.9438 # flow speed [m/s] sweep for flutter
# uRange = [170.0, 190.0] # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# ************************************************
#     Setup solver options
# ************************************************
# AC Rudder
# paramsRudder = Dict(
#     "alfa0" => 0.0, # initial angle of attack [deg]
#     "sweep" => deg2rad(0.0), # sweep angle [rad]
#     "zeta" => 0.04, # modal damping ratio at first 2 modes
#     "c" => ".dat", # chord length [m]
#     "ab" => ".dat", # dist from midchord to EA [m]
#     "toc" => ".dat", # thickness-to-chord ratio (mean)
#     "x_ab" => ".dat", # static imbalance [m]
#     "theta_f" => deg2rad(0), # fiber angle global [rad]
#     # --- Strut vars ---
#     "depth0" => 0.4, # submerged depth of strut [m] # from Yingqian
#     "rake" => 0.0, # rake angle about top of strut [deg]
#     "beta" => 0.0, # yaw angle wrt flow [deg]
#     "s_strut" => 2.8, # strut span [m]
#     "c_strut" => ".dat", # chord length [m]
#     "toc_strut" => ".dat", # thickness-to-chord ratio (mean)
#     "ab_strut" => ".dat", # dist from midchord to EA [m]
#     "x_ab_strut" => ".dat", # static imbalance [m]
#     "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
# )
paramsRudder = Dict(
    "alfa0" => 2.0, # initial angle of attack [deg] (angle of flow vector)
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

paramsList = [paramsRudder]

rudderOptions = Dict(
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

appendageList = [rudderOptions]

solverOptions = Dict(
    # ---------------------------
    #   I/O
    # ---------------------------
    # "name" => "R3E6",
    "name" => "mothrudder-nofs",
    "debug" => debug,
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
    # "use_freeSurface" => true,
    "use_cavitation" => false,
    "use_ventilation" => false,
    # "use_dwCorrection" => true,
    # ---------------------------
    #   Solver modes
    # ---------------------------
    # --- Static solve ---
    "run_static" => run_static,
    # "res_jacobian" => "CS",
    "res_jacobian" => "analytic",
    # "res_jacobian" => "RAD",
    # "onlyStructDerivs" => true,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fRange" => [fSweep[1], fSweep[end]], # forcing frequency sweep [Hz]
    # "df" => 0.05, # frequency step size
    "df" => 0.005, # frequency step size
    "tipForceMag" => tipForceMag,
    # --- p-k (Eigen) solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 100, # that didn't fix the slow run time...
    "rhoKS" => 500.0,
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cd", "cmy", "lift", "moment", "ksflutter", "kscl"]
evalFuncSens = [
    # "wtip",
    "cd",
    "cl", "lift", 
    "ksflutter", "kscl",
]

# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_%s_f%.1f/",
    string(Dates.today()),
    solverOptions["name"],
    rudderOptions["material"],
    rad2deg(paramsList[1]["theta_f"]),
    )
mkpath(outputDir)

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
# TODO: PICKUP HERE ADDING IN THE STRUT/ getting derivs working to do an optimization
GridStruct = DCFoil.MeshIO.add_meshfiles(solverOptions["gridFile"], Dict("junction-first" => true))
LECoords, nodeConn, TECoords = GridStruct.LEMesh, GridStruct.nodeConn, GridStruct.TEMesh
DCFoil.init_model(LECoords, nodeConn, TECoords; solverOptions=solverOptions, appendageParamsList=paramsList)
SOLDICT = DCFoil.run_model(LECoords, nodeConn, TECoords, evalFuncs; solverOptions=solverOptions, appendageParamsList=paramsList)
costFuncs = DCFoil.evalFuncs(SOLDICT, LECoords, nodeConn, TECoords, paramsList, evalFuncs, solverOptions)
costFuncsSens = DCFoil.evalFuncsSens(SOLDICT, paramsList, LECoords, nodeConn, TECoords, evalFuncSens, solverOptions;
    mode="ADJOINT",
    # mode="FiDi",
)