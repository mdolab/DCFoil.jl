# --- Julia ---

# @File    :   main.jl
# @Time    :   2022/06/16
# @Author  :   Galen Ng
# @Desc    :   Main executable for the project

using Printf#, Dates

for headerName in [
    "../../src/solvers/SolveFlutter"
]
    include("$(headerName).jl")
end

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
run_static = true
# run_forced = true
run_modal = true
# run_flutter = true
# debug = true
# tipMass = true

# ************************************************
#     DV Dictionaries (see INPUT directory)
# ************************************************
nNodes = 10 # spatial nodes
nNodesStrut = 10 # spatial nodes
nModes = 4 # number of modes to solve for;
# NOTE: this is the number of starting modes you will solve for, but you will pick up more as you sweep velocity
# This is because poles bifurcate
# nModes is really the starting number of structural modes you want to solve for
df = 1
fRange = [0.0, 1000.0] # forcing and search frequency sweep [Hz]
uRange = [20.0, 40.0] # flow speed [m/s] sweep for flutter
tipForceMag = 0.5 * 0.5 * 1000 * 100 * 0.03 # tip harmonic forcing

# ************************************************
#     Setup solver options
# ************************************************
# Anything in DVDict is what we calculate derivatives wrt
DVDict = Dict(
    "alfa0" => 6.0, # initial angle of attack [deg]
    "sweep" => deg2rad(0.0), # sweep angle [rad]
    "zeta" => 0.04, # modal damping ratio at first 2 modes
    "ab" => 0 * ones(nNodes), # dist from midchord to EA [m]
    "toc" => 0.10 * ones(nNodes), # thickness-to-chord ratio
    "x_ab" => 0 * ones(nNodes), # static imbalance [m]
    "theta_f" => deg2rad(-15), # fiber angle global [rad]
    "rake" => 0.0,
    # --- Strut vars ---
    "beta" => 0.0, # yaw angle wrt flow [deg]
    "s_strut" => 0.4, # from Yingqian
    "c_strut" => 0.14 * ones(nNodesStrut), # chord length [m]
    "toc_strut" => 0.095 * ones(nNodesStrut), # thickness-to-chord ratio (max t/c if using airfoil correction)
    "ab_strut" => 0 * ones(nNodesStrut), # dist from midchord to EA [m]
    "x_ab_strut" => 0 * ones(nNodesStrut), # static imbalance [m]
    "theta_f_strut" => deg2rad(0), # fiber angle global [rad]
    "depth0" => 10.0, # depth of the strut [m]
)
paramsList = [DVDict]
wingOptions = Dict(
    "compName" => "ward2018",
    "config" => "wing",
    "nNodes" => nNodes,
    "nNodeStrut" => 10,
    "material" => "pvc", # preselect from material library
    "use_tipMass" => tipMass,
    "xMount" => 0.0,
    "path_to_struct_props" => "$(@__DIR__)/ward2018_structprops.json",
    "path_to_geom_props" => "$(@__DIR__)/ward2018_structprops.json",
)
appendageOptions = [wingOptions]
solverOptions = Dict(
    # ---------------------------
    #   I/O
    # ---------------------------
    "name" => "ward2018",
    "debug" => debug,
    "writeTecplotSolution" => true,
    "gridFile" => ["$(@__DIR__)/ward2018_foil_stbd_mesh.dcf"],
    # ---------------------------
    #   General appendage options
    # ---------------------------
    "appendageList" => appendageOptions,
    "config" => "wing",
    "nNodes" => nNodes,
    "nNodeStrut" => 10,
    "rotation" => 0.0, # deg
    "gravityVector" => [0.0, 0.0, -9.81],
    "use_tipMass" => tipMass,
    # ---------------------------
    #   Flow
    # ---------------------------
    "Uinf" => 5.0, # free stream velocity [m/s]
    "rhof" => 1000.0, # fluid density [kg/m³]
    "material" => "cfrp", # preselect from material library
    "use_freeSurface" => false,
    "use_cavitation" => false,
    "use_ventilation" => false,
    "use_nlll" => true,
    # --- Static solve ---
    "run_static" => run_static,
    # --- Forced solve ---
    "run_forced" => run_forced,
    "fRange" => fRange,
    "tipForceMag" => tipForceMag,
    # --- Eigen solve ---
    "run_modal" => run_modal,
    "run_flutter" => run_flutter,
    "nModes" => nModes,
    "uRange" => uRange,
    "maxQIter" => 500,
    "rhoKS" => 100.0,
)

# ************************************************
#     Cost functions
# ************************************************
evalFuncs = ["wtip", "psitip", "cl", "cmy", "lift", "moment", "ksflutter"]

# ************************************************
#     I/O
# ************************************************
# The file directory has the convention:
# <name>_<material-name>_f<fiber-angle>_w<sweep-angle>
# But we write the DVDict to a human readable file in the directory anyway so you can double check
outputDir = @sprintf("./OUTPUT/%s_%s_f%.1f_w%.1f/",
    # string(Dates.today()),
    solverOptions["name"],
    solverOptions["material"],
    rad2deg(DVDict["theta_f"]),
    rad2deg(DVDict["sweep"]))
mkpath(outputDir)

solverOptions["outputDir"] = outputDir

# ==============================================================================
#                         Call DCFoil
# ==============================================================================
GridStruct = SolveFlutter.add_meshfiles(solverOptions["gridFile"], Dict("junction-first" => true))
LECoords, nodeConn, TECoords = GridStruct.LEMesh, GridStruct.nodeConn, GridStruct.TEMesh
ptVec, m, n = SolveFlutter.FEMMethods.unpack_coords(LECoords, TECoords)
solverOptions = SolveFlutter.FEMMethods.set_structDamping(ptVec, nodeConn, paramsList[1], solverOptions, appendageOptions[1])

# --- Pre-computes to get lift slopes ---
displacements_col = zeros(6, SolveFlutter.LiftingLine.NPT_WING)
idxTip = SolveFlutter.FEMMethods.get_tipnode(LECoords)
midchords, chordVec, _, sweepAng, _ = SolveFlutter.FEMMethods.compute_1DPropsFromGrid(LECoords, TECoords, nodeConn, idxTip, appendageOptions=appendageOptions[1], appendageParams=paramsList[1])
LLOutputs, LLSystem, FlowCond = SolveFlutter.HydroStrip.compute_hydroLLProperties(midchords, chordVec, sweepAng; appendageParams=paramsList[1], solverOptions=solverOptions, appendageOptions=appendageOptions[1])
claVec = LLOutputs.cla

# --- Do tip loads ---
globalK, globalM, _, DOFBlankingList, FEMESH, _, _ = SolveFlutter.FEMMethods.setup_FEBeamFromCoords(LECoords, nodeConn, TECoords, paramsList, wingOptions, solverOptions)

angleDefault = deg2rad(-90) # default angle of rotation of the axes to match beam
axisDefault = "z"
T = FEMMethods.get_rotate3dMat(angleDefault, axisDefault)
# T = I(3)
transMatL2G = [
    T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
    zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3)
    zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3) zeros(3, 3)
    zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3) zeros(3, 3)
    zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T zeros(3, 3)
    zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) zeros(3, 3) T
]

tipForces = collect(-600.0:50.0:600)
outdata = zeros(length(tipForces))
for (ii, tipForce) in enumerate(tipForces)
    traction_forces = zeros(size(globalK, 1))
    FEMMethods.apply_tip_load!(traction_forces, "COMP2", transMatL2G, "force")
    idxForce = argmax(traction_forces)
    traction_forces[idxForce] *= tipForce

    qSol = FEMMethods.solve_structure(
        globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]],
        traction_forces[1:end.∉[DOFBlankingList]],
    )
    uSol, _ = FEMMethods.put_BC_back(qSol, FEMMethods.ELEMTYPE; appendageOptions=wingOptions)

    tipBend = uSol[end-FEMMethods.NDOF+FEMMethods.WIND]
    tipTwist = uSol[end-FEMMethods.NDOF+FEMMethods.ΘIND]
    println("tip force: ", tipForce)
    println("tip deflection: ", tipBend, " [m]")
    println("tip twist: ", tipTwist, " [rad] $(rad2deg(tipTwist)) [deg]")

    outdata[ii] = tipBend
end

tipTorques = collect(-200.0:20.0:200) # N-m
outTwist = zeros(length(tipTorques))
for (ii, tipTorque) in enumerate(tipTorques)
    traction_forces = zeros(size(globalK, 1))
    FEMMethods.apply_tip_load!(traction_forces, "COMP2", transMatL2G, "torque")
    traction_forces *= tipTorque
    qSol = FEMMethods.solve_structure(
        globalK[1:end.∉[DOFBlankingList], 1:end.∉[DOFBlankingList]],
        traction_forces[1:end.∉[DOFBlankingList]],
    )
    uSol, _ = FEMMethods.put_BC_back(qSol, FEMMethods.ELEMTYPE; appendageOptions=wingOptions)

    tipBend = uSol[end-FEMMethods.NDOF+FEMMethods.WIND]
    tipTwist = uSol[end-FEMMethods.NDOF+FEMMethods.ΘIND]
    println("tip torque: ", tipTorque)
    println("tip deflection: ", tipBend, " [m]")
    println("tip twist: ", tipTwist, " [rad] $(rad2deg(tipTwist)) [deg]")
    outTwist[ii] = tipTwist
end

using JLD2
save("./ward2018_tiploads.jld2", "tipForces", tipForces, "tipTorques", tipTorques, "tipBend", outdata, "tipTwist", outTwist)

# --- Do modal analysis ---
SolveFlutter.solve_frequencies(LECoords, TECoords, nodeConn, paramsList[1], solverOptions, wingOptions)

using Test

ref_val = 0.0
@testset "test ward2018" begin

    @test abs(obj - ref_val) < 1e-2
end

