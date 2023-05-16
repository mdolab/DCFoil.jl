# --- Julia ---

# @File    :   DCFoil.jl
# @Time    :   2022/07/23
# @Author  :   Galen Ng
# @Desc    :   This is a required "gluing" file so the package management works properly

module DCFoil


# --- Public functions ---
export run_model

# --- Libraries ---
include("./solvers/SolveStatic.jl")
include("./solvers/SolveForced.jl")
include("./solvers/SolveFlutter.jl")
include("./io/tecplotIO.jl")
include("./InitModel.jl")
include("./struct/FiniteElements.jl")
using JSON
using .SolveStatic
using .SolveForced
using .SolveFlutter
using .tecplotIO
using .InitModel
using .FEMMethods

function run_model(DVDict, evalFuncs; solverOptions=Dict())
    """
    The interface into the source code
    """
    # ==============================================================================
    #                         Initializations
    # ==============================================================================
    # ---------------------------
    #   Default options
    # ---------------------------
    if isempty(solverOptions)
        solverOptions = set_defaultOptions()
    end
    outputDir = solverOptions["outputDir"]

    # ---------------------------
    #   Write DVs and options
    # ---------------------------
    stringData = JSON.json(DVDict)
    open(outputDir * "init_DVDict.json", "w") do io
        write(io, stringData)
    end
    stringData = JSON.json(solverOptions)
    open(outputDir * "solverOptions.json", "w") do io
        write(io, stringData)
    end

    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsDict = Dict()

    # ---------------------------
    #   Mesh generation
    # ---------------------------
    FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"]; config=solverOptions["config"])
    # --- Write mesh to tecplot for later visualization ---
    tecplotIO.write_mesh(structMesh, outputDir, "mesh.dat")

    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        STATSOL = SolveStatic.solve(structMesh, elemConn, DVDict, evalFuncs, solverOptions)
        costFuncsDict = SolveStatic.evalFuncs(STATSOL.structStates, STATSOL.fHydro, evalFuncs)
    end

    # ==============================================================================
    #                         Forced vibration solution
    # ==============================================================================
    if solverOptions["run_forced"]
        forcedCostFuncs = SolveForced.solve(structMesh, elemConn, DVDict, solverOptions)
    end

    # ==============================================================================
    #                         Flutter solution
    # ==============================================================================
    if solverOptions["run_modal"]
        SolveFlutter.solve_frequencies(structMesh, elemConn, DVDict, solverOptions)
    end
    if solverOptions["run_flutter"]
        obj = SolveFlutter.evalFuncs(DVDict, solverOptions)
        flutterCostFuncsDict = Dict(
            "ksflutter" => obj,
            # "lockin" => obj.lockin,
            # "gap" => obj.gap
        )
        costFuncsDict = merge(costFuncsDict, flutterCostFuncsDict)
    end

    return costFuncsDict
end

function set_defaultOptions()
    """
    Set the default solver options
    Case sensitive
    TODO: maybe move this to a defaultOptions file
    """
    solverOptions = Dict(
        # --- I/O ---
        "debug" => false,
        "outputDir" => "./OUTPUT/",
        # --- General solver options ---
        "config" => "wing",
        "gravityVector" => [0.0, 0.0, -9.81],
        "rotation" => 0.0, # Rotation of the wing about the x-axis [deg]
        "tipMass" => false,
        "use_cavitation" => false,
        "use_freeSurface" => false,
        "use_ventilation" => false,
        # --- Static solve ---
        "run_static" => false,
        # --- Forced solve ---
        "run_forced" => false,
        "fSweep" => 0:0.1:10,
        "tipForceMag" => 0.0,
        # --- Eigen solve ---
        "run_modal" => false,
        "run_flutter" => false,
        "nModes" => 3, # Number of struct modes to solve for (starting)
        "uRange" => nothing, # Range of velocities to sweep
        "maxQIter" => 200, #max dyn pressure iters
    )
    return solverOptions
end # set_defaultOptions

# ==============================================================================
#                         Cost func and sensitivity routines
# ==============================================================================
function compute_costFuncs(SOL, evalFuncs, solverOptions)
    """
    Common interface to compute cost functions

    Inputs
    ------
    sol : Dict()
        Dictionary containing solution data
    evalFuncs : 1d array
        List of what cost functions to evaluate
    """
    x = 0.0 # dummy
    evalFuncsDict = Dict()

    # --- Solver cost funcs ---
    staticCostFuncs = [
    # "psitip"
    # "wtip"
    # "lift"
    # "moment"
    # "cl"
    # "cmy"
    ]
    forcedCostFuncs = [
        "peakpsitip" # maximum deformation amplitude (abs val) across forced frequency sweep
        "peakwtip"
        "vibareapsi" # integrated deformations under the spectral curve (see Ng et al. 2022)
        "vibareaw"
    ]
    flutterCostFuncs = [
        "ksflutter" # flutter value (damping)
        "lockin" # lock-in value
        "gap" # mode gap width
    ]

    # # Assemble all possible
    # allCostFuncs = hcat(staticCostFuncs, forcedCostFuncs, flutterCostFuncs)

    # Loop over all evalFuncs
    for k in evalFuncs

        if k in staticCostFuncs
            staticEvalFuncs = SolveStatic.evalFuncs(states, forces, k)
            evalFuncsDict[k] = staticEvalFuncs[k]
        elseif k in forcedCostFuncs
            SolveForced.evalFuncs()
        elseif k in flutterCostFuncs
            # Unpack solver data
            ρKS = solverOptions["rhoKS"]
            # Get flutter evalFunc and stick into solver evalFuncs
            # flutterEvalFuncs = SolveFlutter.evalFuncs(x, SOL, ρKS)
            flutterEvalFuncs, _ = SolveFlutter.postprocess_damping(SOL.N_MAX_Q_ITER, SOL.flowHistory, SOL.NTotalModesFound, SOL.nFlow, SOL.eigs_r, SOL.iblank, ρKS)
            evalFuncsDict[k] = flutterEvalFuncs
        else
            println("Unsupported cost function: ", k)
        end
    end
    return evalFuncsDict
end # compute_costFuncs

function compute_funcSens(SOL, DVDict, evalFuncs;
    # --- Optional args ---
    mode="FiDi",
    solverOptions=Dict())
    # ---------------------------
    #   Mesh generation
    # ---------------------------
    FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"]; config=solverOptions["config"])
    # --- Write mesh to tecplot for later visualization ---

    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsSensDict = Dict()

    # ==============================================================================
    #                         Flutter solution
    # ==============================================================================
    if solverOptions["run_flutter"]
        costFuncsSensDict = SolveFlutter.evalFuncsSens(SOL, structMesh, elemConn, DVDict, solverOptions, evalFuncs; mode=mode)
    end

    return costFuncsSensDict
end

end # module