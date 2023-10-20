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
using .InitModel
using .tecplotIO
using .FEMMethods
using .SolveStatic, .SolveForced, .SolveFlutter

function run_model(DVDict, evalFuncs; solverOptions=Dict())
    """
    Runs the model but does not return anything.
    The solution structures hang around as global variables.
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
    nElStrut = solverOptions["nNodeStrut"] - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"];
        config=solverOptions["config"],
        nElStrut=nElStrut,
        spanStrut=DVDict["strut"],
        rotation=solverOptions["rotation"]
    )

    # --- Write mesh to tecplot for later visualization ---
    tecplotIO.write_mesh(DVDict, structMesh, outputDir, "mesh.dat")

    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        global STATSOL = SolveStatic.solve(structMesh, elemConn, DVDict, evalFuncs, solverOptions)
    end

    # ==============================================================================
    #                         Forced vibration solution
    # ==============================================================================
    if solverOptions["run_forced"]
        global forcedCostFuncs = SolveForced.solve(structMesh, elemConn, DVDict, solverOptions)
    end

    # ==============================================================================
    #                         Flutter solution
    # ==============================================================================
    if solverOptions["run_modal"]
        @time SolveFlutter.solve_frequencies(structMesh, elemConn, DVDict, solverOptions)
    end
    if solverOptions["run_flutter"]
        @time global FLUTTERSOL = SolveFlutter.get_sol(DVDict, solverOptions)
    end
end

function set_defaultOptions()
    """
    Set the default solver options
    Case sensitive
    """
    solverOptions = Dict(
        # --- I/O ---
        "debug" => false,
        "outputDir" => "./OUTPUT/",
        # --- General solver options ---
        "config" => "wing",
        "gravityVector" => [0.0, 0.0, -9.81],
        "rotation" => 0.0, # Rotation of the wing about the x-axis [deg]
        "use_tipMass" => false,
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
function evalFuncs(evalFuncs, solverOptions)
    """
    Common interface to compute cost functions

    Inputs
    ------
    evalFuncs : 1d array
        List of what cost functions to evaluate
    """
    x = 0.0 # dummy
    evalFuncsDict = Dict()

    # --- Solver cost funcs ---
    staticCostFuncs = [
        "psitip"
        "wtip"
        "lift"
        "moment"
        "cl"
        "cmy"
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

    # Assemble all possible
    allCostFuncs = vcat(staticCostFuncs, forcedCostFuncs, flutterCostFuncs)

    # ************************************************
    #     Loop over all evalFuncs
    # ************************************************
    for k in evalFuncs

        if k in staticCostFuncs
            staticEvalFuncs = SolveStatic.evalFuncs(STATSOL.structStates, STATSOL.fHydro, evalFuncs)
            evalFuncsDict[k] = staticEvalFuncs[k]

        elseif k in forcedCostFuncs
            SolveForced.evalFuncs()

        elseif k in flutterCostFuncs
            obj, _ = SolveFlutter.postprocess_damping(FLUTTERSOL.N_MAX_Q_ITER, FLUTTERSOL.flowHistory, FLUTTERSOL.NTotalModesFound, FLUTTERSOL.nFlow, FLUTTERSOL.p_r, FLUTTERSOL.iblank, solverOptions["rhoKS"])
            flutterCostFuncsDict = Dict(
                "ksflutter" => obj,
            )
            evalFuncsDict = merge(evalFuncsDict, flutterCostFuncsDict)
        else
            println("Unsupported cost function: ", k)
        end
    end

    # --- Write cost funcs to file ---
    outputDir = solverOptions["outputDir"]
    stringData = JSON.json(evalFuncsDict)
    open(outputDir * "funcs.json", "w") do io
        write(io, stringData)
    end

    return evalFuncsDict
end # evalFuncs

function evalFuncsSens(DVDict, evalFuncs, solverOptions; mode="FiDi")

    # # ---------------------------
    # #   Mesh generation
    # # ---------------------------
    # FOIL = InitModel.init_model_wrapper(DVDict, solverOptions)
    # nElem = FOIL.nNodes - 1
    # structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"]; config=solverOptions["config"])

    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsSensDict = Dict()

    # ==============================================================================
    #                         Cost functions
    # ==============================================================================
    if solverOptions["run_flutter"]
        costFuncsSensDict = SolveFlutter.evalFuncsSens(DVDict, solverOptions; mode=mode)
    end

    return costFuncsSensDict
end

end # module