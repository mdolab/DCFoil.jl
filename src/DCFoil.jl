# --- Julia ---
"""
@File    :   DCFoil.jl
@Time    :   2022/07/23
@Author  :   Galen Ng
@Desc    :   This is a required "gluing" file so the package 
             management works properly. This is the module imported
"""
module DCFoil



# --- Libraries ---
include("./solvers/SolveStatic.jl")
include("./solvers/SolveForced.jl")
include("./solvers/SolveFlutter.jl")
include("./io/TecplotIO.jl")
include("./InitModel.jl")
include("./struct/FEMMethods.jl")

using JSON
using Printf
using .InitModel
using .TecplotIO
using .FEMMethods
using .SolveStatic, .SolveForced, .SolveFlutter

# --- Public functions ---
# Don't need 'DCFoil.' prefix when importing
export run_model, evalFuncs, evalFuncsSens

# ==============================================================================
#                         API functions
# ==============================================================================
function init_model(DVDict, evalFuncs; solverOptions=Dict())
    """
    Things that need to be done before running the model.
    Global vars are used in run_model
    """
    # ---------------------------
    #   Default options
    # ---------------------------
    global outputDir = solverOptions["outputDir"]

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
    global costFuncsDict = Dict()

    # ---------------------------
    #   Mesh generation
    # ---------------------------
    global FOIL, STRUT = InitModel.init_model_wrapper(DVDict, solverOptions)
    nElem = FOIL.nNodes - 1
    if solverOptions["config"] == "wing"
        nElStrut = 0
    elseif solverOptions["config"] == "t-foil"
        nElStrut = STRUT.nNodes - 1
    end
    global structMesh, elemConn = FEMMethods.make_mesh(nElem, DVDict["s"];
        config=solverOptions["config"],
        nElStrut=nElStrut,
        spanStrut=DVDict["s_strut"],
        rotation=solverOptions["rotation"]
    )

    global FEMESH = FEMMethods.FEMESH(structMesh, elemConn, DVDict["c"], DVDict["toc"], DVDict["ab"], DVDict["x_αb"], DVDict["θ"], zeros(10,2))

    # --- Write mesh to tecplot for later visualization ---
    TecplotIO.write_mesh(DVDict, FEMESH, solverOptions, outputDir, "mesh.dat")
    if solverOptions["debug"]
        open(outputDir * "elemConn.txt", "w") do io
            for iElem in 1:length(elemConn[:, 1])
                write(io, @sprintf("%03d\t%03d\n", elemConn[iElem, 1], elemConn[iElem, 2]))
            end
        end
    end
end

function run_model(DVDict, evalFuncs; solverOptions=Dict())
    """
    Runs the model but does not return anything.
    The solution structures hang around as global variables.
    """
    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        global STATSOL = SolveStatic.solve(FEMESH, DVDict, evalFuncs, solverOptions)
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
        structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes = SolveFlutter.solve_frequencies(structMesh, elemConn, DVDict, solverOptions)
        if solverOptions["writeTecplotSolution"]
            SolveFlutter.write_tecplot_natural(DVDict, structNatFreqs, structModeShapes, wetNatFreqs, wetModeShapes, structMesh, outputDir)
        end
    end
    if solverOptions["run_flutter"]
        global FLUTTERSOL = SolveFlutter.get_sol(DVDict, solverOptions)
        if solverOptions["writeTecplotSolution"]
            SolveFlutter.write_tecplot(DVDict, FLUTTERSOL, structMesh, outputDir)
        end
    end
end

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