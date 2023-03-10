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

function run_model(
    DVDict, evalFuncs;
    # --- Optional args ---
    solverOptions=Dict()
)
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
    costFuncs = Dict()
    costFuncSens = Dict()

    # ---------------------------
    #   Mesh generation
    # ---------------------------
    FOIL = InitModel.init_static(DVDict["nNodes"], DVDict)
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL; config=solverOptions["config"])
    # --- Write mesh to tecplot for later visualization ---
    tecplotIO.write_mesh(structMesh, outputDir, "mesh.dat")

    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        staticCostFuncs = SolveStatic.solve(structMesh, elemConn, DVDict, evalFuncs, solverOptions)
        costFuncs = merge(costFuncs, staticCostFuncs)
    end

    # ==============================================================================
    #                         Forced vibration solution
    # ==============================================================================
    if solverOptions["run_forced"]
        forcedCostFuncs = SolveForced.solve(structMesh, elemConn, DVDict, solverOptions)
        # costFuncs = merge(costFuncs, forcedCostFuncs) TODO: costFuncs
    end

    # ==============================================================================
    #                         Flutter solution
    # ==============================================================================
    if solverOptions["run_modal"]
        SolveFlutter.solve_frequencies(structMesh, elemConn, DVDict, solverOptions)
    end
    if solverOptions["run_flutter"]
        flutterCostFuncs = SolveFlutter.solve(structMesh, elemConn, DVDict, solverOptions)
        # costFuncs = merge(costFuncs, flutterCostFuncs)
    end

    return costFuncs
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
        "nModes" => 3,
        "uRange" => nothing,
    )
    return solverOptions
end # set_defaultOptions

end # module