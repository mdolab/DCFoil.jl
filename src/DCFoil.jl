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
    costFuncsDict = Dict()

    # ---------------------------
    #   Mesh generation
    # ---------------------------
    FOIL = InitModel.init_static(DVDict, solverOptions)
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL; config=solverOptions["config"])
    # --- Write mesh to tecplot for later visualization ---
    tecplotIO.write_mesh(structMesh, outputDir, "mesh.dat")

    # ==============================================================================
    #                         Static hydroelastic solution
    # ==============================================================================
    if solverOptions["run_static"]
        STATSOL = SolveStatic.solve(structMesh, elemConn, DVDict, evalFuncs, solverOptions)
        # costFuncsDict = merge(costFuncs, staticCostFuncs)
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
        FLUTTERSOL = SolveFlutter.solve(structMesh, elemConn, DVDict, solverOptions)
        flutterCostFuncsDict = compute_costFuncs(FLUTTERSOL, evalFuncs, solverOptions)
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
            # TODO: doesn't work but ok for now while I sort out sensitivities
            # flutterEvalFuncs = SolveFlutter.evalFuncs(x, SOL, ρKS)
            flutterEvalFuncs, _ = SolveFlutter.postprocess_damping(SOL.N_MAX_Q_ITER, SOL.flowHistory, SOL.NTotalModesFound, SOL.nFlow, SOL.eigs_r, SOL.iblank, ρKS)
            evalFuncsDict[k] = flutterEvalFuncs
        else
            println("Unsupported cost function: ", k)
        end
    end
    return evalFuncsDict
end # compute_costFuncs

function compute_funcSens(DVDict, evalFuncs;
    # --- Optional args ---
    mode = "FiDi",
    solverOptions=Dict())
    # ---------------------------
    #   Mesh generation
    # ---------------------------
    FOIL = InitModel.init_static(DVDict, solverOptions)
    nElem = FOIL.nNodes - 1
    structMesh, elemConn = FEMMethods.make_mesh(nElem, FOIL; config=solverOptions["config"])
    # --- Write mesh to tecplot for later visualization ---

    # ---------------------------
    #   Cost functions
    # ---------------------------
    costFuncsSensDict = Dict()

    # ==============================================================================
    #                         Flutter solution
    # ==============================================================================
    if solverOptions["run_flutter"]
        costFuncsSensDict = SolveFlutter.evalFuncsSens(structMesh, elemConn, DVDict, solverOptions, evalFuncs; mode=mode)
    end

    return costFuncsSensDict
end
# function compute_jacobian(partials::Dict, evalFuncs; method="adjoint")
#     """
#     Evaluate the sensitivity of the cost functions in evalFuncs

#     Inputs
#     ------
#     partials : Dict()
#         Dictionary containing the partial derivatives of the total derivative equation
#             df/dx = ∂f/∂x + ∂f/∂u * du/dx
#         in the context of adjoint or direct methods
#     evalFuncs : Array{String}
#         List of what cost functions to evaluate
#     method : String
#         Method to use to compute the sensitivity
#             "adjoint" : Use the adjoint method
#             "direct" : Use the direct method
#     Outputs
#     -------
#     funcsSens : Dict()
#         Dictionary containing the sensitivity of the cost functions
#     """

#     funcsSens = Dict()

#     for func in evalFuncs

#         ∂f∂x = partials["∂f∂x"][func]

#         stateSens = zeros(size(∂f∂x)) # initialize -1 * (∂f/∂u * du/dx)
#         if method == "adjoint"
#             ∂r∂x = partials["∂r∂x"][func]
#             ψ = partials["ψ"][func] # adjoint vector
#             stateSens = transpose(ψ) * ∂r∂x
#         elseif method == "direct"
#             ∂f∂u = partials["∂f∂u"][func]
#             ϕ = partials["ϕ"][func] # direct vector
#             stateSens = ∂f∂u * ϕ
#         end

#         # Compute sensitivity
#         funcsSens[func] = ∂f∂x - stateSens
#     end

#     return funcsSens
# end # compute_jacobian

end # module